import sys
import pickle as pkl
import zmq
import numpy as np
import time
from pytweezer.analysis import analysis as an
from pytweezer import phasemask as pm
from pytweezer import communication as comm
from pytweezer.drivers.imagemX2 import ImagEMX2Camera, ImagEMX2CameraClient
import cupy as cp
import lap
import asyncio
import threading
import queue

import sum_pixel_values_cpp as sum_cpp

update_state_kernel = cp.ElementwiseKernel(
    'float32 dw, float32 dphi, float32 dx, float32 dy, float32 ds',
    'float32 w, float32 phi, float32 x, float32 y',
    '''
    w += dw;
    phi += dphi;
    x += dx * ds;
    y += dy * ds;
    ''',
    'update_hologram_state'
    )

def slm_upload_proc(SLM, frame_queue):
    """
    Runs in a separate background thread. Pulls frames from the GPU to RAM 
    and uploads them to the SLM as soon as they are ready.
    """
    while True:
        # This blocks until a new frame is available in the buffer
        gpu_frame = frame_queue.get()
        
        if gpu_frame is None:  # Sentinel value indicating the sequence is finished
            frame_queue.task_done()
            break
            
        # .get() pulls the frame from GPU VRAM to CPU RAM. 
        # Doing this here offloads the PCIe transfer delay from the main calculation loop!
        frame = gpu_frame.get()
        
        # Replace this with your actual SLM upload hardware command
        SLM.update_mask(frame)  
        
        # Mark the task as finished so the queue knows it's ready for the next step
        frame_queue.task_done()

def get_jv_pairing_lap(init, final):
        """
        Solves using the 'lap' library (C++ Jonker-Volgenant implementation).
        Computes the cost matrix and padding efficiently on the GPU via CuPy.
        """
        N = len(init)
        M = len(final)
        
        # 1. Calculate Cost Matrix directly on GPU using CuPy broadcasting
        # init shape: (N, 2), final shape: (M, 2) -> cost_matrix shape: (N, M)
        cost_matrix = cp.sum((init[:, None, :] - final[None, :, :])**2, axis=-1)
        
        # 2. Handle Rectangularity (N != M) via padding
        if N != M:
            dim = max(N, M)
            # Create a large square matrix on the GPU filled with a high cost
            large_cost = float(cost_matrix.max() * 1000.0) if cost_matrix.size > 0 else 1.0
            padded_cost = cp.full((dim, dim), large_cost, dtype=cp.float32)
            
            # Fill in the real data
            padded_cost[:N, :M] = cost_matrix
            
            # TRANSFER TO CPU FOR C++ LAPJV (lap strictly requires numpy)
            padded_cost_cpu = padded_cost.get()
            opt_cost, x, y = lap.lapjv(padded_cost_cpu, extend_cost=True)
            
            # 3. Extract valid indices and map back to CuPy
            if N < M:
                init_idx = cp.arange(N)
                final_idx = cp.asarray(x[:N])
                
                valid_mask = final_idx < M
                init_idx = init_idx[valid_mask]
                final_idx = final_idx[valid_mask]
            else:
                final_idx = cp.arange(M)
                init_idx = cp.asarray(y[:M])
                
                valid_mask = init_idx < N
                final_idx = final_idx[valid_mask]
                init_idx = init_idx[valid_mask]
                
        else:
            # Square case is simple
            cost_matrix_cpu = cost_matrix.get()
            opt_cost, x, y = lap.lapjv(cost_matrix_cpu, extend_cost=True)
            init_idx = cp.arange(N)
            final_idx = cp.asarray(x)
            
        return init_idx, final_idx

def run_server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    port = 2222
    socket.bind(f"tcp://0.0.0.0:{port}")
    socket.setsockopt(zmq.RCVTIMEO, 1000)
    
    print(f"--- Rearrangement Node Listening on port {port} ---")
    print("Press Ctrl+C to shut down the server safely.")

    state = 0
    
    try:
        while True:
            try:
                parts = socket.recv_multipart()
            except zmq.error.Again:
                continue
            
            # Frame 0 is always our JSON header
            header = pkl.loads(parts[0])
            command = header.get("cmd")
            print(f"[Rearrangement Node] Received command: {command}")
            
            try:
                if command == "INITIALISE":
                    print("[Rearrangement Node] Initialising rearrangement node...")

                    dtype1 = np.dtype(header["dtype1"])
                    dtype2 = np.dtype(header["dtype2"])
                    shape1 = header["shape1"]
                    shape2 = header["shape2"]
                    arr_shape1 = header["array_shape1"]
                    arr_shape2 = header["array_shape2"]
                    d0 = header["d0"]
                    fps = header["fps"]
                    threshold = header["threshold"]
                    grid_positions = header["grid_positions"]
                    roi = header["roi"]

                    PM = pm.OptimisationBasedPhasemaskGeneratorGPU(
                                            wavelength_um=0.852,
                                            focal_length_mm=17.3,
                                            slm_pitch_um=17,
                                            slm_res=(1024,1024),
                                            input_beam_waist_mm=16,
                                            fresnel_f_mm=1072,
                                            blaze_dx_dy_um=(48, -4),
                                            zernike_coeff_dict={5:1.195, 6:0.725, 7:0.970, 8:0.478, 9:-1.091, 10:0.303, 11:0.021, 12:0.072, 13:0.049})
                    print("[Rearrangement Node] Phasemask generator initialized.")

                    SLM = comm.SLMClient()
                    print("[Rearrangement Node] Connected to SLM server.")

                    if state == 1:
                        print("[Rearrangement Node] Camera was already initialised.")
                    else:
                        try:
                            camera = ImagEMX2Camera()
                            camera.setup_acquisition("snap", 1)
                            camera.set_trigger_source("ext")
                            camera.set_external_exposure_mode()
                            camera.enable_em_gain(True)
                            camera.enable_direct_em_gain(True)
                            camera.set_sensitivity(1200)
                            camera.timeout = 2
                            X0, Y0, WIDTH, HEIGHT = roi
                            camera.set_roi(X0, WIDTH, Y0, HEIGHT)
                            print("[Rearrangement Node] Connected to camera")
                        except Exception as e:
                            print(f"[Rearrangement Node] Error connecting to camera: {e}")
                            return

                    data1 = np.frombuffer(parts[1], dtype=dtype1).reshape(shape1)
                    data2 = np.frombuffer(parts[2], dtype=dtype2).reshape(shape2)

                    # Convert each of these into a cupy array
                    data1 = cp.asarray(data1)
                    data2 = cp.asarray(data2)

                    w1, phi1, x1, y1 = data1
                    w2, phi2, x2, y2 = data2

                    pos1 = cp.stack((x1, y1), axis=-1)
                    pos2 = cp.stack((x2, y2), axis=-1)

                    terms1 = [w1, phi1, x1, y1, arr_shape1]
                    terms2 = [w2, phi2, x2, y2, arr_shape2]

                    static_background = PM.superimpose([PM.fresnel, PM.blaze, PM.zernike])
                    pm_array_init = PM.generate_phasemask(terms1)
                    pm_init = PM.superimpose([pm_array_init, PM.fresnel, PM.blaze, PM.zernike])
                    pm_init_uint8 = PM.transform_phase_8bit(pm_init).get()
                    
                    state = 1
                    print("[Rearrangement Node] Node initialised. Ready and waiting for commands...")
                    socket.send_json({"status": "success", "msg": "Rearrangement node initialised."})
                    
                elif command == "ARM_REARRANGEMENT":
                    if state != 1:
                        socket.send_json({"status": "error", "msg": "Node not initialised. Please send INITIALISE command first."})
                        continue

                    print("[Rearrangement Node] Arming rearrangement procedure...")

                    # INITIALIZE VRAM STATE MACHINE
                    curr_w = cp.asarray(w1, dtype=cp.float32)
                    curr_phi = cp.asarray(phi1, dtype=cp.float32)
                    curr_x = cp.asarray(x1, dtype=cp.float32)
                    curr_y = cp.asarray(y1, dtype=cp.float32)

                    # Initialize step vectors with zeros
                    dw = cp.zeros_like(curr_w)
                    dphi = cp.zeros_like(curr_phi)
                    total_dx = cp.zeros_like(curr_x)
                    total_dy = cp.zeros_like(curr_y)

                    # Ensure array operands are on the GPU to avoid implicit CPU conversion
                    w1_gpu, w2_gpu = cp.asarray(w1), cp.asarray(w2)
                    phi1_gpu, phi2_gpu = cp.asarray(phi1), cp.asarray(phi2)

                    # Load initial array
                    SLM.update_mask(pm_init_uint8)
                    print("[Rearrangement Node] Initial array loaded.")

                    # --- Set up the Buffer Queue and Worker Thread HERE ---
                    # maxsize=5 ensures the GPU doesn't run too far ahead of the SLM upload process
                    upload_queue = queue.Queue(maxsize=5)
                    upload_thread = threading.Thread(target=slm_upload_proc, args=(SLM, upload_queue))
                    upload_thread.daemon = True
                    upload_thread.start() 

                    print("[Rearrangement Node] Starting camera acquisition. Waiting for image...")

                    # Obtain image
                    try:
                        camera.start_acquisition()
                        img_array0 = camera.acquire_n_frames(1)[0]
                        t1 = time.time()
                        print(f"[Rearrangement Node] Image received! Executing...")
                    except Exception as e:
                        print(f"[Rearrangement Node] Error acquiring image: {e}")
                        return

                    # Extract occupancy mask
                    try:
                        img = an.morphological_tophat_high_pass(img_array0, feature_size=10)
                        # previous Pythonic implementation of sum_pixel_values 
                        # pixel_sums = np.fliplr(an.sum_pixel_values(img, grid_positions, arr_shape1, window_size=3))

                        # the C++ accelerated version
                        pixel_sums = np.fliplr(sum_cpp.sum_pixel_values(img, grid_positions, arr_shape1, window_size=3))

                        occ_mask = np.zeros(len(pixel_sums.flatten()), dtype=bool)
                        occ_mask[pixel_sums.flatten() > threshold] = True
                        occ_mask = cp.asarray(occ_mask)
                        t2 = time.time()
                        print("[Rearrangement Node] Occupancy mask extracted.")
                    except Exception as e:
                        print(f"[Rearrangement Node] Error during image processing: {e}")
                        return

                    # Generate the rearrangement phasemask sequence
                    try:
                        # Jonker-Volgenant rearrangement algorithm implementation
                        occ_indices = cp.where(occ_mask)[0]
                        init = pos1[occ_indices]
                        final = pos2
                        
                        init_idx, final_idx = get_jv_pairing_lap(init, final)
                        
                        # Map the Hungarian output back to the original array indices
                        moving_idx = occ_indices[init_idx]
                        
                        # Compute mask for traps to be switched off
                        off_mask = cp.ones(len(pos1), dtype=bool)
                        off_mask[moving_idx] = False
                        
                        # 2. CALCULATE INTERPOLATION STEPS
                        pos_init = pos1[moving_idx]
                        pos_final = pos2[final_idx]
                        vec = pos_final - pos_init
                        
                        # Faster L2 norm calculation using cupy linear algebra
                        max_dist = cp.linalg.norm(vec, axis=1).max()
                        #n_steps = int(cp.ceil(1.875 * max_dist / d0))          # Minimum jerk profile
                        n_steps = int(cp.ceil(max_dist / d0))                   # Linear profile
                        
                        # Pre-calculate the minimum jerk step multipliers on the GPU
                        tau = cp.linspace(0, 1, n_steps + 1, dtype=cp.float32)  
                        #s_profile = 10 * tau**3 - 15 * tau**4 + 6 * tau**5     # Minimum jerk profile
                        s_profile = tau                                         # Linear profile

                        # ds_profile contains the fractional progression for each step n
                        ds_profile = cp.diff(s_profile)
                        
                        # CRITICAL: Pull ds_profile back to the CPU! 
                        # Accessing GPU array scalars inside a loop causes implicit device synchronizations.
                        ds_profile_cpu = ds_profile.get()

                        # Load steps for MOVING traps
                        dw[moving_idx] = (w2_gpu[final_idx] - w1_gpu[moving_idx]) / n_steps
                        total_dx[moving_idx] = vec[:, 0].astype(cp.float32)
                        total_dy[moving_idx] = vec[:, 1].astype(cp.float32)
                        
                        # Ensure Phase Interpolation takes the shortest angular path to prevent wrapping tears
                        phase_diff = (phi2_gpu[final_idx] - phi1_gpu[moving_idx] + cp.pi) % (2 * cp.pi) - cp.pi
                        dphi[moving_idx] = phase_diff / n_steps
                        
                        # Load steps for OFF traps (Ramp down weights to 0)
                        dw[off_mask] = 0.0
                        curr_w[off_mask] = 0.0
                        
                        # 4. THE ULTRA-FAST GPU LOOP
                        for n in range(n_steps):
                            
                            # Use the fused Elementwise kernel: 1 kernel launch instead of 4
                            # We cast ds to float so it's passed as a fast C-scalar to the kernel
                            ds = float(ds_profile_cpu[n])
                            update_state_kernel(dw, dphi, total_dx, total_dy, ds, curr_w, curr_phi, curr_x, curr_y)
                            
                            # Repack the terms and call the generator.
                            terms_gpu = (curr_w, curr_phi, curr_x, curr_y, arr_shape1)
                            pm_slm = PM.generate_phasemask(terms_gpu)
                            
                            # Only superimpose the moving traps with the pre-calculated static background
                            composite_pm = PM.superimpose([pm_slm, static_background])

                            upload_queue.put(PM.transform_phase_8bit(composite_pm))
                            
                        upload_queue.put(None)
                        
                        t3 = time.time()

                        print(f"[Rearrangement Node] {n_steps} Frame rearrangement sequence generated.")
                    except Exception as e:
                        print(f"[Rearrangement Node] Error during sequence generation: {e}")
                        return

                    t4 = time.time()
                    print(f"[Rearrangement Node] Occupancy mask extraction duration: {(t2 - t1):.6f} s")
                    print(f"[Rearrangement Node] Rearrangement sequence generation duration: {(t3 - t2):.6f} s")
                    print(f"[Rearrangement Node] SLM upload duration: {(t4 - t3):.6f} s")
                    print(f"[Rearrangement Node] Total rearrangement duration: {(t4 - t1):.6f} s")

                    print(f"[Rearrangement Node] Rearrangement complete. Waiting for reset trigger...")

                    try:
                        camera.start_acquisition()
                        img_array1 = camera.acquire_n_frames(1)[0]
                        print(f"[Rearrangement Node] Image received! Disarming...")
                    except Exception as e:
                        print(f"[Rearrangement Node] Error acquiring image: {e}")
                        print(f"[Rearrangement Node] Proceeding with empty image for reset due to acquisition error.")
                        img_array1 = np.zeros_like(img_array0)

                    output_header = {
                        "status": "SUCCESS",
                        "img0_dtype": str(img_array0.dtype),
                        "img0_shape": img_array0.shape,
                        "img1_dtype": str(img_array1.dtype),
                        "img1_shape": img_array1.shape,
                        "timings": {
                            "n_frames": n_steps,
                            "occupancy_extraction_s": (t2 - t1),
                            "rearrangement_sequence_generation_s": (t3 - t2),
                            "slm_upload_s": (t4 - t3),
                            "total_rearrangement_s": (t4 - t1)
                        }
                    }

                    socket.send_multipart([
                        pkl.dumps(output_header),
                        img_array0,
                        img_array1,
                    ], copy=False)

                elif command == "TEST":
                    print("[Rearrangement Node] Test command received. Sending test response...")
                    time.sleep(10)
                    img0 = np.random.randint(0, 256, (10,10), dtype=np.uint8)
                    img1 = np.random.randint(0, 256, (10,10), dtype=np.uint8)

                    output_header = {
                        "status": "SUCCESS",
                        "img0_dtype": str(img0.dtype),
                        "img0_shape": img0.shape,
                        "img1_dtype": str(img1.dtype),
                        "img1_shape": img1.shape
                    }

                    socket.send_multipart([
                        pkl.dumps(output_header),
                        img0,
                        img1
                    ], copy=False)

                elif command == "SHUTDOWN":
                    print("[Rearrangement Node] Shutdown command received. Initiating safe shutdown sequence...")
                    socket.send_json({"status": "success", "msg": "Server shutting down."})
                    break

                else:
                    socket.send_json({"status": "error", "msg": f"Unknown command: {command}"})
                    
            except Exception as e:
                print(f"[Error] {str(e)}")
                socket.send_json({"status": "error", "msg": str(e)})
                
    except KeyboardInterrupt:
        print("\n[Rearrangement Node] Keyboard interrupt received! Initiating safe shutdown sequence...")
        
    finally:
        # This block ALWAYS runs, whether the loop exits via Ctrl+C, 
        # a remote shutdown command, or an unexpected error.
        if state == 1:
            camera.close()
        socket.close()
        context.term()
        print("[Rearrangement Node] Camera connection closed.")
        print("[Rearrangement Node] ZMQ sockets closed. Server has exited safely.")
        sys.exit(0)

if __name__ == "__main__":
    run_server()