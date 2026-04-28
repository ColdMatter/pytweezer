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
                                            blaze_dx_dy_um=(46.50, 10.54),
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
                            camera.timeout = 60*2
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

                    w1, theta1, x1, y1 = data1
                    w2, theta2, x2, y2 = data2

                    terms1 = [w1, theta1, x1, y1, arr_shape1]
                    terms2 = [w2, theta2, x2, y2, arr_shape2]

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

                    # Load initial array
                    SLM.update_mask(pm_init_uint8)
                    print("[Rearrangement Node] Initial array loaded.")

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
                        pixel_sums = np.fliplr(an.sum_pixel_values(img_array0, grid_positions, arr_shape1, window_size=3))
                        occ_mask = np.zeros(len(pixel_sums.flatten()), dtype=bool)
                        occ_mask[pixel_sums.flatten() > threshold] = True
                        t2 = time.time()
                        print("[Rearrangement Node] Occupancy mask extracted.")
                    except Exception as e:
                        print(f"[Rearrangement Node] Error during image processing: {e}")
                        return

                    # Generate the rearrangement phasemask sequence
                    try:
                        sequence, debug_sequence = PM.generate_rearrangement_sequence(terms1, terms2, occ_mask, d0=d0)
                        t3 = time.time()
                        print("[Rearrangement Node] Rearrangement sequence generated.")
                    except Exception as e:
                        print(f"[Rearrangement Node] Error during sequence generation: {e}")
                        return

                    # Upload to SLM
                    SLM.run_sequence(sequence, fps=fps)
                    t4 = time.time()
                    print(f"[Rearrangement Node] Occupancy mask extraction duration: {(t2 - t1):.6f} s")
                    print(f"[Rearrangement Node] Rearrangement sequence generation duration: {(t3 - t2):.6f} s")
                    print(f"[Rearrangement Node] SLM upload duration: {(t4 - t3):.6f} s")
                    print(f"[Rearrangement Node] Total rearrangement duration: {(t4 - t1):.6f} s")

                    print(f"[Rearrangement Node] Rearrangement complete. Waiting for reset trigger...")

                    try:
                        camera.start_acquisition()
                        img_array1 = camera.acquire_n_frames(1, timeout=5)[0]
                        print(f"[Rearrangement Node] Image received! Disarming...")
                    except Exception as e:
                        print(f"[Rearrangement Node] Error acquiring image: {e}")
                        print(f"[Rearrangement Node] Proceeding with empty image for reset due to acquisition error.")
                        img_array1 = np.zeros_like(img_array0)

                    debug_sequence_cpu = debug_sequence.get()

                    output_header = {
                        "status": "SUCCESS",
                        "img0_dtype": str(img_array0.dtype),
                        "img0_shape": img_array0.shape,
                        "img1_dtype": str(img_array1.dtype),
                        "img1_shape": img_array1.shape,
                        "debug_sequence_dtype": str(debug_sequence_cpu.dtype),
                        "debug_sequence_shape": debug_sequence_cpu.shape,
                        "occ_mask_dtype": str(occ_mask.dtype),
                        "occ_mask_shape": occ_mask.shape,
                        "timings": {
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
                        debug_sequence_cpu,
                        occ_mask
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