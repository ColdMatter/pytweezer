import multiprocessing as mp
import zmq
import numpy as np
import time
from pytweezer import analysis as an
from pytweezer import phasemask as pm
from pytweezer import communication as comm

class RearrangementNode(mp.Process):
    def __init__(self, control_queue, cam_port=5556, slm_port=5555):
        super().__init__()
        self.control_queue = control_queue
        self.cam_port = cam_port
        self.slm_port = slm_port
        self.daemon = True 
        print(f"[Rearrangement Node] Initialised.")

    def run(self):
        """This runs in a completely separate CPU core and memory space."""
        print("[Rearrangement Node] Booting up...")
        
        # 1. Setup ZeroMQ Connections
        context = zmq.Context()
        
        # Subscribe to Camera (SUB)
        cam_sub = context.socket(zmq.SUB)
        cam_sub.connect(f"tcp://10.59.3.1:{self.cam_port}")
        cam_sub.setsockopt_string(zmq.SUBSCRIBE, "") # Listen to everything
        
        # Connect to SLM Server (REQ)
        SLM = comm.SLMClient()
        
        # Setup Poller to listen to both the Notebook and the Camera efficiently
        poller = zmq.Poller()
        poller.register(cam_sub, zmq.POLLIN)
        
        # 2. Initialize the GPU Generator
        # (Replace with your actual SLM dimensions and hardware physics)
        generator = pm.OptimisationBasedPhasemaskGeneratorGPU(
                 wavelength_um=0.852,
                 focal_length_mm=10.0,
                 slm_pitch_um=8,
                 slm_res=(1200,1920),
                 input_beam_waist_mm=9.6)
        
        # State variables
        armed = False
        pm_init, terms1, terms2, d0, threshold, img_shape, grid_positions, array_shape = None, None, None, None, None, None, None, None
        
        print("[Rearrangement Node] Ready and waiting for commands.")
        
        while True:
            # --- CHECK FOR JUPYTER COMMANDS ---
            try:
                # get_nowait() checks the queue instantly without blocking the loop
                cmd = self.control_queue.get_nowait()
                if cmd["type"] == "ARM":
                    pm_init = cmd["pm_init"]
                    terms1 = cmd["terms1"]
                    terms2 = cmd["terms2"]
                    d0 = cmd["d0"]
                    threshold = cmd["threshold"]
                    img_shape = cmd["img_shape"]
                    grid_positions = cmd["grid_positions"]
                    array_shape = cmd["initial_array_shape"]
                    
                    # Flush any old images sitting in the camera buffer before arming
                    while cam_sub.poll(0):
                        cam_sub.recv()
                        
                    armed = True
                    print("[Rearrangement Node] ARMED. Waiting for hardware camera trigger...")
                    
                elif cmd["type"] == "SHUTDOWN":
                    print("[Rearrangement Node] Shutting down.")
                    break
            except mp.queues.Empty:
                pass

            # --- CHECK FOR CAMERA TRIGGERS ---
            if armed:
                # Wait for an image to arrive (timeout after 10ms to check Jupyter commands again)
                socks = dict(poller.poll(10))
                
                if cam_sub in socks:
                    # 1. Receive Image from Camera Server
                    img_bytes = cam_sub.recv()
                    # (Assuming camera sends a raw 2D numpy buffer; adjust decode as needed)
                    image = np.frombuffer(img_bytes, dtype=np.uint16).reshape(img_shape)
                    
                    print("[Rearrangement Node] Image received! Executing pipeline...")
                    t_start = time.time()
                    
                    # 2. Extract Occupancy Mask
                    # (Insert your image processing/thresholding logic here)
                    pixel_sums = an.sum_pixel_values(image, grid_positions, array_shape, window_size=3)
                    occ_mask = np.zeros(len(pixel_sums.flatten()), dtype=bool)
                    occ_mask[pixel_sums.flatten() > threshold] = True
                    
                    # 3. Generate the highly-optimized sequence (Runs instantly on GPU)
                    sequence = generator.generate_rearrangement_sequence(terms1, terms2, occ_mask, d0=d0)
                    
                    # 4. Send directly to SLM Server
                    SLM.run_sequence(sequence, fps=10000)
                    print(f"[Rearrangement Node] Loop Complete in {(time.time()-t_start)*1000:.2f}ms. Disarming.")
                    
                    # Disarm to prevent accidental re-firing on subsequent images
                    armed = False

            elif not armed:
                socks = dict(poller.poll(10))
                if cam_sub in socks:
                    img_bytes = cam_sub.recv()
                    SLM.update_mask(pm_init)
                    print(f"[Rearrangement Node] Array reset.")

                    
