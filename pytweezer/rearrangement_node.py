import multiprocessing as mp
import zmq
import numpy as np
import time
from pytweezer.analysis import analysis as an
from pytweezer import phasemask as pm
from pytweezer import communication as comm
from pytweezer.drivers.imagemX2 import ImagEMX2Camera, ImagEMX2CameraClient

class RearrangementNode(mp.Process):
    def __init__(self, control_queue, message_queue, slm_port=5555):
        super().__init__()
        self.control_queue = control_queue
        self.message_queue = message_queue
        self.slm_port = slm_port
        self.daemon = True 
        self.message_queue.put(f"[Fast SLM Node] Initialised.")
        

    def _log(self, message):
        """Helper method to send print statements back to the Jupyter UI."""
        self.message_queue.put(message)

    def do_rearrangement(self, grid_positions, array_shape, threshold, pm_init, terms1, terms2, d0, fps):
        # Wait for an image to arrive (timeout after 10ms to check Jupyter commands again)
        # 1. Receive Image from Camera Server

        self._log("[Fast SLM Node] Starting camera acquisition for rearrangement.")
        try:
            self.camera.start_acquisition()
            img_array = self.camera.acquire_n_frames(1, broadcast=True)[0]
            start1 = time.time()
            self._log("[Fast SLM Node] Image received! Executing pipeline...")
        except Exception as e:
            self._log(f"[Fast SLM Node] Error during camera acquisition: {e}")
            return

        # 2. Extract Occupancy Mask
        # (Insert your image processing/thresholding logic here)
        try:
            pixel_sums = an.sum_pixel_values(img_array, grid_positions, array_shape, window_size=3)[::-1, ::-1]
            occ_mask = np.zeros(len(pixel_sums.flatten()), dtype=bool)
            occ_mask[pixel_sums.flatten() > threshold] = True
            self._log("[Fast SLM Node] Occupancy mask extracted.")
        except Exception as e:
            self._log(f"[Fast SLM Node] Error during image processing: {e}")
        
        # 3. Generate the highly-optimized sequence (Runs instantly on GPU)
        try:
            sequence = self.phasemask.generate_rearrangement_sequence(terms1, terms2, occ_mask, d0=d0)
            self._log("[Fast SLM Node] Rearrangement sequence generated.")
        except Exception as e:
            self._log(f"[Fast SLM Node] Error during sequence generation: {e}")
            return
        
        # 4. Send directly to SLM Server
        start2 = time.time()
        self.SLM.run_sequence(sequence, fps=fps)
        self._log(f"[Fast SLM Node] SLM upload duration: {(time.time() - start2):.6f} s")
        self._log(f"[Fast SLM Node] Total rearrangement duration: {(time.time() - start1):.6f} s")
        
        self._log(f"[Fast SLM Node] Rearrangement complete. Waiting for reset trigger.")
        self.camera.start_acquisition()
        img_array = self.camera.acquire_n_frames(1, broadcast=True)[0]
        self.SLM.update_mask(pm_init)
        self._log(f"[Fast SLM Node] Array reset.")

    def do_sequence(self, sequence,fps):
        # Wait for an image to arrive (timeout after 10ms to check Jupyter commands again)
        # 1. Receive Image from Camera Server

        self._log("[Fast SLM Node] Starting camera acquisition for sequence.")
        try:
            self.camera.start_acquisition()
            img_array = self.camera.acquire_n_frames(1, broadcast=True)[0]
            start1 = time.time()
            self._log("[Fast SLM Node] Image received! Executing pipeline...")
        except Exception as e:
            self._log(f"[Fast SLM Node] Error during camera acquisition: {e}")
            return
        
        # 4. Send directly to SLM Server
        start1 = time.time()
        self.SLM.run_sequence(sequence, fps=fps)
        self._log(f"[Fast SLM Node] SLM upload duration: {(time.time() - start1):.6f} s")
        
        self._log(f"[Fast SLM Node] Sequence upload complete. Waiting for reset trigger.")
        self.camera.start_acquisition()
        img_array = self.camera.acquire_n_frames(1, broadcast=True)[0]
        self.SLM.update_mask(sequence[0])
        self._log(f"[Fast SLM Node] Array reset.")




    def run(self):
        """This runs in a completely separate CPU core and memory space."""
        self._log("[Fast SLM Node] Booting up...")

        # Setup SLM
        try:
            self.SLM = comm.SLMClient()
            self._log("[Fast SLM Node] Connected to SLM server.")
        except Exception as e:
            self._log(f"[Fast SLM Node] Error connecting to SLM server: {e}")
            return

        # Set up phasemask generation
        self.phasemask = pm.OptimisationBasedPhasemaskGeneratorGPU()

        # Setup camera
        self.camera = ImagEMX2CameraClient()
        self.camera.setup_acquisition("snap", 1)
        self._log("[Fast SLM Node] Connected to camera server.")
        
        # State variables
        pm_init, terms1, terms2, d0, threshold, grid_positions, array_shape, fps = None, None, None, None, None, None, None, None
        
        self._log("[Fast SLM Node] Ready and waiting for commands.")
        
        while True:
            # --- CHECK FOR JUPYTER COMMANDS ---
            try:
                # get_nowait() checks the queue instantly without blocking the loop
                cmd = self.control_queue.get_nowait()
                if cmd["type"] == "ARM_REARRANGEMENT":
                    pm_init = cmd["pm_init"]
                    terms1 = cmd["terms1"]
                    terms2 = cmd["terms2"]
                    d0 = cmd["d0"]
                    threshold = cmd["threshold"]
                    grid_positions = cmd["grid_positions"]
                    array_shape = cmd["initial_array_shape"]
                    fps = cmd["fps"]
                    self._log("[Fast SLM Node] Arm command received. Starting rearrangement sequence...")
                    self.do_rearrangement(grid_positions, array_shape, threshold, pm_init, terms1, terms2, d0, fps)

                elif cmd["type"] == "ARM_SEQUENCE":
                    sequence = cmd["sequence"]
                    fps = cmd["fps"]
                    self._log("[Fast SLM Node] Arm sequence command received. Uploading sequence to SLM...")
                    self.do_sequence(sequence, fps)
                    
                elif cmd["type"] == "SHUTDOWN":
                    self._log("[Fast SLM Node] Shutting down.")
                    break
            except mp.queues.Empty:
                pass


                    
