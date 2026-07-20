import zmq
import numpy as np
import json
from ctypes import *
from PIL import Image
import sys
import time

class SLMHardwareController:
    def __init__(self):
        print("[Hardware] Initializing SLM...")

        cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink Plus\\SDK\\Blink_C_wrapper")
        self.slm_lib = CDLL("Blink_C_wrapper")
        self.slm_lib.Read_SLM_temperature.argtypes = [c_int]
        self.slm_lib.Read_SLM_temperature.restype = c_double

        num_boards_found = c_uint(0)
        constructed_okay = c_uint(-1)
        self.board_number = 1
        wait_For_Trigger = 0 #image writes to the SLM hold off until an external trigger is received by the hardware
        flip_immediate = 0 #only supported on the 1024
        OutputPulseImageFlip = 0 #enables the hardware to generate an output pulse when new images data is loaded to the SLM
        self.timeout_ms = 5000

        self.slm_lib.Create_SDK(byref(num_boards_found), byref(constructed_okay))
        if constructed_okay.value == 0:
            print ("Blink SDK did not construct successfully")

        if num_boards_found.value == 1:
            print("Blink SDK was successfully constructed")
            print("Found %s SLM controller(s)" % num_boards_found.value)
            height = self.slm_lib.Get_image_height(self.board_number)
            width = self.slm_lib.Get_image_width(self.board_number)
            depth = self.slm_lib.Get_image_depth(self.board_number) #Bits per pixel
            Bytes = depth//8
            center_x = width//2
            center_y = height//2
            serial = self.slm_lib.Read_Serial_Number(self.board_number)
            temp = self.slm_lib.Read_SLM_temperature(self.board_number)
            print(f"SLM temperature = {temp:.4f} deg")
            print(f"width = {width}, height = {height}, depth = {depth} bits, serial number = {serial}")
            self.slm_lib.SetWaitForTrigger(self.board_number, wait_For_Trigger)
            self.slm_lib.SetFlipImmediate(self.board_number, flip_immediate)
            self.slm_lib.SetOutputPulse(self.board_number, OutputPulseImageFlip)

            if width == 1024:
                load_lut_status = self.slm_lib.Load_LUT_file(self.board_number, b"C:\\Program Files\\Meadowlark Optics\\Blink Plus\\LUT Files\\slm40_at852.LUT")
            if load_lut_status == 1:
                print("LUT was successfully loaded")
            if load_lut_status == 0:
                print("Error loading LUT file. Check LUT file and path.")

            BlankImage = np.zeros([width*height*Bytes], np.uint8, 'C')

            retVal = self.slm_lib.Write_image(self.board_number, BlankImage.ctypes.data_as(POINTER(c_ubyte)), self.timeout_ms)
            if retVal != 1:
                print("DMA Failed")
            else:
                print("Blank phasemask uploaded.")
        
    def update_mask(self, mask_array):
        ret0 = self.slm_lib.Write_image(self.board_number,
            mask_array.ctypes.data_as(POINTER(c_ubyte)),
            self.timeout_ms)
        if ret0 != 1:
            print("[Hardware] DMA Failed")
        ret1 = self.slm_lib.ImageWriteComplete(self.board_number, self.timeout_ms)
        if ret1 != 1:
            print("[Hardware] Image write complete failed")
        print(f"[Hardware] Phasemask updated.")


    def preload_sequence(self, mask_sequence):
        """Uploads a whole sequence into the SLM's own on-board memory in one
        call. Does NOT display anything by itself - times just the upload."""
        list_length = mask_sequence.shape[0]
        if not mask_sequence.flags['C_CONTIGUOUS']:
            mask_sequence = np.ascontiguousarray(mask_sequence)

        t0 = time.perf_counter()
        retVal = self.slm_lib.PreLoad_sequence(
            self.board_number,
            mask_sequence.ctypes.data_as(POINTER(c_ubyte)),
            list_length,
            self.timeout_ms,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if retVal != 1:
            print("[Hardware] PreLoad_sequence failed")
        else:
            print(f"[Hardware] Preloaded {list_length} frames in {elapsed_ms:.3f} ms.")
        return elapsed_ms

    def get_temperature(self):
        return self.slm_lib.Read_SLM_temperature(self.board_number)

    def run_sequence(self, mask_sequence, fps):
        for i in range(mask_sequence.shape[0]):
            self.update_mask(mask_sequence[i])
            time.sleep(1/fps)
            print(f"[Hardware] Displaying frame {i+1}/{mask_sequence.shape[0]} at {fps} FPS.")
        
        print(f"[Hardware] Phasemask sequence uploaded.")
    
    def shutdown(self):
        """Safely powers down or disconnects from the SLM hardware."""
        print("[Hardware] Safely disconnecting from SLM...")
        self.slm_lib.Delete_SDK()
        print("Blink SDK closed.")

def run_slm_server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    
    port = 5555
    socket.bind(f"tcp://0.0.0.0:{port}")
    socket.setsockopt(zmq.RCVTIMEO, 1000)
    
    slm = SLMHardwareController()
    
    print(f"--- Fast SLM Server Listening on port {port} ---")
    print("Press Ctrl+C to shut down the server safely.")
    
    try:
        while True:
            try:
                frames = socket.recv_multipart()
            except zmq.error.Again:
                continue
            
            # Frame 0 is always our JSON header
            header = json.loads(frames[0].decode('utf-8'))
            command = header.get("cmd")
            
            try:
                if command == "UPDATE_MASK":
                    dtype = np.dtype(header["dtype"])
                    shape = header["shape"]
                    mask_array = np.frombuffer(frames[1], dtype=dtype).reshape(shape)
                    slm.update_mask(mask_array)
                    
                    socket.send_json({"status": "success", "msg": "Mask updated successfully."})

                elif command == "RUN_SEQUENCE":
                    dtype = np.dtype(header["dtype"])
                    shape = header["shape"]
                    fps = header["fps"]
                    sequence = np.frombuffer(frames[1], dtype=dtype).reshape(shape)
                    slm.run_sequence(sequence, fps)
                    
                    socket.send_json({"status": "success", "msg": "Sequence completed successfully."})
                
                elif command == "PRELOAD_SEQUENCE":
                    dtype = np.dtype(header["dtype"])
                    shape = header["shape"]
                    sequence = np.frombuffer(frames[1], dtype=dtype).reshape(shape)
                    elapsed_ms = slm.preload_sequence(sequence)

                    socket.send_json({
                        "status": "success",
                        "msg": "Sequence preloaded.",
                        "preload_ms": elapsed_ms,
                        "n_frames": int(shape[0]),
                    })

                    
                elif command == "CHECK_TEMP":
                    temp = slm.get_temperature()
                    socket.send_json({"status": "success", "data": temp})
                    
                elif command == "SHUTDOWN":
                    print("[Server] Remote shutdown requested.")

                    socket.send_json({"status": "success"})
                    break
                    
                else:
                    socket.send_json({"status": "error", "msg": f"Unknown command: {command}"})
                    
            except Exception as e:
                print(f"[Error] {str(e)}")
                socket.send_json({"status": "error", "msg": str(e)})
                
    except KeyboardInterrupt:
        print("\n[Server] Keyboard interrupt received! Initiating safe shutdown sequence...")
        
    finally:
        slm.shutdown()
        socket.close()
        context.term()
        print("[Server] ZMQ sockets closed. Server has exited safely.")
        sys.exit(0)

if __name__ == "__main__":
    run_slm_server()