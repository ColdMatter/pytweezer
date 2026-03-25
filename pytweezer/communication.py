import zmq
import numpy as np
import json

class SLMClient:
    def __init__(self, address="tcp://10.59.3.1:5555", timeout_ms=2000):
        self.address = address
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self._connect_socket()

    def _connect_socket(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.connect(self.address)

    def _send_multipart_command(self, header_dict, array=None):
        """Packs the header and raw array into a multipart message."""
        # 1. Encode the header as bytes
        header_bytes = json.dumps(header_dict).encode('utf-8')
        frames = [header_bytes]
        
        # 2. Append the raw array memory if provided
        if array is not None:
            # Crucial: Ensure the memory is a contiguous C-style block
            if not array.flags['C_CONTIGUOUS']:
                array = np.ascontiguousarray(array)
            
            # array.data accesses the raw memory buffer without copying it
            frames.append(array.data)
            
        # 3. Send all frames at once
        self.socket.send_multipart(frames)
        
        try:
            # The server replies with a simple JSON dictionary
            return self.socket.recv_json()
            
        except zmq.error.Again:
            print(f"Network Error: SLM Server at {self.address} timed out.")
            self.socket.close(linger=0)
            self._connect_socket()
            return {"status": "error", "msg": "timeout"}

    # --- User-Facing Methods ---
    
    def update_mask(self, mask_array):
        """Sends metadata in the header and the raw bytes in the payload."""
        if not isinstance(mask_array, np.ndarray):
            raise ValueError("Mask must be a NumPy array.")
            
        header = {
            "cmd": "UPDATE_MASK",
            "dtype": str(mask_array.dtype),
            "shape": mask_array.shape
        }
        
        return self._send_multipart_command(header, array=mask_array)

    def get_temperature(self):
        """Sends a simple command with no array payload."""
        return self._send_multipart_command({"cmd": "GET_TEMP"})