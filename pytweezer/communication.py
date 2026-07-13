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
    
    def run_sequence(self, sequence, fps=1):
        """Sends a 3D array of masks to be played in rapid succession."""
        if not isinstance(sequence, np.ndarray):
            raise ValueError("Sequence must be a NumPy array.")
        if len(sequence.shape) != 3:
            raise ValueError("Sequence must be a 3D array (frames, height, width).")
            
        header = {
            "cmd": "RUN_SEQUENCE",
            "dtype": str(sequence.dtype),
            "shape": sequence.shape,
            "fps": fps
        }
        
        expected_play_time_ms = int((sequence.shape[0] / fps) * 1000) + 5000 
        self.socket.setsockopt(zmq.RCVTIMEO, expected_play_time_ms)
        
        reply = self._send_multipart_command(header, array=sequence)
        
        # Restore original timeout for future quick commands
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        return reply

    def preload_sequence(self, sequence):
        """ Returns the reply dict, which
        includes the server-measured upload time."""
        if not isinstance(sequence, np.ndarray):
            raise ValueError("Sequence must be a NumPy array.")
        if len(sequence.shape) != 3:
            raise ValueError("Sequence must be a 3D array (frames, height, width).")

        header = {
            "cmd": "PRELOAD_SEQUENCE",
            "dtype": str(sequence.dtype),
            "shape": sequence.shape,
        }

        expected_upload_time_ms = sequence.shape[0] * 100 + 5000
        self.socket.setsockopt(zmq.RCVTIMEO, expected_upload_time_ms)

        reply = self._send_multipart_command(header, array=sequence)

        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        return reply

    def get_temperature(self):
        """Sends a simple command with no array payload."""
        return self._send_multipart_command({"cmd": "CHECK_TEMP"})


import zmq
import zmq.asyncio
import pickle as pkl

class RearrangementNode:
    def __init__(self, address="tcp://10.59.3.1:2222", timeout_ms=2000):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(address)

    async def initialise(self, terms1, terms2, grid_positions, threshold, d0, fps, roi=[50, 70, 384, 384]):
        """Sends metadata in the header and the raw bytes in the payload."""
        
        w1, theta1, x1, y1, arr_shape1 = terms1
        w2, theta2, x2, y2, arr_shape2 = terms2

        data1 = np.array([w1.get(), theta1.get(), x1.get(), y1.get()])
        data2 = np.array([w2.get(), theta2.get(), x2.get(), y2.get()])

        header = {
            "cmd": "INITIALISE",
            "dtype1": str(data1.dtype),
            "dtype2": str(data2.dtype),
            "shape1": data1.shape,
            "shape2": data2.shape,
            "array_shape1": arr_shape1,
            "array_shape2": arr_shape2,
            "d0": d0,
            "fps": fps,
            "threshold": threshold,
            "grid_positions": grid_positions,
            "roi": roi
        }

        await self.socket.send_multipart([pkl.dumps(header), data1, data2], copy=False)
        reply = await self.socket.recv_string()
        print(reply)

    async def arm_rearrangement(self):
        """Sends a simple command with no array payload."""
        header = {"cmd": "ARM_REARRANGEMENT"}
        await self.socket.send_multipart([pkl.dumps(header)])

        print("ARM command sent...")
        
        parts = await self.socket.recv_multipart()
        reply_header = pkl.loads(parts[0])
        img0 = np.frombuffer(parts[1], dtype=reply_header["img0_dtype"]).reshape(reply_header["img0_shape"])
        img1 = np.frombuffer(parts[2], dtype=reply_header["img1_dtype"]).reshape(reply_header["img1_shape"])
        timings = reply_header["timings"]
        return img0, img1, timings

    async def test(self):
        """Sends a simple command with no array payload."""
        header = {"cmd": "TEST"}
        await self.socket.send_multipart([pkl.dumps(header)])

        print("TEST command sent...")
        
        parts = await self.socket.recv_multipart()
        reply_header = pkl.loads(parts[0])
        img0 = np.frombuffer(parts[1], dtype=reply_header["img0_dtype"]).reshape(reply_header["img0_shape"])
        img1 = np.frombuffer(parts[2], dtype=reply_header["img1_dtype"]).reshape(reply_header["img1_shape"])
        
        print("Images obtained from server.")
        return img0, img1

    async def shutdown(self):
        """Sends a shutdown command to the server."""
        header = {"cmd": "SHUTDOWN"}
        await self.socket.send_multipart([pkl.dumps(header)])
        reply = await self.socket.recv_string()
        print(reply)