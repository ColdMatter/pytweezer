import zmq
import numpy as np
import time

class SyntheticCameraServer:
    def __init__(self, port=5556, image_size=(200, 200)):
        self.port = port
        self.image_size = image_size
        
        # Setup ZeroMQ Publisher
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://0.0.0.0:{self.port}")
        
        # Pre-calculate the pixel grid for the image
        x = np.arange(self.image_size[1])
        y = np.arange(self.image_size[0])
        self.X, self.Y = np.meshgrid(x, y)

    def generate_image(self, N, M, spacing, spot_width, fill_fraction=0.5):
        """
        Generates a synthetic camera image of a randomly occupied atom array.
        """
        # Start with a baseline of camera read-noise (mean=100, std=15)
        image = np.random.normal(100, 15, self.image_size)
        
        # Calculate the center of the image
        cx = self.image_size[1] / 2.0
        cy = self.image_size[0] / 2.0
        
        # Peak intensity for a single atom
        peak_intensity = 40000 
        
        # Generate the grid of spots
        for i in range(N):
            for j in range(M):
                # Random occupancy check
                if np.random.rand() < fill_fraction:
                    # Calculate physical pixel coordinates for this spot
                    spot_x = cx + (j - (M - 1) / 2.0) * spacing
                    spot_y = cy + (i - (N - 1) / 2.0) * spacing
                    
                    # Add 2D Gaussian spot
                    r_squared = (self.X - spot_x)**2 + (self.Y - spot_y)**2
                    spot_profile = peak_intensity * np.exp(-r_squared / (2 * spot_width**2))
                    image += spot_profile
                    
        # Clip values to valid uint16 range to mimic a real scientific camera
        image = np.clip(image, 0, 65535).astype(np.uint16)
        return image

    def run_loop(self, N=10, M=10, spacing=12.0, spot_width=1.5):
        """
        Main server loop: Generates and publishes an image every 10 seconds.
        """
        print(f"--- Synthetic Camera Server Started on Port {self.port} ---")
        print(f"Array: {N}x{M} | Spacing: {spacing}px | Spot Width: {spot_width}px")
        print("Publishing new image every 5 seconds. Press Ctrl+C to stop.")
        
        try:
            frame_count = 0
            while True:
                # 1. Generate synthetic data
                img = self.generate_image(N, M, spacing, spot_width, fill_fraction=0.6)
                
                # 2. Publish raw bytes over ZeroMQ
                self.socket.send(img.tobytes())
                
                frame_count += 1
                print(f"[Camera] Published Frame {frame_count:04d} at {time.strftime('%H:%M:%S')}")
                
                # 3. Wait 10 seconds before the next trigger
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n[Camera] Shutting down server...")
        finally:
            self.socket.close()
            self.context.term()

if __name__ == "__main__":
    server = SyntheticCameraServer(port=5556, image_size=(200, 200))
    
    # You can tweak the array dimensions and spot parameters here
    server.run_loop(
        N=15, 
        M=15, 
        spacing=10.0,     # pixels between traps
        spot_width=1.0    # Gaussian sigma in pixels
    )