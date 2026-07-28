import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import time
import os
os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
from scipy.spatial.distance import cdist
import lap


def get_zernike_polynomial(noll_index, rho, theta, mask):
    """
    Returns the Zernike polynomial evaluated on the rho, theta grid 
    for a given Noll index.
    
    Includes the standard normalization constants so that the RMS 
    value of each polynomial over the unit disk is 1.
    """
    if noll_index == 1:
        # Piston
        Z = np.ones_like(rho)
    elif noll_index == 2:
        # Tip (X Tilt)
        Z = 2 * rho * np.cos(theta)
    elif noll_index == 3:
        # Tilt (Y Tilt)
        Z = 2 * rho * np.sin(theta)
    elif noll_index == 4:
        # Defocus
        Z = np.sqrt(3) * (2 * rho**2 - 1)
    elif noll_index == 5:
        # Oblique Astigmatism
        Z = np.sqrt(6) * rho**2 * np.sin(2 * theta)
    elif noll_index == 6:
        # Vertical Astigmatism
        Z = np.sqrt(6) * rho**2 * np.cos(2 * theta)
    elif noll_index == 7:
        # Vertical Coma
        Z = np.sqrt(8) * (3 * rho**3 - 2 * rho) * np.sin(theta)
    elif noll_index == 8:
        # Horizontal Coma
        Z = np.sqrt(8) * (3 * rho**3 - 2 * rho) * np.cos(theta)
    elif noll_index == 9:
        # Vertical Trefoil
        Z = np.sqrt(8) * rho**3 * np.sin(3 * theta)
    elif noll_index == 10:
        # Oblique Trefoil
        Z = np.sqrt(8) * rho**3 * np.cos(3 * theta)
    elif noll_index == 11:
        # Primary Spherical Aberration
        Z = np.sqrt(5) * (6 * rho**4 - 6 * rho**2 + 1)
    elif noll_index == 12:
        # Secondary Vertical Astigmatism
        Z = np.sqrt(10) * (4 * rho**4 - 3 * rho**2) * np.cos(2 * theta)
    elif noll_index == 13:
        # Secondary Oblique Astigmatism
        Z = np.sqrt(10) * (4 * rho**4 - 3 * rho**2) * np.sin(2 * theta)
    else:
        raise ValueError(f"Noll index {noll_index} is not implemented in this basic dictionary.")
        
    return Z * mask

def rotate_positions(x_n, y_n, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    
    x_rotated = cos_angle * x_n - sin_angle * y_n
    y_rotated = sin_angle * x_n + cos_angle * y_n
    
    return x_rotated, y_rotated


class ZernikeCalibrator:
    def __init__(self, json_filepath):
        """Loads the calibration weights from the JSON file."""
        with open(json_filepath, "r") as f:
            data = json.load(f)
        
        self.poly = data["polynomial_degree"]
        self.coeffs = data["coefficients"]
        
    def get_zernike_coeff(self, x, y):
        """
        Predicts the Z value for given x and y coordinates using the fitted polynomial.
        x and y can be scalar values or numpy arrays.
        """
        # Ensure the output shape matches the input shape
        z_pred = np.zeros_like(x, dtype=float)
        term_idx = 0
        
        for i in range(4):
            for j in range(4 - i):
                z_pred += self.coeffs[term_idx] * (x**i) * (y**j)
                term_idx += 1
                
        return z_pred
    

####################################################################################################################################

import cupy as cp

def rotate_coordinates(x, y, angle_deg, center_x=0, center_y=0):
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Shift to origin
    x_shifted = x - center_x
    y_shifted = y - center_y

    # Rotate
    x_rotated = x_shifted * cos_a - y_shifted * sin_a
    y_rotated = x_shifted * sin_a + y_shifted * cos_a

    # Shift back
    x_final = x_rotated + center_x
    y_final = y_rotated + center_y

    return x_final, y_final

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



class OptimisationBasedPhasemaskGeneratorGPU:
    def __init__(self,
                 wavelength_um=0.852,
                 focal_length_mm=17.3,
                 slm_pitch_um=17,
                 slm_res=(1024,1024),
                 input_beam_waist_mm=16,
                 fresnel_f_mm=1072,
                 blaze_dx_dy_um=(46.50, 10.54),
                 zernike_coeff_dict={5:1.195, 6:0.725, 7:0.970, 8:0.478, 9:-1.091, 10:0.303, 11:0.021, 12:0.072, 13:0.049}):
        
        self.lam = wavelength_um
        self.f_mm = focal_length_mm
        self.f_um = focal_length_mm * 1000
        self.dx_slm = slm_pitch_um
        self.Ny, self.Nx = slm_res
        self.w0_mm = input_beam_waist_mm
        self.w0_um = input_beam_waist_mm * 1000
        self.Lx_slm = self.Nx * self.dx_slm # In units of um
        self.Ly_slm = self.Ny * self.dx_slm # In units of um
        self.dx_f = (self.lam * self.f_um) / self.Lx_slm # In units of um
        self.dy_f = (self.lam * self.f_um) / self.Ly_slm # In units of um
        self.Lx_foc = self.Nx * self.dx_f
        self.Ly_foc = self.Ny * self.dy_f
        self.x_slm = cp.linspace(-self.Lx_slm/2, self.Lx_slm/2, self.Nx, dtype=cp.float32)
        self.y_slm = cp.linspace(-self.Ly_slm/2, self.Ly_slm/2, self.Ny, dtype=cp.float32)

        # Precalculations
        self.k = np.float32(2 * np.pi / (self.lam * self.f_um))
        self.j_k = cp.complex64(1j) * self.k
        self.uint8_scale = cp.float32(255.0) / (2 * cp.pi)

        # Additional phasemasks - phase wrapped and in radians
        self.fresnel = self.generate_fresnel_lens_phasemask(focal_length_mm=fresnel_f_mm)
        self.blaze = self.generate_blazed_grating_phasemask(dx_um=blaze_dx_dy_um[0], dy_um=blaze_dx_dy_um[1])
        self.zernike = self.generate_zernike_phasemask(zernike_coeff_dict, wrap=True)
        
        print(f"--- System Configuration ---")
        print(f"SLM Plane Width: {self.Lx_slm/1000:.2f} mm")
        print(f"SLM Plane Height: {self.Ly_slm/1000:.2f} mm")
        print(f"Focal Plane Resolution x (pixel size): {self.dx_f:.4f} um")
        print(f"Focal Plane Resolution y (pixel size): {self.dy_f:.4f} um")
        print(f"Focal Plane Width: {self.Lx_foc:.2f} um")
        print(f"Focal Plane Height: {self.Ly_foc:.2f} um")
        print(f"Fresnel Lens Focal Length: {fresnel_f_mm:.2f} mm")
        print(f"Blazed Grating Displacement (dx, dy): {blaze_dx_dy_um} um")

    def generate_source_amplitude(self):
        """Generates Gaussian input beam amplitude."""
        X, Y = cp.meshgrid(self.x_slm, self.y_slm)
        R2 = X**2 + Y**2
        return cp.exp(- R2 / (2*(self.w0_um/2.355)**2))
    
    def generate_weighted_array(self, weights, spacing, init_phase_randomness=1.0, angle_deg=0):
        """
        Generates a weighted mask indicating where the tweezers should be and
        how strong they should be.
        Returns both the full 2D array and a list of the specific (y, x) indices.
        """
        dim = weights.shape
        xspan = (dim[1] - 1) * spacing
        yspan = (dim[0] - 1) * spacing
        xpos = np.linspace(-(xspan)/2, (xspan)/2, dim[1])
        ypos = np.linspace(-(yspan)/2, (yspan)/2, dim[0])
        Xpos, Ypos = np.meshgrid(xpos, ypos)
        Xn, Yn = Xpos.flatten(), Ypos.flatten()
        Xn, Yn = rotate_coordinates(Xn, Yn, angle_deg)
        Wn = weights.flatten()
        Thetan = np.random.rand(len(Wn)) * init_phase_randomness * 2 * np.pi

        print(f"--- Target Generation ---")
        print(f"Grid: {dim[0]}x{dim[1]}")
        print(f"Spacing: {spacing} um")
                    
        return [Wn, Thetan, Xn, Yn, dim]
    
    def generate_zernike_phasemask(self, zernike_coeffs, wrap=False):
        """
        Generates a 2D Zernike correction phasemask defined on the SLM plane.
        
        Parameters:
        -----------
        settings : dict
            Must contain 'slm_shape' (Ny, Nx) and 'pixel_pitch' (p).
        zernike_coeffs : dict
            A dictionary mapping the Noll Index (int) to its coefficient (float).
            Example: {4: 1.5, 8: -0.5} applies 1.5 rads of Defocus and -0.5 rads of Horizontal Coma.
            
        Returns:
        --------
        total_phase : np.ndarray
            The 2D phase mask in radians to be applied to the SLM.
        """

        # 1. Setup SLM Coordinates (Centered at 0)
        X_slm, Y_slm = cp.meshgrid(self.x_slm, self.y_slm)
        R_max = cp.sqrt(self.Lx_slm**2 + self.Ly_slm**2) / 2.0
        rho = cp.sqrt(X_slm**2 + Y_slm**2) / R_max
        theta = cp.arctan2(Y_slm, X_slm)
        
        # 3. Define the unit circle mask
        mask = (rho <= 1.0).astype(float)
        
        # Initialize an empty phase array
        total_phase = cp.zeros((self.Ny, self.Nx))
        
        # 4. Superpose the requested Zernike modes
        for noll_index, coeff in zernike_coeffs.items():
            if coeff != 0.0:
                Z_mode = get_zernike_polynomial(noll_index, rho, theta, mask)
                total_phase += coeff * Z_mode

        return (total_phase % (2 * cp.pi) - cp.pi).astype(cp.float32)

    def generate_fresnel_lens_phasemask(self, focal_length_mm):
        """
        Generates a Fresnel lens phase mask to focus the beam at a specific focal length.
        This is useful for correcting defocus or creating a virtual focus plane.
        """
        # 1. Setup SLM Coordinates (Centered at 0)
        X_slm, Y_slm = cp.meshgrid(self.x_slm, self.y_slm)
        
        # 2. Calculate the Fresnel lens phase profile
        f_um = focal_length_mm * 1000
        k = 2 * cp.pi / self.lam
        R_squared = X_slm**2 + Y_slm**2
        fresnel_phase = (k / (2 * f_um)) * R_squared
        
        return (fresnel_phase % (2*cp.pi) - cp.pi).astype(cp.float32)
    
    def generate_blazed_grating_phasemask(self, dx_um, dy_um):
        """
        Generates a blazed grating phase mask to steer the beam by (dx_um, dy_um) in the focal plane.
        The steering angles are calculated based on the desired displacement and the system's focal length.
        """
        # 1. Setup SLM Coordinates (Centered at 0)
        X_slm, Y_slm = cp.meshgrid(self.x_slm, self.y_slm)
        
        # 2. Calculate steering angles
        theta_x = cp.arctan(dx_um / self.f_um)  # Steering angle in radians for x
        theta_y = cp.arctan(dy_um / self.f_um)  # Steering angle in radians for y
        
        # 3. Calculate the blazed grating phase profile
        k = 2 * cp.pi / self.lam
        blazed_phase = k * (X_slm * cp.sin(theta_x) + Y_slm * cp.sin(theta_y))
        return (blazed_phase % (2*cp.pi) - cp.pi).astype(cp.float32)
    
    def superposition_optimization(self, target, max_iter=30, damping=0.4, verbose=True):
        """
        Optimizes a phasemask for a discrete array of optical tweezers using 
        the weighted superposition method, accelerated by the GPU using CuPy.
        """
            
        print("--- Starting GPU Superposition Phase Retrieval ---")
        start_time = time.time()

        # 3. Setup Illumination Beam (Gaussian)
        am_slm = cp.asarray(self.generate_source_amplitude())

        # 4. Define Target Trap Coordinates
        w_n, theta_n, x_n, y_n, array_shape = target
        
        # Move variables to GPU
        w_n_cp = cp.asarray(w_n)
        theta_n_cp = cp.asarray(theta_n)
        x_n_cp = cp.asarray(x_n)
        y_n_cp = cp.asarray(y_n)
        target_0_cp = cp.asarray(target[0])

        X_phase = cp.exp(self.j_k * x_n_cp[:, None] * self.x_slm[None, :]) # Shape: (N_traps, Nx)
        Y_phase = cp.exp(self.j_k * y_n_cp[:, None] * self.y_slm[None, :]) # Shape: (N_traps, Ny)
        X_phase_conj = cp.conj(X_phase)
        Y_phase_conj = cp.conj(Y_phase)

        uniformity_history = cp.zeros(max_iter)
        minmax_history = cp.zeros(max_iter)
        mse_history = cp.zeros(max_iter)
        
        mask_cp = target_0_cp > 0
        target_norm = target_0_cp / target_0_cp.sum()
        
        for iteration in range(max_iter):
            
            # Step A: Generate the Superposition Complex Field (Vectorized)
            C_n = w_n_cp * cp.exp(1j * theta_n_cp)
            Y_term = Y_phase * C_n[:, None]   # Shape: (N_traps, Ny)
            U_tot = Y_term.T @ X_phase      # Shape: (Ny, N_traps) @ (N_traps, Nx) -> (Ny, Nx)
            pm_slm = cp.angle(U_tot)
            
            # Step B: Evaluate Exact Intensity at Target Sites (Vectorized)
            U_slm = am_slm * cp.exp(1j * pm_slm)
            U_foc_n = U_slm @ X_phase_conj.T # Shape: (Ny, Nx) @ (Nx, N_traps) -> (Ny, N_traps)
            U_foc = cp.sum(Y_phase_conj * U_foc_n.T, axis=1) # Shape: (N_traps,)
            I_foc = cp.abs(U_foc)**2
                
            # Step C: Calculate Metrics
            I_foc_masked = I_foc[mask_cp]
            uniformity = 1.0 - (I_foc_masked.std() / I_foc_masked.mean()) # 1.0 is perfect uniformity
            uniformity_history[iteration] = uniformity
            
            I_foc_sum = I_foc.sum()
            I_foc_norm = I_foc / I_foc_sum
            mse = cp.mean((I_foc_norm - target_norm)**2)
            mse_history[iteration] = mse
            
            minmax_ratio = I_foc_masked.min() / I_foc_masked.max()
            minmax_history[iteration] = minmax_ratio
            
            # Step D: Update Weights
            update_ratio = (target_norm / I_foc_norm)**damping
            w_n_cp = w_n_cp * update_ratio
            
            # Update the target phases to match the simulated arriving field
            theta_n_cp = cp.angle(U_foc)
            
            if verbose:
                if iteration % 10 == 0:
                    print(f"Iteration {iteration:03d} | Mean-Squared Error: {float(mse):.2e} | Uniformity: {float(uniformity)*100:.2f}% | Min/Max ratio: {float(minmax_ratio):.3f}")

        trap_weights = I_foc.reshape(array_shape)
        print(f"Iteration {iteration:03d} | Mean-Squared Error: {float(mse):.2e} | Uniformity: {float(uniformity)*100:.2f}% | Min/Max ratio: {float(minmax_ratio):.3f}")
        print(f"Optimization finished in {time.time() - start_time:.2f} seconds.")
        
        return pm_slm, [w_n_cp, theta_n_cp, x_n_cp, y_n_cp, array_shape], [cp.asnumpy(uniformity_history), cp.asnumpy(minmax_history), cp.asnumpy(mse_history), cp.asnumpy(trap_weights)]

    def generate_phasemask(self, trap_terms):
        """
        Generates the phase mask using pure CuPy high-speed matrix multiplication.
        """
        # Unpack once
        w_n, theta_n, x_n, y_n, array_shape = trap_terms
        
        # 3. TRANSFER AND CAST TO 32-BIT
        # If your arrays are already on the GPU, cp.asarray does nothing (zero overhead)
        # Force float32 for maximum cuBLAS tensor core acceleration
        w_n_cp = cp.asarray(w_n, dtype=cp.float32)
        theta_n_cp = cp.asarray(theta_n, dtype=cp.float32)
        x_n_cp = cp.asarray(x_n, dtype=cp.float32)
        y_n_cp = cp.asarray(y_n, dtype=cp.float32)
        
        # 4. BUILD STEERING MATRICES (Using complex64 arithmetic)
        # Broadcasting: (N_traps, 1) * (1, Nx) -> (N_traps, Nx)
        X_phase = cp.exp(self.j_k * x_n_cp[:, None] * self.x_slm[None, :]) 
        Y_phase = cp.exp(self.j_k * y_n_cp[:, None] * self.y_slm[None, :]) 

        # 5. SUPERPOSITION COMPLEX FIELD
        # Combine weights and relative phases
        C_n = w_n_cp * cp.exp(cp.complex64(1j) * theta_n_cp)
        
        # Multiply weights into the Y_phase to create the Y_term: (N_traps, Ny)
        Y_term = Y_phase * C_n[:, None]   
        
        # The ultimate speedup: Matrix Multiplication (Ny, N_traps) @ (N_traps, Nx) -> (Ny, Nx)
        # This leverages NVIDIA's highly optimized cuBLAS backend instantly
        U_tot = Y_term.T @ X_phase      
        
        return cp.angle(U_tot).astype(cp.float32)

    def simulate_focal_plane(self, pm_slm, Nx_pad=2048, Ny_pad=2048, show=False, zoom_pixels=100, cmap='viridis'):
        am_slm = cp.asarray(self.generate_source_amplitude())
        field_slm = am_slm * cp.exp(1j * cp.asarray(pm_slm))
        field_slm_pad = cp.zeros((Ny_pad, Nx_pad), dtype=cp.complex128)
        field_slm_pad[(Ny_pad - self.Ny)//2:(Ny_pad + self.Ny)//2, (Nx_pad - self.Nx)//2:(Nx_pad + self.Nx)//2] = field_slm
        field_foc_pad = cp.asnumpy(cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(field_slm_pad))))

        if show:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            I_foc = np.abs(field_foc_pad)**2
            norm_intensity = I_foc / I_foc.max()
            x_focal_um = np.linspace(-self.Nx*self.dx_f/2, self.Nx*self.dx_f/2, Nx_pad)
            y_focal_um = np.linspace(-self.Ny*self.dy_f/2, self.Ny*self.dy_f/2, Ny_pad)
            im1 = ax.imshow(norm_intensity, 
                                extent=[x_focal_um[0], x_focal_um[-1], y_focal_um[0], y_focal_um[-1]],
                                cmap=cmap)

            zoom_range_um = zoom_pixels * self.dx_f
            ax.set_xlim(-zoom_range_um, zoom_range_um)
            ax.set_ylim(-zoom_range_um, zoom_range_um)
            ax.grid(False)
            ax.set_title(f"Simulated Intensity (Focal Plane)\nZoomed Central Region")
            ax.set_xlabel("x [um]")
            ax.set_ylabel("y [um]")
            cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
            cbar1.set_label("Normalized Intensity")

        return field_foc_pad
    
    def visualise_algorithm_performance(self, pm_slm, metrics, zoom_pixels, pad=1):
        U_foc = self.simulate_focal_plane(pm_slm, Nx_pad=self.Nx*pad, Ny_pad=self.Ny*pad, show=False)
        I_foc = np.abs(U_foc)**2

        fig, ax = plt.subplots(1, 3, figsize=(21, 6))
        uniformity_history, minmax_history, mse_history, trap_weights = metrics

        ax[0].plot(uniformity_history, 'b-o', label="Uniformity")
        ax[0].plot(minmax_history, 'r-s', label="Min/Max Ratio")
        ax0_secondary = ax[0].twinx()
        ax0_secondary.plot(mse_history, 'g-o', label="MSE")
        ax0_secondary.set_ylabel("MSE Value")
        ax0_secondary.tick_params(axis='y')
        ax0_secondary.legend(loc='lower right')
        ax[0].set_xlabel("Iteration")
        ax[0].set_ylabel("Metric Value")
        ax[0].legend(loc='upper right')
        ax[0].grid(True)
        
        ax[0].set_title("Optimization Convergence Metrics")

        norm_intensity = I_foc / I_foc.max()
        x_focal_um = np.linspace(-self.Nx*self.dx_f/2, self.Nx*self.dx_f/2, self.Nx*pad)
        y_focal_um = np.linspace(-self.Ny*self.dy_f/2, self.Ny*self.dy_f/2, self.Ny*pad)
        im1 = ax[1].imshow(norm_intensity, 
                            extent=[x_focal_um[0], x_focal_um[-1], y_focal_um[0], y_focal_um[-1]],
                            cmap='viridis')

        zoom_range_um = zoom_pixels * self.dx_f
        ax[1].set_xlim(-zoom_range_um, zoom_range_um)
        ax[1].set_ylim(-zoom_range_um, zoom_range_um)
        ax[1].grid(False)
        ax[1].set_title(f"Simulated Intensity (Focal Plane)\nZoomed Central Region")
        ax[1].set_xlabel("x [um]")
        ax[1].set_ylabel("y [um]")
        cbar1 = plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
        cbar1.set_label("Normalized Intensity")

        # Plot a heatmap of the trap weights with ax[2]
        im2 = ax[2].imshow(trap_weights/trap_weights.max(), cmap='viridis')
        ax[2].grid(False)
        ax[2].set_title("Simulated Trap Weights")
        ax[2].set_xlabel("Trap X Index")
        ax[2].set_ylabel("Trap Y Index")
        cbar2 = plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
        cbar2.set_label("Normalized Trap Weight")

        plt.tight_layout()
        plt.show()

    def save_phasemask(self, phasemask):
        phasemask_bmp = Image.fromarray(phasemask)
        phasemask_bmp.save('C:\\Users\\CaFMOT\\OneDrive - Imperial College London\\caftweezers\\MeadowController\\phasemasks\\phasemask.bmp')
        print('Phasemask generated and saved as an 8bit.bmp')

    def superimpose(self, phasemasks):
        return cp.mod(sum(phasemasks), 2*cp.pi)
    
    def transform_phase_8bit(self, phasemask):
        return ((phasemask + cp.pi) * self.uint8_scale).astype(cp.uint8)

    def generate_rearrangement_sequence(self, terms1, terms2, occ_mask, d0=0.5):
            """
            Calculates the optimal Hungarian rearrangement path and generates 
            the full sequence of interpolated phasemasks efficiently on the GPU.
            """
            start = time.time()
        
            w1, phi1, x1, y1, arr1 = terms1
            w2, phi2, x2, y2, _ = terms2
            occ_mask = cp.asarray(occ_mask)
            
            pos1 = cp.stack((x1, y1), axis=-1)
            pos2 = cp.stack((x2, y2), axis=-1)

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
            n_steps = int(cp.ceil(1.875 * max_dist / d0))
                
            # 3. INITIALIZE VRAM STATE MACHINE
            curr_w = cp.asarray(w1, dtype=cp.float32)
            curr_phi = cp.asarray(phi1, dtype=cp.float32)
            curr_x = cp.asarray(x1, dtype=cp.float32)
            curr_y = cp.asarray(y1, dtype=cp.float32)

            # Pre-calculate the minimum jerk step multipliers on the GPU
            tau = cp.linspace(0, 1, n_steps + 1, dtype=cp.float32)
            s_profile = 10 * tau**3 - 15 * tau**4 + 6 * tau**5

            # ds_profile contains the fractional progression for each step n
            ds_profile = cp.diff(s_profile)
            
            # CRITICAL: Pull ds_profile back to the CPU! 
            # Accessing GPU array scalars inside a loop causes implicit device synchronizations.
            ds_profile_cpu = ds_profile.get()
            
            # Initialize step vectors with zeros
            dw = cp.zeros_like(curr_w)
            dphi = cp.zeros_like(curr_phi)
            total_dx = cp.zeros_like(curr_x)
            total_dy = cp.zeros_like(curr_y)
            
            # Ensure array operands are on the GPU to avoid implicit CPU conversion
            w1_gpu, w2_gpu = cp.asarray(w1), cp.asarray(w2)
            phi1_gpu, phi2_gpu = cp.asarray(phi1), cp.asarray(phi2)

            # Load steps for MOVING traps
            dw[moving_idx] = (w2_gpu[final_idx] - w1_gpu[moving_idx]) / n_steps
            total_dx[moving_idx] = vec[:, 0].astype(cp.float32)
            total_dy[moving_idx] = vec[:, 1].astype(cp.float32)
            
            # Ensure Phase Interpolation takes the shortest angular path to prevent wrapping tears
            # Used cp.pi instead of np.pi to keep the modulo arithmetic entirely on the GPU
            phase_diff = (phi2_gpu[final_idx] - phi1_gpu[moving_idx] + cp.pi) % (2 * cp.pi) - cp.pi
            dphi[moving_idx] = phase_diff / n_steps
            
            # Load steps for OFF traps (Ramp down weights to 0)
            dw[off_mask] = 0.0
            curr_w[off_mask] = 0.0
            
            phasemasks_sequence = np.empty((n_steps, self.Ny, self.Nx), dtype=np.uint8)
            gpu_sequence = cp.empty((n_steps, self.Ny, self.Nx), dtype=cp.uint8)
            
            # OPTIMIZATION: Pre-calculate the static background mask ONCE
            static_background = self.superimpose([self.fresnel, self.blaze, self.zernike])
            
            # 4. THE ULTRA-FAST GPU LOOP
            for n in range(n_steps):
                
                # Use the fused Elementwise kernel: 1 kernel launch instead of 4
                # We cast ds to float so it's passed as a fast C-scalar to the kernel
                ds = float(ds_profile_cpu[n])
                update_state_kernel(dw, dphi, total_dx, total_dy, ds, curr_w, curr_phi, curr_x, curr_y)
                
                # Repack the terms and call the generator.
                terms_gpu = (curr_w, curr_phi, curr_x, curr_y, arr1)
                pm_slm = self.generate_phasemask(terms_gpu)
                
                # Only superimpose the moving traps with the pre-calculated static background
                composite_pm = self.superimpose([pm_slm, static_background])
                
                # Store calculated 2D mask directly into pre-allocated VRAM chunk
                gpu_sequence[n] = self.transform_phase_8bit(composite_pm)

            # Batch copy the entire sequence back to the host (CPU) once at the end
            gpu_sequence.get(out=phasemasks_sequence)
                
            print(f"Time Taken for {n_steps} frames: {(time.time() - start)*1000:.4f} ms")
            return phasemasks_sequence

    def iter_rearrangement_sequence(self, terms1, terms2, occ_mask, d0=0.5,
                                    profile="minimum_jerk", to_host=True):
        """Streaming variant of :meth:`generate_rearrangement_sequence`.

        Yields each interpolated phasemask the moment it is computed, instead of
        buffering the whole ``(n, Ny, Nx)`` sequence and returning it at the end.
        This lets a caller push each frame to the SLM while the GPU keeps
        generating the rest, so generation and upload overlap (pipelined) rather
        than running back to back.

        ``profile`` selects the transport trajectory:

        * ``"minimum_jerk"`` - quintic profile, ``n = ceil(1.875 * max_dist / d0)``
          frames. Smoother acceleration, so gentler on the atoms.
        * ``"linear"`` - constant velocity, ``n = ceil(max_dist / d0)`` frames.
          1.875x fewer frames for the same ``d0``, at the cost of abrupt start/stop.

        ``to_host`` controls where the GPU->host copy happens. ``True`` yields
        ``numpy`` frames (the copy runs here). ``False`` yields ``cupy`` frames so a
        consumer thread can do the ``.get()`` itself, keeping the PCIe transfer off
        this loop.
        """
        if profile not in ("minimum_jerk", "linear"):
            raise ValueError(
                f"profile must be 'minimum_jerk' or 'linear', got {profile!r}"
            )
        w1, phi1, x1, y1, arr1 = terms1
        w2, phi2, x2, y2, _ = terms2
        occ_mask = cp.asarray(occ_mask)

        pos1 = cp.stack((x1, y1), axis=-1)
        pos2 = cp.stack((x2, y2), axis=-1)

        occ_indices = cp.where(occ_mask)[0]
        init = pos1[occ_indices]
        final = pos2

        init_idx, final_idx = get_jv_pairing_lap(init, final)
        moving_idx = occ_indices[init_idx]

        off_mask = cp.ones(len(pos1), dtype=bool)
        off_mask[moving_idx] = False

        pos_init = pos1[moving_idx]
        pos_final = pos2[final_idx]
        vec = pos_final - pos_init

        max_dist = cp.linalg.norm(vec, axis=1).max()
        steps_scale = 1.0 if profile == "linear" else 1.875
        n_steps = int(cp.ceil(steps_scale * max_dist / d0))

        curr_w = cp.asarray(w1, dtype=cp.float32)
        curr_phi = cp.asarray(phi1, dtype=cp.float32)
        curr_x = cp.asarray(x1, dtype=cp.float32)
        curr_y = cp.asarray(y1, dtype=cp.float32)

        tau = cp.linspace(0, 1, n_steps + 1, dtype=cp.float32)
        if profile == "linear":
            s_profile = tau
        else:
            s_profile = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        ds_profile_cpu = cp.diff(s_profile).get()

        dw = cp.zeros_like(curr_w)
        dphi = cp.zeros_like(curr_phi)
        total_dx = cp.zeros_like(curr_x)
        total_dy = cp.zeros_like(curr_y)

        w1_gpu, w2_gpu = cp.asarray(w1), cp.asarray(w2)
        phi1_gpu, phi2_gpu = cp.asarray(phi1), cp.asarray(phi2)

        dw[moving_idx] = (w2_gpu[final_idx] - w1_gpu[moving_idx]) / n_steps
        total_dx[moving_idx] = vec[:, 0].astype(cp.float32)
        total_dy[moving_idx] = vec[:, 1].astype(cp.float32)

        phase_diff = (phi2_gpu[final_idx] - phi1_gpu[moving_idx] + cp.pi) % (2 * cp.pi) - cp.pi
        dphi[moving_idx] = phase_diff / n_steps

        dw[off_mask] = 0.0
        curr_w[off_mask] = 0.0

        static_background = self.superimpose([self.fresnel, self.blaze, self.zernike])

        for n in range(n_steps):
            ds = float(ds_profile_cpu[n])
            update_state_kernel(dw, dphi, total_dx, total_dy, ds, curr_w, curr_phi, curr_x, curr_y)
            terms_gpu = (curr_w, curr_phi, curr_x, curr_y, arr1)
            pm_slm = self.generate_phasemask(terms_gpu)
            composite_pm = self.superimpose([pm_slm, static_background])
            frame = self.transform_phase_8bit(composite_pm)
            yield frame.get() if to_host else frame

# Define this fused kernel outside the loop (or at the module level).
# CuPy will compile it once on the first run and cache it, saving massive
# overhead by doing 4 state updates in a single GPU kernel launch.
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


################################################################################################################################################################
################################################################################################################################################################

class OptimisationBasedPhasemaskGeneratorGPU3D:
    def __init__(self,
                 wavelength_um=0.852,
                 focal_length_mm=17.3,
                 slm_pitch_um=17,
                 slm_res=(1024,1024),
                 input_beam_waist_mm=16,
                 fresnel_f_mm=1072,
                 blaze_dx_dy_um=(40.0, -8.0),
                 zernike_coeff_dict={5:1.195, 6:0.725, 7:0.970, 8:0.478, 9:-1.091, 10:0.303, 11:0.021, 12:0.072, 13:0.049}):
        
        self.lam = wavelength_um
        self.f_mm = focal_length_mm
        self.f_um = focal_length_mm * 1000
        self.dx_slm = slm_pitch_um
        self.Ny, self.Nx = slm_res
        self.w0_mm = input_beam_waist_mm
        self.w0_um = input_beam_waist_mm * 1000
        self.Lx_slm = self.Nx * self.dx_slm # In units of um
        self.Ly_slm = self.Ny * self.dx_slm # In units of um
        self.dx_f = (self.lam * self.f_um) / self.Lx_slm # In units of um
        self.dy_f = (self.lam * self.f_um) / self.Ly_slm # In units of um
        self.Lx_foc = self.Nx * self.dx_f
        self.Ly_foc = self.Ny * self.dy_f

        self.x_slm = cp.linspace(-self.Lx_slm/2, self.Lx_slm/2, self.Nx, dtype=cp.float32)
        self.y_slm = cp.linspace(-self.Ly_slm/2, self.Ly_slm/2, self.Ny, dtype=cp.float32)

        # Precalculations
        self.k = np.float32(2 * np.pi / (self.lam * self.f_um))
        self.j_k = cp.complex64(1j) * self.k
        self.z_scale = cp.complex64(1j) * cp.pi / (self.lam * self.f_um **2)
        self.uint8_scale = cp.float32(255.0) / (2 * cp.pi)

        # Additional phasemasks - phase wrapped and in radians
        self.fresnel = self.generate_fresnel_lens_phasemask(focal_length_mm=fresnel_f_mm)
        self.blaze = self.generate_blazed_grating_phasemask(dx_um=blaze_dx_dy_um[0], dy_um=blaze_dx_dy_um[1])
        self.zernike = self.generate_zernike_phasemask(zernike_coeff_dict, wrap=True)
        
        print(f"--- System Configuration ---")
        print(f"SLM Plane Width: {self.Lx_slm/1000:.2f} mm")
        print(f"SLM Plane Height: {self.Ly_slm/1000:.2f} mm")
        print(f"Focal Plane Resolution x (pixel size): {self.dx_f:.4f} um")
        print(f"Focal Plane Resolution y (pixel size): {self.dy_f:.4f} um")
        print(f"Focal Plane Width: {self.Lx_foc:.2f} um")
        print(f"Focal Plane Height: {self.Ly_foc:.2f} um")
        print(f"Fresnel Lens Focal Length: {fresnel_f_mm:.2f} mm")
        print(f"Blazed Grating Displacement (dx, dy): {blaze_dx_dy_um} um")

    def generate_source_amplitude(self):
        """Generates Gaussian input beam amplitude."""
        X, Y = cp.meshgrid(self.x_slm, self.y_slm)
        R2 = X**2 + Y**2
        return cp.exp(- R2 / (2*(self.w0_um/2.355)**2))
    
    def generate_weighted_array(self, weights, spacing, init_phase_randomness=1.0, angle_deg=0):
        """
        Generates a weighted mask indicating where the tweezers should be and
        how strong they should be.
        Returns both the full 2D array and a list of the specific (y, x) indices.
        """
        dim = weights.shape
        xspan = (dim[1] - 1) * spacing
        yspan = (dim[0] - 1) * spacing
        xpos = np.linspace(-(xspan)/2, (xspan)/2, dim[1])
        ypos = np.linspace(-(yspan)/2, (yspan)/2, dim[0])
        Xpos, Ypos = np.meshgrid(xpos, ypos)
        Xn, Yn = Xpos.flatten(), Ypos.flatten()
        Xn, Yn = rotate_coordinates(Xn, Yn, angle_deg)
        Zn = np.zeros_like(Xn) # For 3D extension, currently all traps are in the same focal plane (z=0)    
        Wn = weights.flatten()
        Thetan = np.random.rand(len(Wn)) * init_phase_randomness * 2 * np.pi

        print(f"--- Target Generation ---")
        print(f"Grid: {dim[0]}x{dim[1]}")
        print(f"Spacing: {spacing} um")
                    
        return [Wn, Thetan, Xn, Yn, Zn, dim]
    
    def generate_zernike_phasemask(self, zernike_coeffs, wrap=False):
        """
        Generates a 2D Zernike correction phasemask defined on the SLM plane.
        
        Parameters:
        -----------
        settings : dict
            Must contain 'slm_shape' (Ny, Nx) and 'pixel_pitch' (p).
        zernike_coeffs : dict
            A dictionary mapping the Noll Index (int) to its coefficient (float).
            Example: {4: 1.5, 8: -0.5} applies 1.5 rads of Defocus and -0.5 rads of Horizontal Coma.
            
        Returns:
        --------
        total_phase : np.ndarray
            The 2D phase mask in radians to be applied to the SLM.
        """

        # 1. Setup SLM Coordinates (Centered at 0)
        X_slm, Y_slm = cp.meshgrid(self.x_slm, self.y_slm)
        R_max = cp.sqrt(self.Lx_slm**2 + self.Ly_slm**2) / 2.0
        rho = cp.sqrt(X_slm**2 + Y_slm**2) / R_max
        theta = cp.arctan2(Y_slm, X_slm)
        
        # 3. Define the unit circle mask
        mask = (rho <= 1.0).astype(float)
        
        # Initialize an empty phase array
        total_phase = cp.zeros((self.Ny, self.Nx))
        
        # 4. Superpose the requested Zernike modes
        for noll_index, coeff in zernike_coeffs.items():
            if coeff != 0.0:
                Z_mode = get_zernike_polynomial(noll_index, rho, theta, mask)
                total_phase += coeff * Z_mode

        return total_phase % (2 * cp.pi) - cp.pi

    def generate_fresnel_lens_phasemask(self, focal_length_mm):
        """
        Generates a Fresnel lens phase mask to focus the beam at a specific focal length.
        This is useful for correcting defocus or creating a virtual focus plane.
        """
        # 1. Setup SLM Coordinates (Centered at 0)
        X_slm, Y_slm = cp.meshgrid(self.x_slm, self.y_slm)
        
        # 2. Calculate the Fresnel lens phase profile
        f_um = focal_length_mm * 1000
        k = 2 * cp.pi / self.lam
        R_squared = X_slm**2 + Y_slm**2
        fresnel_phase = (k / (2 * f_um)) * R_squared
        
        return fresnel_phase % (2*cp.pi) - cp.pi
    
    def generate_blazed_grating_phasemask(self, dx_um, dy_um):
        """
        Generates a blazed grating phase mask to steer the beam by (dx_um, dy_um) in the focal plane.
        The steering angles are calculated based on the desired displacement and the system's focal length.
        """
        # 1. Setup SLM Coordinates (Centered at 0)
        X_slm, Y_slm = cp.meshgrid(self.x_slm, self.y_slm)
        
        # 2. Calculate steering angles
        theta_x = cp.arctan(dx_um / self.f_um)  # Steering angle in radians for x
        theta_y = cp.arctan(dy_um / self.f_um)  # Steering angle in radians for y
        
        # 3. Calculate the blazed grating phase profile
        k = 2 * cp.pi / self.lam
        blazed_phase = k * (X_slm * cp.sin(theta_x) + Y_slm * cp.sin(theta_y))
        return blazed_phase % (2*cp.pi) - cp.pi
    
    def superposition_optimization(self, target, max_iter=30, damping=0.4, verbose=True):
        """
        Optimizes a phasemask for a discrete array of 3D optical tweezers using 
        the weighted superposition method, accelerated by the GPU using CuPy.
        """
            
        print("--- Starting GPU Superposition Phase Retrieval (3D) ---")
        start_time = time.time()

        # Setup Illumination Beam (Gaussian)
        am_slm = cp.asarray(self.generate_source_amplitude())

        # Define Target Trap Coordinates (Now utilizing z_n)
        w_n, theta_n, x_n, y_n, z_n, array_shape = target
        
        # Move variables to GPU
        w_n_cp = cp.asarray(w_n)
        theta_n_cp = cp.asarray(theta_n)
        x_n_cp = cp.asarray(x_n)
        y_n_cp = cp.asarray(y_n)
        z_n_cp = cp.asarray(z_n)
        target_0_cp = cp.asarray(target[0])
        
        X_phase = cp.exp(self.j_k * x_n_cp[:, None] * self.x_slm[None, :] + 
                         self.z_scale * z_n_cp[:, None] * (self.x_slm**2)[None, :])
        
        Y_phase = cp.exp(self.j_k * y_n_cp[:, None] * self.y_slm[None, :] + 
                         self.z_scale * z_n_cp[:, None] * (self.y_slm**2)[None, :])

        X_phase_conj = cp.conj(X_phase)
        Y_phase_conj = cp.conj(Y_phase)

        uniformity_history = cp.zeros(max_iter)
        minmax_history = cp.zeros(max_iter)
        mse_history = cp.zeros(max_iter)
        
        mask_cp = target_0_cp > 0
        target_norm = target_0_cp / target_0_cp.sum()
        
        for iteration in range(max_iter):
            
            # Step A: Generate the Superposition Complex Field
            C_n = w_n_cp * cp.exp(1j * theta_n_cp)
            Y_term = Y_phase * C_n[:, None]   
            U_tot = Y_term.T @ X_phase      
            pm_slm = cp.angle(U_tot)
            
            # Step B: Evaluate Exact Intensity at Target Sites
            # Because X_phase and Y_phase now contain the z-defocus, multiplying by 
            # their conjugates exactly simulates propagation to that specific z-plane!
            U_slm = am_slm * cp.exp(1j * pm_slm)
            U_foc_n = U_slm @ X_phase_conj.T 
            U_foc = cp.sum(Y_phase_conj * U_foc_n.T, axis=1) 
            I_foc = cp.abs(U_foc)**2
                
            # Step C: Calculate Metrics
            I_foc_masked = I_foc[mask_cp]
            uniformity = 1.0 - (I_foc_masked.std() / I_foc_masked.mean())
            uniformity_history[iteration] = uniformity
            
            I_foc_sum = I_foc.sum()
            I_foc_norm = I_foc / I_foc_sum
            mse = cp.mean((I_foc_norm - target_norm)**2)
            mse_history[iteration] = mse
            
            minmax_ratio = I_foc_masked.min() / I_foc_masked.max()
            minmax_history[iteration] = minmax_ratio
            
            # Step D: Update Weights
            update_ratio = (target_norm / I_foc_norm)**damping
            w_n_cp = w_n_cp * update_ratio
            
            # Update the target phases to match the simulated arriving field
            theta_n_cp = cp.angle(U_foc)
            
            if verbose:
                if iteration % 10 == 0:
                    print(f"Iteration {iteration:03d} | Mean-Squared Error: {float(mse):.2e} | Uniformity: {float(uniformity)*100:.2f}% | Min/Max ratio: {float(minmax_ratio):.3f}")

        # Ensure array_shape matches 3D output if necessary, or keep as 1D list of traps
        trap_weights = I_foc.reshape(array_shape) 
        print(f"Iteration {iteration:03d} | Mean-Squared Error: {float(mse):.2e} | Uniformity: {float(uniformity)*100:.2f}% | Min/Max ratio: {float(minmax_ratio):.3f}")
        print(f"Optimization finished in {time.time() - start_time:.2f} seconds.")
        
        return pm_slm, [w_n_cp, theta_n_cp, x_n_cp, y_n_cp, z_n_cp, array_shape], [cp.asnumpy(uniformity_history), cp.asnumpy(minmax_history), cp.asnumpy(mse_history), cp.asnumpy(trap_weights)]

    def generate_phasemask(self, trap_terms):
        """
        Generates the phase mask using pure CuPy high-speed matrix multiplication.
        """
        # Unpack once
        w_n, theta_n, x_n, y_n, z_n, array_shape = trap_terms
        
        # 3. TRANSFER AND CAST TO 32-BIT
        w_n_cp = cp.asarray(w_n, dtype=cp.float32)
        theta_n_cp = cp.asarray(theta_n, dtype=cp.float32)
        x_n_cp = cp.asarray(x_n, dtype=cp.float32)
        y_n_cp = cp.asarray(y_n, dtype=cp.float32)
        z_n_cp = cp.asarray(z_n, dtype=cp.float32)
        
        # 4. BUILD STEERING MATRICES (Using complex64 arithmetic)
        X_phase = cp.exp(self.j_k * x_n_cp[:, None] * self.x_slm[None, :] + 
                         self.z_scale * z_n_cp[:, None] * (self.x_slm**2)[None, :])
        
        Y_phase = cp.exp(self.j_k * y_n_cp[:, None] * self.y_slm[None, :] + 
                         self.z_scale * z_n_cp[:, None] * (self.y_slm**2)[None, :])

        # 5. SUPERPOSITION COMPLEX FIELD
        C_n = w_n_cp * cp.exp(cp.complex64(1j) * theta_n_cp)
        Y_term = Y_phase * C_n[:, None]   
        U_tot = Y_term.T @ X_phase      
        
        return cp.angle(U_tot)

    def simulate_focal_plane(self, pm_slm, Nx_pad=2048, Ny_pad=2048, show=False, zoom_pixels=100, cmap='viridis'):
        am_slm = cp.asarray(self.generate_source_amplitude())
        field_slm = am_slm * cp.exp(1j * cp.asarray(pm_slm))
        field_slm_pad = cp.zeros((Ny_pad, Nx_pad), dtype=cp.complex128)
        field_slm_pad[(Ny_pad - self.Ny)//2:(Ny_pad + self.Ny)//2, (Nx_pad - self.Nx)//2:(Nx_pad + self.Nx)//2] = field_slm
        field_foc_pad = cp.asnumpy(cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(field_slm_pad))))

        if show:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            I_foc = np.abs(field_foc_pad)**2
            norm_intensity = I_foc / I_foc.max()
            x_focal_um = np.linspace(-self.Nx*self.dx_f/2, self.Nx*self.dx_f/2, Nx_pad)
            y_focal_um = np.linspace(-self.Ny*self.dy_f/2, self.Ny*self.dy_f/2, Ny_pad)
            im1 = ax.imshow(norm_intensity, 
                                extent=[x_focal_um[0], x_focal_um[-1], y_focal_um[0], y_focal_um[-1]],
                                cmap=cmap)

            zoom_range_um = zoom_pixels * self.dx_f
            ax.set_xlim(-zoom_range_um, zoom_range_um)
            ax.set_ylim(-zoom_range_um, zoom_range_um)
            ax.grid(False)
            ax.set_title(f"Simulated Intensity (Focal Plane)\nZoomed Central Region")
            ax.set_xlabel("x [um]")
            ax.set_ylabel("y [um]")
            cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
            cbar1.set_label("Normalized Intensity")

        return field_foc_pad
    
    def visualise_algorithm_performance(self, pm_slm, metrics, zoom_pixels, pad=1):
        U_foc = self.simulate_focal_plane(pm_slm, Nx_pad=self.Nx*pad, Ny_pad=self.Ny*pad, show=False)
        I_foc = np.abs(U_foc)**2

        fig, ax = plt.subplots(1, 3, figsize=(21, 6))
        uniformity_history, minmax_history, mse_history, trap_weights = metrics

        ax[0].plot(uniformity_history, 'b-o', label="Uniformity")
        ax[0].plot(minmax_history, 'r-s', label="Min/Max Ratio")
        ax0_secondary = ax[0].twinx()
        ax0_secondary.plot(mse_history, 'g-o', label="MSE")
        ax0_secondary.set_ylabel("MSE Value")
        ax0_secondary.tick_params(axis='y')
        ax0_secondary.legend(loc='lower right')
        ax[0].set_xlabel("Iteration")
        ax[0].set_ylabel("Metric Value")
        ax[0].legend(loc='upper right')
        ax[0].grid(True)
        
        ax[0].set_title("Optimization Convergence Metrics")

        norm_intensity = I_foc / I_foc.max()
        x_focal_um = np.linspace(-self.Nx*self.dx_f/2, self.Nx*self.dx_f/2, self.Nx*pad)
        y_focal_um = np.linspace(-self.Ny*self.dy_f/2, self.Ny*self.dy_f/2, self.Ny*pad)
        im1 = ax[1].imshow(norm_intensity, 
                            extent=[x_focal_um[0], x_focal_um[-1], y_focal_um[0], y_focal_um[-1]],
                            cmap='viridis')

        zoom_range_um = zoom_pixels * self.dx_f
        ax[1].set_xlim(-zoom_range_um, zoom_range_um)
        ax[1].set_ylim(-zoom_range_um, zoom_range_um)
        ax[1].grid(False)
        ax[1].set_title(f"Simulated Intensity (Focal Plane)\nZoomed Central Region")
        ax[1].set_xlabel("x [um]")
        ax[1].set_ylabel("y [um]")
        cbar1 = plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
        cbar1.set_label("Normalized Intensity")

        # Plot a heatmap of the trap weights with ax[2]
        im2 = ax[2].imshow(trap_weights/trap_weights.max(), cmap='viridis')
        ax[2].grid(False)
        ax[2].set_title("Simulated Trap Weights")
        ax[2].set_xlabel("Trap X Index")
        ax[2].set_ylabel("Trap Y Index")
        cbar2 = plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
        cbar2.set_label("Normalized Trap Weight")

        plt.tight_layout()
        plt.show()

    def save_phasemask(self, phasemask):
        phasemask_bmp = Image.fromarray(phasemask)
        phasemask_bmp.save('C:\\Users\\CaFMOT\\OneDrive - Imperial College London\\caftweezers\\MeadowController\\phasemasks\\phasemask.bmp')
        print('Phasemask generated and saved as an 8bit.bmp')

    def superimpose(self, phasemasks):
        return cp.mod(sum(phasemasks), 2*cp.pi)
    
    def transform_phase_8bit(self, phasemask):
        return ((phasemask + cp.pi) * self.uint8_scale).astype(cp.uint8)

    def generate_rearrangement_sequence(self, terms1, terms2, occ_mask, d0=0.5):
            """
            Calculates the optimal Hungarian rearrangement path and generates 
            the full sequence of interpolated phasemasks efficiently on the GPU.
            """
            start = time.time()
        
            w1, phi1, x1, y1, z1, arr1 = terms1
            w2, phi2, x2, y2, z2, arr2 = terms2
            occ_mask = cp.asarray(occ_mask)
            
            pos1 = cp.stack((x1, y1), axis=-1)
            pos2 = cp.stack((x2, y2), axis=-1)

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
            vec_z = z2[final_idx] - z1[moving_idx]
            
            # Faster L2 norm calculation using cupy linear algebra
            max_dist = cp.linalg.norm(vec, axis=1).max()
            n_steps = int(cp.ceil(1.875 * max_dist / d0))
                
            # 3. INITIALIZE VRAM STATE MACHINE
            curr_w = cp.asarray(w1, dtype=cp.float32)
            curr_phi = cp.asarray(phi1, dtype=cp.float32)
            curr_x = cp.asarray(x1, dtype=cp.float32)
            curr_y = cp.asarray(y1, dtype=cp.float32)
            curr_z = cp.asarray(z1, dtype=cp.float32)

            # Pre-calculate the minimum jerk step multipliers on the GPU
            tau = cp.linspace(0, 1, n_steps + 1, dtype=cp.float32)
            s_profile = 10 * tau**3 - 15 * tau**4 + 6 * tau**5

            # ds_profile contains the fractional progression for each step n
            ds_profile = cp.diff(s_profile)
            
            # CRITICAL: Pull ds_profile back to the CPU! 
            # Accessing GPU array scalars inside a loop causes implicit device synchronizations.
            ds_profile_cpu = ds_profile.get()
            
            # Initialize step vectors with zeros
            dw = cp.zeros_like(curr_w)
            dphi = cp.zeros_like(curr_phi)
            total_dx = cp.zeros_like(curr_x)
            total_dy = cp.zeros_like(curr_y)
            total_dz = cp.zeros_like(curr_z)
            
            # Ensure array operands are on the GPU to avoid implicit CPU conversion
            w1_gpu, w2_gpu = cp.asarray(w1), cp.asarray(w2)
            phi1_gpu, phi2_gpu = cp.asarray(phi1), cp.asarray(phi2)
            z1_gpu, z2_gpu = cp.asarray(z1), cp.asarray(z2)

            # Load steps for MOVING traps
            dw[moving_idx] = (w2_gpu[final_idx] - w1_gpu[moving_idx]) / n_steps
            total_dx[moving_idx] = vec[:, 0].astype(cp.float32)
            total_dy[moving_idx] = vec[:, 1].astype(cp.float32)
            total_dz[moving_idx] = vec_z.astype(cp.float32)

            # Ensure Phase Interpolation takes the shortest angular path to prevent wrapping tears
            # Used cp.pi instead of np.pi to keep the modulo arithmetic entirely on the GPU
            phase_diff = (phi2_gpu[final_idx] - phi1_gpu[moving_idx] + cp.pi) % (2 * cp.pi) - cp.pi
            dphi[moving_idx] = phase_diff / n_steps
            
            # Load steps for OFF traps (Ramp down weights to 0)
            dw[off_mask] = 0.0
            curr_w[off_mask] = 0.0
            
            phasemasks_sequence = np.empty((n_steps, self.Ny, self.Nx), dtype=np.uint8)
            gpu_sequence = cp.empty((n_steps, self.Ny, self.Nx), dtype=cp.uint8)
            
            # OPTIMIZATION: Pre-calculate the static background mask ONCE
            static_background = self.superimpose([self.fresnel, self.blaze, self.zernike])
            
            # 4. THE ULTRA-FAST GPU LOOP
            for n in range(n_steps):
                
                # Use the fused Elementwise kernel: 1 kernel launch instead of 4
                # We cast ds to float so it's passed as a fast C-scalar to the kernel
                ds = float(ds_profile_cpu[n])
                update_state_kernel_3D(dw, dphi, total_dx, total_dy, total_dz, ds, curr_w, curr_phi, curr_x, curr_y, curr_z)
                
                # Repack the terms and call the generator.
                terms_gpu = (curr_w, curr_phi, curr_x, curr_y, curr_z, arr1)
                pm_slm = self.generate_phasemask(terms_gpu)
                
                # Only superimpose the moving traps with the pre-calculated static background
                composite_pm = self.superimpose([pm_slm, static_background])
                
                # Store calculated 2D mask directly into pre-allocated VRAM chunk
                gpu_sequence[n] = self.transform_phase_8bit(composite_pm)

            # Batch copy the entire sequence back to the host (CPU) once at the end
            gpu_sequence.get(out=phasemasks_sequence)
                
            print(f"Time Taken for {n_steps} frames: {(time.time() - start)*1000:.4f} ms")
            return phasemasks_sequence
    
update_state_kernel_3D = cp.ElementwiseKernel(
    'float32 dw, float32 dphi, float32 dx, float32 dy, float32 dz, float32 ds',
    'float32 w, float32 phi, float32 x, float32 y, float32 z',
    '''
    w += dw;
    phi += dphi;
    x += dx * ds;
    y += dy * ds;
    z += dz * ds;
    ''',
    'update_hologram_state'
    )