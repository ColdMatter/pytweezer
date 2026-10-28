import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
from functools import partial

import lap

DEFAULT_ZERNIKE_COEFFS = {
    5: 1.195,
    6: 0.725,
    7: 0.970,
    8: 0.478,
    9: -1.091,
    10: 0.303,
    11: 0.021,
    12: 0.072,
    13: 0.049,
}


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
        raise ValueError(
            f"Noll index {noll_index} is not implemented in this basic dictionary."
        )

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
    cost_matrix = cp.sum((init[:, None, :] - final[None, :, :]) ** 2, axis=-1)

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
    def __init__(
        self,
        wavelength_um=0.852,
        focal_length_mm=17.3,
        slm_pitch_um=17,
        slm_res=(1024, 1024),
        input_beam_waist_mm=16,
        fresnel_f_mm=1072,
        blaze_dx_dy_um=(46.50, 10.54),
        zernike_coeff_dict=None,
    ):

        self.lam = wavelength_um
        self.f_mm = focal_length_mm
        self.f_um = focal_length_mm * 1000
        self.dx_slm = slm_pitch_um
        self.Ny, self.Nx = slm_res
        self.w0_mm = input_beam_waist_mm
        self.w0_um = input_beam_waist_mm * 1000
        self.Lx_slm = self.Nx * self.dx_slm  # In units of um
        self.Ly_slm = self.Ny * self.dx_slm  # In units of um
        self.dx_f = (self.lam * self.f_um) / self.Lx_slm  # In units of um
        self.dy_f = (self.lam * self.f_um) / self.Ly_slm  # In units of um
        self.Lx_foc = self.Nx * self.dx_f
        self.Ly_foc = self.Ny * self.dy_f

        self.x_slm = cp.linspace(
            -self.Lx_slm / 2, self.Lx_slm / 2, self.Nx, dtype=cp.float32
        )
        self.y_slm = cp.linspace(
            -self.Ly_slm / 2, self.Ly_slm / 2, self.Ny, dtype=cp.float32
        )

        # Precalculations
        self.k = np.float32(2 * np.pi / (self.lam * self.f_um))
        self.j_k = cp.complex64(1j) * self.k
        self.uint8_scale = cp.float32(255.0) / (2 * cp.pi)

        # Additional phasemasks - phase wrapped and in radians
        self.fresnel = self.generate_fresnel_lens_phasemask(
            focal_length_mm=fresnel_f_mm
        )
        self.blaze = self.generate_blazed_grating_phasemask(
            dx_um=blaze_dx_dy_um[0], dy_um=blaze_dx_dy_um[1]
        )
        if zernike_coeff_dict is None:
            zernike_coeff_dict = DEFAULT_ZERNIKE_COEFFS
        self.zernike = self.generate_zernike_phasemask(zernike_coeff_dict, wrap=True)

        print("--- System Configuration ---")
        print(f"SLM Plane Width: {self.Lx_slm / 1000:.2f} mm")
        print(f"SLM Plane Height: {self.Ly_slm / 1000:.2f} mm")
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
        return cp.exp(-R2 / (2 * (self.w0_um / 2.355) ** 2))

    def generate_weighted_array(
        self, weights, spacing, init_phase_randomness=1.0, angle_deg=0
    ):
        """
        Generates a weighted mask indicating where the tweezers should be and
        how strong they should be.
        Returns both the full 2D array and a list of the specific (y, x) indices.
        """
        dim = weights.shape
        xspan = (dim[1] - 1) * spacing
        yspan = (dim[0] - 1) * spacing
        xpos = np.linspace(-(xspan) / 2, (xspan) / 2, dim[1])
        ypos = np.linspace(-(yspan) / 2, (yspan) / 2, dim[0])
        Xpos, Ypos = np.meshgrid(xpos, ypos)
        Xn, Yn = Xpos.flatten(), Ypos.flatten()
        Xn, Yn = rotate_coordinates(Xn, Yn, angle_deg)
        Wn = weights.flatten()
        Thetan = np.random.rand(len(Wn)) * init_phase_randomness * 2 * np.pi

        print("--- Target Generation ---")
        print(f"Grid: {dim[0]}x{dim[1]}")
        print(f"Spacing: {spacing} um")

        return [Wn, Thetan, Xn, Yn]

    def generate_kagome_lattice(self, target_sites, spacing):
        """
        Generates a circularly symmetric Kagome lattice of tweezer sites.

        Parameters:
        - target_sites (int): The desired number of trap sites.
        - spacing (float): The nearest-neighbor site spacing in microns.

        Returns:
        - x_pos (np.ndarray): Array of x coordinates for the trap sites.
        - y_pos (np.ndarray): Array of y coordinates for the trap sites.
        """
        k_max = int(1.5 * np.sqrt(target_sites)) + 2
        m_vals = np.arange(-k_max, k_max + 1)
        n_vals = np.arange(-k_max, k_max + 1)
        m, n = np.meshgrid(m_vals, n_vals)
        m = m.flatten()
        n = n.flatten()
        a1 = np.array([2 * spacing, 0])
        a2 = np.array([spacing, np.sqrt(3) * spacing])
        b1 = np.array([spacing, 0])
        b2 = np.array([spacing / 2, np.sqrt(3) * spacing / 2])
        b3 = np.array([-spacing / 2, np.sqrt(3) * spacing / 2])
        centers = np.outer(m, a1) + np.outer(n, a2)
        sites_b1 = centers + b1
        sites_b2 = centers + b2
        sites_b3 = centers + b3
        all_sites = np.vstack((sites_b1, sites_b2, sites_b3))
        distances = np.linalg.norm(all_sites, axis=1)
        rounded_distances = np.round(distances, decimals=5)
        unique_radii, counts = np.unique(rounded_distances, return_counts=True)
        cumulative_sites = np.cumsum(counts)
        idx_closest = np.argmin(np.abs(cumulative_sites - target_sites))
        optimal_radius = unique_radii[idx_closest]
        actual_sites = cumulative_sites[idx_closest]
        mask = rounded_distances <= optimal_radius
        final_sites = all_sites[mask]
        x_pos = final_sites[:, 0]
        y_pos = final_sites[:, 1]

        print(
            f"Target sites: {target_sites} | Closest symmetric match: {actual_sites} sites"
        )

        return x_pos, y_pos

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

        return (fresnel_phase % (2 * cp.pi) - cp.pi).astype(cp.float32)

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
        return (blazed_phase % (2 * cp.pi) - cp.pi).astype(cp.float32)

    def superposition_optimization(
        self, target, max_iter=30, damping=0.4, verbose=True
    ):
        """
        Optimizes a phasemask for a discrete array of optical tweezers using
        the weighted superposition method, accelerated by the GPU using CuPy.
        """

        print("--- Starting GPU Superposition Phase Retrieval ---")
        start_time = time.time()

        # 3. Setup Illumination Beam (Gaussian)
        am_slm = cp.asarray(self.generate_source_amplitude())

        # 4. Define Target Trap Coordinates
        w_n, theta_n, x_n, y_n = target

        # Move variables to GPU
        w_n_cp = cp.asarray(w_n)
        theta_n_cp = cp.asarray(theta_n)
        x_n_cp = cp.asarray(x_n)
        y_n_cp = cp.asarray(y_n)
        target_0_cp = cp.asarray(target[0])

        X_phase = cp.exp(
            self.j_k * x_n_cp[:, None] * self.x_slm[None, :]
        )  # Shape: (N_traps, Nx)
        Y_phase = cp.exp(
            self.j_k * y_n_cp[:, None] * self.y_slm[None, :]
        )  # Shape: (N_traps, Ny)
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
            Y_term = Y_phase * C_n[:, None]  # Shape: (N_traps, Ny)
            U_tot = (
                Y_term.T @ X_phase
            )  # Shape: (Ny, N_traps) @ (N_traps, Nx) -> (Ny, Nx)
            pm_slm = cp.angle(U_tot)

            # Step B: Evaluate Exact Intensity at Target Sites (Vectorized)
            U_slm = am_slm * cp.exp(1j * pm_slm)
            U_foc_n = (
                U_slm @ X_phase_conj.T
            )  # Shape: (Ny, Nx) @ (Nx, N_traps) -> (Ny, N_traps)
            U_foc = cp.sum(Y_phase_conj * U_foc_n.T, axis=1)  # Shape: (N_traps,)
            I_foc = cp.abs(U_foc) ** 2

            # Step C: Calculate Metrics
            I_foc_masked = I_foc[mask_cp]
            uniformity = 1.0 - (
                I_foc_masked.std() / I_foc_masked.mean()
            )  # 1.0 is perfect uniformity
            uniformity_history[iteration] = uniformity

            I_foc_sum = I_foc.sum()
            I_foc_norm = I_foc / I_foc_sum
            mse = cp.mean((I_foc_norm - target_norm) ** 2)
            mse_history[iteration] = mse

            minmax_ratio = I_foc_masked.min() / I_foc_masked.max()
            minmax_history[iteration] = minmax_ratio

            # Step D: Update Weights
            update_ratio = (target_norm / I_foc_norm) ** damping
            w_n_cp = w_n_cp * update_ratio

            # Update the target phases to match the simulated arriving field
            theta_n_cp = cp.angle(U_foc)

            if verbose and iteration % 10 == 0:
                print(
                    f"Iteration {iteration:03d} | Mean-Squared Error: {float(mse):.2e} | Uniformity: {float(uniformity) * 100:.2f}% | Min/Max ratio: {float(minmax_ratio):.3f}"
                )

        trap_weights = I_foc
        print(
            f"Iteration {iteration:03d} | Mean-Squared Error: {float(mse):.2e} | Uniformity: {float(uniformity) * 100:.2f}% | Min/Max ratio: {float(minmax_ratio):.3f}"
        )
        print(f"Optimization finished in {time.time() - start_time:.2f} seconds.")

        return (
            pm_slm,
            [w_n_cp, theta_n_cp, x_n_cp, y_n_cp],
            [
                cp.asnumpy(uniformity_history),
                cp.asnumpy(minmax_history),
                cp.asnumpy(mse_history),
                cp.asnumpy(trap_weights),
            ],
        )

    def generate_phasemask(self, trap_terms):
        """
        Generates the phase mask using pure CuPy high-speed matrix multiplication.
        """
        # Unpack once
        w_n, theta_n, x_n, y_n = trap_terms

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

    def refine_phasemask(self, trap_terms, target_norm, iters, damping=0.5):
        """generate_phasemask plus `iters` weighted-GS updates, trap phases held fixed.

        Only the weights move; theta stays at the interpolated value. Letting theta float
        makes the iteration non-contractive, so freezing it is what allows a handful of
        iterations to help rather than wander.

        target_norm is the desired normalised intensity per site (zero on inactive sites).
        Returns the phasemask and the refined weights, for warm-starting the next frame.
        """
        w_n, theta_n, x_n, y_n = trap_terms
        w = cp.asarray(w_n, dtype=cp.float32)
        theta = cp.asarray(theta_n, dtype=cp.float32)

        X_phase = cp.exp(
            self.j_k * cp.asarray(x_n, dtype=cp.float32)[:, None] * self.x_slm[None, :]
        )
        Y_phase = cp.exp(
            self.j_k * cp.asarray(y_n, dtype=cp.float32)[:, None] * self.y_slm[None, :]
        )
        X_conj, Y_conj = cp.conj(X_phase), cp.conj(Y_phase)
        am_slm = self.generate_source_amplitude().astype(cp.complex64)

        for _ in range(iters):
            pm_slm = cp.angle(
                (Y_phase * (w * cp.exp(cp.complex64(1j) * theta))[:, None]).T @ X_phase
            )
            U_slm = am_slm * cp.exp(cp.complex64(1j) * pm_slm)
            I_foc = cp.abs(cp.sum(Y_conj * (U_slm @ X_conj.T).T, axis=1)) ** 2
            I_norm = I_foc / I_foc.sum()
            # inactive sites carry target_norm == 0 and w == 0, so leave them alone
            ratio = cp.where(
                target_norm > 0,
                (target_norm / cp.maximum(I_norm, cp.float32(1e-20))) ** damping,
                cp.float32(1.0),
            )
            w = w * ratio

        U_tot = (Y_phase * (w * cp.exp(cp.complex64(1j) * theta))[:, None]).T @ X_phase
        return cp.angle(U_tot).astype(cp.float32), w

    def evaluate_traps(self, pm_slm, x_n, y_n):
        am_slm = cp.asarray(self.generate_source_amplitude())
        pm_slm = cp.asarray(pm_slm)
        x_n_cp = cp.asarray(x_n)
        y_n_cp = cp.asarray(y_n)

        X_phase = cp.exp(
            self.j_k * x_n_cp[:, None] * self.x_slm[None, :]
        )  # Shape: (N_traps, Nx)
        Y_phase = cp.exp(
            self.j_k * y_n_cp[:, None] * self.y_slm[None, :]
        )  # Shape: (N_traps, Ny)
        X_phase_conj = cp.conj(X_phase)
        Y_phase_conj = cp.conj(Y_phase)

        U_slm = am_slm * cp.exp(1j * pm_slm)
        U_foc_n = (
            U_slm @ X_phase_conj.T
        )  # Shape: (Ny, Nx) @ (Nx, N_traps) -> (Ny, N_traps)
        U_foc = cp.sum(Y_phase_conj * U_foc_n.T, axis=1)  # Shape: (N_traps,)
        w_foc = cp.abs(U_foc)
        phi_foc = cp.angle(U_foc)
        I_foc = w_foc**2

        return I_foc.get(), phi_foc.get()

    def simulate_focal_plane(
        self,
        pm_slm,
        Nx_pad=2048,
        Ny_pad=2048,
        show=False,
        zoom_pixels=100,
        cmap="viridis",
    ):
        am_slm = cp.asarray(self.generate_source_amplitude())
        field_slm = am_slm * cp.exp(1j * cp.asarray(pm_slm))
        field_slm_pad = cp.zeros((Ny_pad, Nx_pad), dtype=cp.complex128)
        field_slm_pad[
            (Ny_pad - self.Ny) // 2 : (Ny_pad + self.Ny) // 2,
            (Nx_pad - self.Nx) // 2 : (Nx_pad + self.Nx) // 2,
        ] = field_slm
        field_foc_pad = cp.asnumpy(
            cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(field_slm_pad)))
        )

        if show:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            I_foc = np.abs(field_foc_pad) ** 2
            norm_intensity = I_foc / I_foc.max()
            x_focal_um = np.linspace(
                -self.Nx * self.dx_f / 2, self.Nx * self.dx_f / 2, Nx_pad
            )
            y_focal_um = np.linspace(
                -self.Ny * self.dy_f / 2, self.Ny * self.dy_f / 2, Ny_pad
            )
            im1 = ax.imshow(
                norm_intensity,
                extent=[x_focal_um[0], x_focal_um[-1], y_focal_um[0], y_focal_um[-1]],
                cmap=cmap,
            )

            zoom_range_um = zoom_pixels * self.dx_f
            ax.set_xlim(-zoom_range_um, zoom_range_um)
            ax.set_ylim(-zoom_range_um, zoom_range_um)
            ax.grid(False)
            ax.set_title("Simulated Intensity (Focal Plane)\nZoomed Central Region")
            ax.set_xlabel("x [um]")
            ax.set_ylabel("y [um]")
            cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
            cbar1.set_label("Normalized Intensity")

        return field_foc_pad

    def visualise_algorithm_performance(self, pm_slm, metrics, zoom_pixels, pad=1):
        U_foc = self.simulate_focal_plane(
            pm_slm, Nx_pad=self.Nx * pad, Ny_pad=self.Ny * pad, show=False
        )
        I_foc = np.abs(U_foc) ** 2

        fig, ax = plt.subplots(1, 3, figsize=(21, 6))
        uniformity_history, minmax_history, mse_history, trap_weights = metrics

        ax[0].plot(uniformity_history, "b-o", label="Uniformity")
        ax[0].plot(minmax_history, "r-s", label="Min/Max Ratio")
        ax0_secondary = ax[0].twinx()
        ax0_secondary.plot(mse_history, "g-o", label="MSE")
        ax0_secondary.set_ylabel("MSE Value")
        ax0_secondary.tick_params(axis="y")
        ax0_secondary.legend(loc="lower right")
        ax[0].set_xlabel("Iteration")
        ax[0].set_ylabel("Metric Value")
        ax[0].legend(loc="upper right")
        ax[0].grid(True)

        ax[0].set_title("Optimization Convergence Metrics")

        norm_intensity = I_foc / I_foc.max()
        x_focal_um = np.linspace(
            -self.Nx * self.dx_f / 2, self.Nx * self.dx_f / 2, self.Nx * pad
        )
        y_focal_um = np.linspace(
            -self.Ny * self.dy_f / 2, self.Ny * self.dy_f / 2, self.Ny * pad
        )
        im1 = ax[1].imshow(
            norm_intensity,
            extent=[x_focal_um[0], x_focal_um[-1], y_focal_um[0], y_focal_um[-1]],
            cmap="viridis",
        )

        zoom_range_um = zoom_pixels * self.dx_f
        ax[1].set_xlim(-zoom_range_um, zoom_range_um)
        ax[1].set_ylim(-zoom_range_um, zoom_range_um)
        ax[1].grid(False)
        ax[1].set_title("Simulated Intensity (Focal Plane)\nZoomed Central Region")
        ax[1].set_xlabel("x [um]")
        ax[1].set_ylabel("y [um]")
        cbar1 = plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
        cbar1.set_label("Normalized Intensity")

        # Plot a heatmap of the trap weights with ax[2]
        im2 = ax[2].imshow(trap_weights / trap_weights.max(), cmap="viridis")
        ax[2].grid(False)
        ax[2].set_title("Simulated Trap Weights")
        ax[2].set_xlabel("Trap X Index")
        ax[2].set_ylabel("Trap Y Index")
        cbar2 = plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
        cbar2.set_label("Normalized Trap Weight")

        plt.tight_layout()
        plt.show()

    def superimpose(self, phasemasks):
        return cp.mod(sum(phasemasks), 2 * cp.pi)

    def transform_phase_8bit(self, phasemask):
        return ((phasemask + cp.pi) * self.uint8_scale).astype(cp.uint8)

    def generate_rearrangement_sequence(self, terms1, terms2, occ_mask, d0=0.5):
        """
        Calculates the optimal Hungarian rearrangement path and generates
        the full sequence of interpolated phasemasks efficiently on the GPU.
        """
        start = time.time()

        w1, phi1, x1, y1 = terms1
        w2, phi2, x2, y2 = terms2
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
        phase_diff = (phi2_gpu[final_idx] - phi1_gpu[moving_idx] + cp.pi) % (
            2 * cp.pi
        ) - cp.pi
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
            update_state_kernel_linear(
                dw, dphi, total_dx, total_dy, ds, curr_w, curr_phi, curr_x, curr_y
            )

            # Repack the terms and call the generator.
            terms_gpu = (curr_w, curr_phi, curr_x, curr_y)
            pm_slm = self.generate_phasemask(terms_gpu)

            # Only superimpose the moving traps with the pre-calculated static background
            composite_pm = self.superimpose([pm_slm, static_background])

            # Store calculated 2D mask directly into pre-allocated VRAM chunk
            gpu_sequence[n] = self.transform_phase_8bit(composite_pm)

        # Batch copy the entire sequence back to the host (CPU) once at the end
        gpu_sequence.get(out=phasemasks_sequence)

        print(f"Time Taken for {n_steps} frames: {(time.time() - start) * 1000:.4f} ms")
        return phasemasks_sequence

    def iter_rearrangement_sequence(
        self,
        terms1,
        terms2,
        occ_mask,
        d0=0.5,
        profile="minimum_jerk",
        to_host=True,
        refine_iters=0,
        refine_damping=0.5,
        init_refine_iters=0,
        init_refine_damping=0.5,
    ):
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

        ``refine_iters`` > 0 spends the GPU idle time inside each frame: instead of one
        superposition GEMM from the interpolated terms, it runs that many weighted-GS
        weight updates with the interpolated trap phases frozen, warm-started from the
        previous frame. The pipeline is board-write bound (~0.59 ms) against ~0.26 ms of
        compute, so 1-2 iterations land inside the existing cycle time.
        """
        if profile not in ("minimum_jerk", "linear"):
            raise ValueError(
                f"profile must be 'minimum_jerk' or 'linear', got {profile!r}"
            )
        w1, phi1, x1, y1 = terms1
        w2, phi2, x2, y2 = terms2
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

        # .copy() matters: cp.asarray is a no-op on an array already float32 on the GPU,
        # and update_state_kernel writes in place - without it the caller's terms1 is
        # walked to the final positions and a second rearrangement starts wrong.
        curr_w = cp.asarray(w1, dtype=cp.float32).copy()
        curr_phi = cp.asarray(phi1, dtype=cp.float32).copy()
        curr_x = cp.asarray(x1, dtype=cp.float32).copy()
        curr_y = cp.asarray(y1, dtype=cp.float32).copy()

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

        dw[off_mask] = 0.0
        curr_w[off_mask] = 0.0

        dw[moving_idx] = (w2_gpu[final_idx] - w1_gpu[moving_idx]) / n_steps
        total_dx[moving_idx] = vec[:, 0].astype(cp.float32)
        total_dy[moving_idx] = vec[:, 1].astype(cp.float32)

        phase_diff = (phi2_gpu[final_idx] - phi1_gpu[moving_idx] + cp.pi) % (
            2 * cp.pi
        ) - cp.pi
        dphi[moving_idx] = phase_diff / n_steps

        static_background = self.superimpose([self.fresnel, self.blaze, self.zernike])

        # uniform target intensity across the moving traps; parked traps stay at zero
        target_norm = cp.zeros_like(curr_w)
        target_norm[moving_idx] = cp.float32(1.0 / len(moving_idx))
        refine_w = curr_w.copy() if refine_iters else None

        for n in range(n_steps):
            ds = float(ds_profile_cpu[n])
            update_state_kernel_linear(
                dw, dphi, total_dx, total_dy, ds, curr_w, curr_phi, curr_x, curr_y
            )
            if refine_iters:
                terms_gpu = (refine_w, curr_phi, curr_x, curr_y)
                pm_slm, refine_w = self.refine_phasemask(
                    terms_gpu, target_norm, refine_iters, refine_damping
                )
            else:
                terms_gpu = (curr_w, curr_phi, curr_x, curr_y)
                pm_slm = self.generate_phasemask(terms_gpu)
            composite_pm = self.superimpose([pm_slm, static_background])
            frame = self.transform_phase_8bit(composite_pm)
            yield frame.get() if to_host else frame

    def plan_rearrangement(
        self, terms1, terms2, occ_mask, d0=0.5, profile="minimum_jerk"
    ):
        if profile not in ("minimum_jerk", "linear"):
            raise ValueError(
                f"profile must be 'minimum_jerk' or 'linear', got {profile!r}"
            )
        w1, phi1, x1, y1 = terms1
        w2, phi2, x2, y2 = terms2
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

        phase_diff = (phi2_gpu[final_idx] - phi1_gpu[moving_idx] + cp.pi) % (
            2 * cp.pi
        ) - cp.pi
        dphi[moving_idx] = phase_diff / n_steps

        dw[off_mask] = 0.0
        curr_w[off_mask] = 0.0

        static_background = self.superimpose([self.fresnel, self.blaze, self.zernike])

        # The trailing fancy-index writes and the background superposition are
        # async launches. Without this sync their GPU tail would land on frame 0
        # and be billed as frame time - exactly the misattribution this split
        # exists to remove. Costs a stall here, which is the point: it belongs
        # to setup.
        cp.cuda.Stream.null.synchronize()

        return {
            "n_steps": n_steps,
            "ds_profile_cpu": ds_profile_cpu,
            "dw": dw,
            "dphi": dphi,
            "total_dx": total_dx,
            "total_dy": total_dy,
            "curr_w": curr_w,
            "curr_phi": curr_phi,
            "curr_x": curr_x,
            "curr_y": curr_y,
            "static_background": static_background,
            "n_moving": int(moving_idx.size),
            "profile": profile,
            # The JV assignment itself, kept on the device. Reading it here would
            # cost a sync inside the timed setup span, so the caller copies it to
            # the host once the run is over.
            "occ_mask": occ_mask,
            "occ_indices": occ_indices,
            "moving_idx": moving_idx,
            "final_idx": final_idx,
            "pos_init": pos_init,
            "pos_final": pos_final,
            "n_targets": int(pos2.shape[0]),
        }

    def iter_rearrangement_sequence_timings(self, plan, to_host=False):
        dw, dphi = plan["dw"], plan["dphi"]
        total_dx, total_dy = plan["total_dx"], plan["total_dy"]
        curr_w, curr_phi = plan["curr_w"], plan["curr_phi"]
        curr_x, curr_y = plan["curr_x"], plan["curr_y"]
        static_background = plan["static_background"]
        ds_profile_cpu = plan["ds_profile_cpu"]

        for n in range(plan["n_steps"]):
            start_ev = cp.cuda.Event()
            end_ev = cp.cuda.Event()
            start_ev.record()
            ds = float(ds_profile_cpu[n])
            update_state_kernel_linear(
                dw, dphi, total_dx, total_dy, ds, curr_w, curr_phi, curr_x, curr_y
            )
            terms_gpu = (curr_w, curr_phi, curr_x, curr_y)
            pm_slm = self.generate_phasemask(terms_gpu)
            composite_pm = self.superimpose([pm_slm, static_background])
            frame = self.transform_phase_8bit(composite_pm)
            end_ev.record()
            yield (frame.get() if to_host else frame), start_ev, end_ev


# Define this fused kernel outside the loop (or at the module level).
# CuPy will compile it once on the first run and cache it, saving massive
# overhead by doing 4 state updates in a single GPU kernel launch.
update_state_kernel_linear = cp.ElementwiseKernel(
    "float32 dw, float32 dphi, float32 dx, float32 dy, float32 ds",
    "float32 w, float32 phi, float32 x, float32 y",
    """
    w += dw;
    phi += dphi;
    x += dx * ds;
    y += dy * ds;
    """,
    "update_hologram_state",
)


################################################################################################################################################################
################################################################################################################################################################

# =============================================================================
# Helpers
# =============================================================================
STEP_NORMS = ("max", "ellipse")


def step_cost(delta, d0, d0_z=None, step_norm="max"):
    """Dimensionless cost of a displacement delta (..., 3): 1 = one full allowed
    step. A frame is allowed if step_cost(frame-to-frame displacement) <= 1."""
    if step_norm not in STEP_NORMS:
        raise ValueError(f"step_norm must be one of {STEP_NORMS}, got {step_norm!r}")
    xp = cp.get_array_module(delta)
    d0_z = d0 if d0_z is None else d0_z
    c_xy = xp.linalg.norm(delta[..., :2], axis=-1) / d0
    c_z = xp.abs(delta[..., 2]) / d0_z
    if step_norm == "max":
        return xp.maximum(c_xy, c_z)
    return xp.sqrt(c_xy**2 + c_z**2)


def _quintic(t):
    """Minimum-jerk ramp 0 -> 1 with zero velocity and acceleration at both ends."""
    return t * t * t * (10 - 15 * t + 6 * t * t)


def _vertices_path(s, pts, d0=1.0, d0_z=None, step_norm="max", smooth_corners=False):
    """Piecewise-linear path through vertices pts, shape (N, V, 3), first vertex
    = start, last = end. Each segment gets a share of s proportional to its step
    cost (the number of full-size frames it needs), so every segment runs at
    the same fraction of its own limit. Zero-length segments are allowed.
    Only the ratio d0 / d0_z matters here; pass the generator's values.

    smooth_corners=True runs each segment with its own minimum-jerk ramp, so the
    trap comes to rest at every vertex instead of turning a corner at speed."""
    xp = cp.get_array_module(pts)
    d0 = 1.0 if d0 is None else d0
    seg = step_cost(xp.diff(pts, axis=1), d0, d0_z, step_norm)  # (N, V-1)
    cum = xp.concatenate((xp.zeros_like(seg[:, :1]), xp.cumsum(seg, axis=1)), axis=1)
    cum = cum / xp.where(cum[:, -1:] > 0, cum[:, -1:], 1)  # breakpoints in s
    s = xp.asarray(s, dtype=pts.dtype)
    # segment index for every (frame, trap): breakpoints passed, minus one
    i = (cum[None] <= s[:, None, None]).sum(axis=2) - 1
    i = xp.clip(i, 0, pts.shape[1] - 2)  # (K, N)
    n = xp.arange(pts.shape[0])[None, :]
    a, b = pts[n, i], pts[n, i + 1]  # (K, N, 3)
    ca, cb = cum[n, i], cum[n, i + 1]  # (K, N)
    t = xp.clip((s[:, None] - ca) / xp.where(cb > ca, cb - ca, 1), 0, 1)
    if smooth_corners:
        t = _quintic(t)
    return a + t[..., None] * (b - a)


def _moving(p0, p1, tol=1e-6):
    xp = cp.get_array_module(p0)
    return xp.linalg.norm(p1 - p0, axis=1) > tol


def _bindable(factory, kwargs, path_fn):
    """If the caller left d0 unset, let the generator supply its step limits
    later (see _bind_limits), so the path and the generator always agree."""
    if kwargs.get("d0") is None:
        path_fn.bind_limits = lambda d0, d0_z, step_norm: factory(
            **{**kwargs, "d0": d0, "d0_z": d0_z, "step_norm": step_norm}
        )
    return path_fn


def _bind_limits(path_fn, d0, d0_z, step_norm):
    """Give a path function the generator's step limits, unless it already has
    its own. Works for the factories below (lift_path, polyline_path) and for
    functools.partial of functions marked accepts_limits (manhattan_path)."""
    if getattr(path_fn, "bind_limits", None) is not None:
        return path_fn.bind_limits(d0, d0_z, step_norm)
    if isinstance(path_fn, partial):
        f, given = path_fn.func, path_fn.keywords
    else:
        f, given = path_fn, {}
    if getattr(f, "accepts_limits", False) and given.get("d0") is None:
        return partial(path_fn, d0=d0, d0_z=d0_z, step_norm=step_norm)
    return path_fn


# =============================================================================
# Path functions
# =============================================================================
def linear_path(s, p0, p1):
    """Straight line in 3D. Reproduces the original behaviour."""
    s = s[:, None, None]
    return p0 + s * (p1 - p0)


def lift_path(
    d, d0=None, d0_z=None, step_norm="max", smooth_corners=False, only_moving=True
):
    """Lift by +d along z, travel in a straight line, lower by -d.

    Vertices for each trap: p0 -> p0 + d z^ -> p1 + d z^ -> p1.

    d              : scalar or (N,) array, same units as positions. Sign sets the
                     direction (+d away from / towards the objective, depending on
                     your defocus sign convention). A trap with d = 0 goes straight.
    d0, d0_z,      : leave unset (recommended) and the generator fills in its own
    step_norm        values, so the legs share s in proportion to the frames they
                     need: vertical legs run up to d0_z per frame, the lateral leg
                     up to d0. Set them only to call the path on its own.
    smooth_corners : stop at each corner (see _vertices_path). Pair it with
                     profile="linear"; with "minimum_jerk" it is smoothed twice.
    only_moving    : traps with p0 == p1 are not lifted. Leave this on unless you
                     really want stationary traps to bob up and down.
    """

    def path_fn(s, p0, p1):
        xp = cp.get_array_module(p0)
        dz = xp.zeros_like(p0)
        dz[:, 2] = xp.asarray(d, dtype=p0.dtype)
        if only_moving:
            dz[~_moving(p0, p1)] = 0
        pts = xp.stack((p0, p0 + dz, p1 + dz, p1), axis=1)  # (N, 4, 3)
        return _vertices_path(s, pts, d0, d0_z, step_norm, smooth_corners)

    return _bindable(
        lift_path,
        {
            "d": d,
            "d0": d0,
            "d0_z": d0_z,
            "step_norm": step_norm,
            "smooth_corners": smooth_corners,
            "only_moving": only_moving,
        },
        path_fn,
    )


def polyline_path(waypoints, d0=None, d0_z=None, step_norm="max", smooth_corners=False):
    """Piecewise-linear path through user-supplied 3D waypoints.

    waypoints : (N, M, 3) intermediate points for each trap, NOT including start
                and end. Every trap needs the same M; a trap that should go
                straight (or stay put) gets its start position repeated."""

    def path_fn(s, p0, p1):
        xp = cp.get_array_module(p0)
        pts = xp.concatenate(
            (p0[:, None], xp.asarray(waypoints, dtype=p0.dtype), p1[:, None]), axis=1
        )
        return _vertices_path(s, pts, d0, d0_z, step_norm, smooth_corners)

    return _bindable(
        polyline_path,
        {
            "waypoints": waypoints,
            "d0": d0,
            "d0_z": d0_z,
            "step_norm": step_norm,
            "smooth_corners": smooth_corners,
        },
        path_fn,
    )


def manhattan_path(
    s, p0, p1, order="xyz", d0=None, d0_z=None, step_norm="max", smooth_corners=False
):
    """Axis-by-axis move in the given order, e.g. "xy", "zxy", "xyz".
    Any axis left out of `order` is moved on a final leg straight to p1."""
    xp = cp.get_array_module(p0)
    axis = {"x": 0, "y": 1, "z": 2}
    pts, cur = [p0], p0
    for a in order:
        cur = cur.copy()
        cur[:, axis[a]] = p1[:, axis[a]]
        pts.append(cur)
    pts.append(p1)
    return _vertices_path(s, xp.stack(pts, axis=1), d0, d0_z, step_norm, smooth_corners)


manhattan_path.accepts_limits = True  # the generator fills in d0, d0_z, step_norm


def arc_path(s, p0, p1, bulge=0.25, out_of_plane=False):
    """Quadratic Bezier with its control point displaced by bulge x chord length.

    out_of_plane=False : displacement along z^ x chord (horizontal). For a move in
                         the focal plane this is exactly the 2D arc_path.
    out_of_plane=True  : displacement along the part of z^ perpendicular to the
                         chord, i.e. a smooth hop over whatever is in the way.
                         The apex height is bulge * |chord| / 2.
    bulge may be a scalar or an (N, 1) array."""
    xp = cp.get_array_module(p0)
    d = p1 - p0
    L = xp.linalg.norm(d, axis=1, keepdims=True)
    zhat = xp.zeros_like(d)
    zhat[:, 2] = 1
    if out_of_plane:
        dhat = d / xp.maximum(L, 1e-12)
        u = zhat - dhat * (dhat[:, 2:3])  # z^ minus its chord component
    else:
        u = xp.cross(zhat, d)  # (-dy, dx, 0)
    u = u / xp.maximum(xp.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    ctrl = 0.5 * (p0 + p1) + bulge * L * u
    s = s[:, None, None]
    return (1 - s) ** 2 * p0 + 2 * (1 - s) * s * ctrl + s**2 * p1


def min_separation(traj, z_scale=1.0, chunk=64):
    """Smallest distance between any two traps over all frames, measured as
    sqrt(dx^2 + dy^2 + (z_scale * dz)^2). z_scale = 1 is plain Euclidean
    distance; z_scale < 1 discounts axial separation because the trap is longer
    along z. Geometric only: it does not see defocused light from one trap
    landing on another (see the notes)."""
    xp = cp.get_array_module(traj)
    scale = xp.asarray([1.0, 1.0, z_scale], dtype=traj.dtype)
    n = traj.shape[1]
    best = float("inf")
    for k in range(0, traj.shape[0], chunk):
        t = traj[k : k + chunk] * scale
        d = xp.linalg.norm(t[:, :, None, :] - t[:, None, :, :], axis=-1)
        d[:, xp.arange(n), xp.arange(n)] = xp.inf
        best = min(best, float(d.min()))
    return best


def build_trajectory(
    path_fn,
    pos_init,
    pos_final,
    d0=0.5,
    d0_z=None,
    profile="minimum_jerk",
    step_norm="max",
):
    """Frame count and trajectory exactly as iter_transport_sequence_3D uses them.
    Handy for checking a path (min_separation, plots) before touching the SLM.
    Returns traj, shape (n_steps + 1, N, 3), frame 0 = pos_init."""
    if profile not in ("minimum_jerk", "linear"):
        raise ValueError(f"profile must be 'minimum_jerk' or 'linear', got {profile!r}")
    xp = cp.get_array_module(pos_init)
    d0_z = d0 if d0_z is None else d0_z
    path_fn = _bind_limits(path_fn, d0, d0_z, step_norm)

    def cost(delta):
        return step_cost(delta, d0, d0_z, step_norm)

    # the path must start and end where terms1 and terms2 put the traps
    ends = path_fn(xp.asarray([0.0, 1.0], dtype=pos_init.dtype), pos_init, pos_final)
    end_err = float(xp.abs(ends - xp.stack((pos_init, pos_final))).max())
    if end_err > 1e-3 * min(d0, d0_z):
        raise ValueError(f"path_fn(0), path_fn(1) miss the endpoints by {end_err:.3g}")

    # first estimate: total step cost of the most expensive path, i.e. the number
    # of full-size frames it needs at constant speed
    s_fine = xp.linspace(0, 1, 512, dtype=pos_init.dtype)
    P_fine = path_fn(s_fine, pos_init, pos_final)
    path_cost = cost(xp.diff(P_fine, axis=0)).sum(axis=0)
    steps_scale = 1.0 if profile == "linear" else 1.875
    n_steps = max(1, int(xp.ceil(steps_scale * float(path_cost.max()))))

    # refine until every trap's step cost is <= 1 in every frame
    n_cap = 50 * n_steps
    while True:
        tau = xp.linspace(0, 1, n_steps + 1, dtype=pos_init.dtype)
        s_profile = (
            tau if profile == "linear" else 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        )
        traj = path_fn(s_profile, pos_init, pos_final).astype(xp.float32)
        max_cost = float(cost(xp.diff(traj, axis=0)).max())
        if max_cost <= 1 + 1e-4:
            return traj
        n_steps = int(xp.ceil(n_steps * max_cost))
        if n_steps > n_cap:
            raise RuntimeError(
                f"step still {max_cost:.3g}x the limit after refining to "
                f"{n_steps} frames: is path_fn continuous in s?"
            )


class OptimisationBasedPhasemaskGeneratorGPU3D:
    def __init__(
        self,
        wavelength_um=0.852,
        focal_length_mm=17.3,
        slm_pitch_um=17,
        slm_res=(1024, 1024),
        input_beam_waist_mm=16,
        fresnel_f_mm=1072,
        blaze_dx_dy_um=(40.0, -8.0),
        zernike_coeff_dict=None,
    ):

        self.lam = wavelength_um
        self.f_mm = focal_length_mm
        self.f_um = focal_length_mm * 1000
        self.dx_slm = slm_pitch_um
        self.Ny, self.Nx = slm_res
        self.w0_mm = input_beam_waist_mm
        self.w0_um = input_beam_waist_mm * 1000
        self.Lx_slm = self.Nx * self.dx_slm  # In units of um
        self.Ly_slm = self.Ny * self.dx_slm  # In units of um
        self.dx_f = (self.lam * self.f_um) / self.Lx_slm  # In units of um
        self.dy_f = (self.lam * self.f_um) / self.Ly_slm  # In units of um
        self.Lx_foc = self.Nx * self.dx_f
        self.Ly_foc = self.Ny * self.dy_f

        self.x_slm = cp.linspace(
            -self.Lx_slm / 2, self.Lx_slm / 2, self.Nx, dtype=cp.float32
        )
        self.y_slm = cp.linspace(
            -self.Ly_slm / 2, self.Ly_slm / 2, self.Ny, dtype=cp.float32
        )

        # Precalculations
        self.k = np.float32(2 * np.pi / (self.lam * self.f_um))
        self.j_k = cp.complex64(1j) * self.k
        self.z_scale = cp.complex64(1j) * cp.pi / (self.lam * self.f_um**2)
        self.uint8_scale = cp.float32(255.0) / (2 * cp.pi)

        # Additional phasemasks - phase wrapped and in radians
        self.fresnel = self.generate_fresnel_lens_phasemask(
            focal_length_mm=fresnel_f_mm
        )
        self.blaze = self.generate_blazed_grating_phasemask(
            dx_um=blaze_dx_dy_um[0], dy_um=blaze_dx_dy_um[1]
        )
        if zernike_coeff_dict is None:
            zernike_coeff_dict = DEFAULT_ZERNIKE_COEFFS
        self.zernike = self.generate_zernike_phasemask(zernike_coeff_dict, wrap=True)

        print("--- System Configuration ---")
        print(f"SLM Plane Width: {self.Lx_slm / 1000:.2f} mm")
        print(f"SLM Plane Height: {self.Ly_slm / 1000:.2f} mm")
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
        return cp.exp(-R2 / (2 * (self.w0_um / 2.355) ** 2))

    def generate_weighted_array(
        self, weights, spacing, init_phase_randomness=1.0, angle_deg=0
    ):
        """
        Generates a weighted mask indicating where the tweezers should be and
        how strong they should be.
        Returns both the full 2D array and a list of the specific (y, x) indices.
        """
        dim = weights.shape
        xspan = (dim[1] - 1) * spacing
        yspan = (dim[0] - 1) * spacing
        xpos = np.linspace(-(xspan) / 2, (xspan) / 2, dim[1])
        ypos = np.linspace(-(yspan) / 2, (yspan) / 2, dim[0])
        Xpos, Ypos = np.meshgrid(xpos, ypos)
        Xn, Yn = Xpos.flatten(), Ypos.flatten()
        Xn, Yn = rotate_coordinates(Xn, Yn, angle_deg)
        Zn = np.zeros_like(
            Xn
        )  # For 3D extension, currently all traps are in the same focal plane (z=0)
        Wn = weights.flatten()
        Thetan = np.random.rand(len(Wn)) * init_phase_randomness * 2 * np.pi

        print("--- Target Generation ---")
        print(f"Grid: {dim[0]}x{dim[1]}")
        print(f"Spacing: {spacing} um")

        return [Wn, Thetan, Xn, Yn, Zn]

    def generate_kagome_lattice(self, target_sites, spacing):
        """
        Generates a circularly symmetric Kagome lattice of tweezer sites.

        Parameters:
        - target_sites (int): The desired number of trap sites.
        - spacing (float): The nearest-neighbor site spacing in microns.

        Returns:
        - x_pos (np.ndarray): Array of x coordinates for the trap sites.
        - y_pos (np.ndarray): Array of y coordinates for the trap sites.
        - z_pos (np.ndarray): Array of z coordinates for the trap sites (currently all zeros).
        """
        k_max = int(1.5 * np.sqrt(target_sites)) + 2
        m_vals = np.arange(-k_max, k_max + 1)
        n_vals = np.arange(-k_max, k_max + 1)
        m, n = np.meshgrid(m_vals, n_vals)
        m = m.flatten()
        n = n.flatten()
        a1 = np.array([2 * spacing, 0])
        a2 = np.array([spacing, np.sqrt(3) * spacing])
        b1 = np.array([spacing, 0])
        b2 = np.array([spacing / 2, np.sqrt(3) * spacing / 2])
        b3 = np.array([-spacing / 2, np.sqrt(3) * spacing / 2])
        centers = np.outer(m, a1) + np.outer(n, a2)
        sites_b1 = centers + b1
        sites_b2 = centers + b2
        sites_b3 = centers + b3
        all_sites = np.vstack((sites_b1, sites_b2, sites_b3))
        distances = np.linalg.norm(all_sites, axis=1)
        rounded_distances = np.round(distances, decimals=5)
        unique_radii, counts = np.unique(rounded_distances, return_counts=True)
        cumulative_sites = np.cumsum(counts)
        idx_closest = np.argmin(np.abs(cumulative_sites - target_sites))
        optimal_radius = unique_radii[idx_closest]
        actual_sites = cumulative_sites[idx_closest]
        mask = rounded_distances <= optimal_radius
        final_sites = all_sites[mask]
        x_pos = final_sites[:, 0]
        y_pos = final_sites[:, 1]
        z_pos = np.zeros_like(
            x_pos
        )  # For 3D extension, currently all traps are in the same focal plane (z=0)

        print(
            f"Target sites: {target_sites} | Closest symmetric match: {actual_sites} sites"
        )

        return x_pos, y_pos, z_pos

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

        return fresnel_phase % (2 * cp.pi) - cp.pi

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
        return blazed_phase % (2 * cp.pi) - cp.pi

    def superposition_optimization(
        self, target, max_iter=30, damping=0.4, verbose=True
    ):
        """
        Optimizes a phasemask for a discrete array of 3D optical tweezers using
        the weighted superposition method, accelerated by the GPU using CuPy.
        """

        print("--- Starting GPU Superposition Phase Retrieval (3D) ---")
        start_time = time.time()

        # Setup Illumination Beam (Gaussian)
        am_slm = cp.asarray(self.generate_source_amplitude())

        # Define Target Trap Coordinates (Now utilizing z_n)
        w_n, theta_n, x_n, y_n, z_n = target

        # Move variables to GPU
        w_n_cp = cp.asarray(w_n)
        theta_n_cp = cp.asarray(theta_n)
        x_n_cp = cp.asarray(x_n)
        y_n_cp = cp.asarray(y_n)
        z_n_cp = cp.asarray(z_n)
        target_0_cp = cp.asarray(target[0])

        X_phase = cp.exp(
            self.j_k * x_n_cp[:, None] * self.x_slm[None, :]
            + self.z_scale * z_n_cp[:, None] * (self.x_slm**2)[None, :]
        )

        Y_phase = cp.exp(
            self.j_k * y_n_cp[:, None] * self.y_slm[None, :]
            + self.z_scale * z_n_cp[:, None] * (self.y_slm**2)[None, :]
        )

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
            I_foc = cp.abs(U_foc) ** 2

            # Step C: Calculate Metrics
            I_foc_masked = I_foc[mask_cp]
            uniformity = 1.0 - (I_foc_masked.std() / I_foc_masked.mean())
            uniformity_history[iteration] = uniformity

            I_foc_sum = I_foc.sum()
            I_foc_norm = I_foc / I_foc_sum
            mse = cp.mean((I_foc_norm - target_norm) ** 2)
            mse_history[iteration] = mse

            minmax_ratio = I_foc_masked.min() / I_foc_masked.max()
            minmax_history[iteration] = minmax_ratio

            # Step D: Update Weights
            update_ratio = (target_norm / I_foc_norm) ** damping
            w_n_cp = w_n_cp * update_ratio

            # Update the target phases to match the simulated arriving field
            theta_n_cp = cp.angle(U_foc)

            if verbose and iteration % 10 == 0:
                print(
                    f"Iteration {iteration:03d} | Mean-Squared Error: {float(mse):.2e} | Uniformity: {float(uniformity) * 100:.2f}% | Min/Max ratio: {float(minmax_ratio):.3f}"
                )

        # Ensure array_shape matches 3D output if necessary, or keep as 1D list of traps
        trap_weights = I_foc
        print(
            f"Iteration {iteration:03d} | Mean-Squared Error: {float(mse):.2e} | Uniformity: {float(uniformity) * 100:.2f}% | Min/Max ratio: {float(minmax_ratio):.3f}"
        )
        print(f"Optimization finished in {time.time() - start_time:.2f} seconds.")

        return (
            pm_slm,
            [w_n_cp, theta_n_cp, x_n_cp, y_n_cp, z_n_cp],
            [
                cp.asnumpy(uniformity_history),
                cp.asnumpy(minmax_history),
                cp.asnumpy(mse_history),
                cp.asnumpy(trap_weights),
            ],
        )

    def generate_phasemask(self, trap_terms):
        """
        Generates the phase mask using pure CuPy high-speed matrix multiplication.
        """
        # Unpack once
        w_n, theta_n, x_n, y_n, z_n = trap_terms

        # 3. TRANSFER AND CAST TO 32-BIT
        w_n_cp = cp.asarray(w_n, dtype=cp.float32)
        theta_n_cp = cp.asarray(theta_n, dtype=cp.float32)
        x_n_cp = cp.asarray(x_n, dtype=cp.float32)
        y_n_cp = cp.asarray(y_n, dtype=cp.float32)
        z_n_cp = cp.asarray(z_n, dtype=cp.float32)

        # 4. BUILD STEERING MATRICES (Using complex64 arithmetic)
        X_phase = cp.exp(
            self.j_k * x_n_cp[:, None] * self.x_slm[None, :]
            + self.z_scale * z_n_cp[:, None] * (self.x_slm**2)[None, :]
        )

        Y_phase = cp.exp(
            self.j_k * y_n_cp[:, None] * self.y_slm[None, :]
            + self.z_scale * z_n_cp[:, None] * (self.y_slm**2)[None, :]
        )

        # 5. SUPERPOSITION COMPLEX FIELD
        C_n = w_n_cp * cp.exp(cp.complex64(1j) * theta_n_cp)
        Y_term = Y_phase * C_n[:, None]
        U_tot = Y_term.T @ X_phase

        return cp.angle(U_tot)

    def refine_phasemask(self, trap_terms, target_norm, iters, damping=0.5):
        """generate_phasemask plus `iters` weighted-GS updates, trap phases held fixed.

        Only the weights move; theta stays at the interpolated value. Letting theta float
        makes the iteration non-contractive, so freezing it is what allows a handful of
        iterations to help rather than wander.

        target_norm is the desired normalised intensity per site (zero on inactive sites).
        Returns the phasemask and the refined weights, for warm-starting the next frame.
        """
        w_n, theta_n, x_n, y_n, z_n = trap_terms

        # 3. TRANSFER AND CAST TO 32-BIT
        w = cp.asarray(w_n, dtype=cp.float32)
        theta = cp.asarray(theta_n, dtype=cp.float32)
        x_n_cp = cp.asarray(x_n, dtype=cp.float32)
        y_n_cp = cp.asarray(y_n, dtype=cp.float32)
        z_n_cp = cp.asarray(z_n, dtype=cp.float32)

        # 4. BUILD STEERING MATRICES (Using complex64 arithmetic)
        X_phase = cp.exp(
            self.j_k * x_n_cp[:, None] * self.x_slm[None, :]
            + self.z_scale * z_n_cp[:, None] * (self.x_slm**2)[None, :]
        )

        Y_phase = cp.exp(
            self.j_k * y_n_cp[:, None] * self.y_slm[None, :]
            + self.z_scale * z_n_cp[:, None] * (self.y_slm**2)[None, :]
        )

        X_conj, Y_conj = cp.conj(X_phase), cp.conj(Y_phase)
        am_slm = self.generate_source_amplitude().astype(cp.complex64)

        for _ in range(iters):
            pm_slm = cp.angle(
                (Y_phase * (w * cp.exp(cp.complex64(1j) * theta))[:, None]).T @ X_phase
            )
            U_slm = am_slm * cp.exp(cp.complex64(1j) * pm_slm)
            I_foc = cp.abs(cp.sum(Y_conj * (U_slm @ X_conj.T).T, axis=1)) ** 2
            I_norm = I_foc / I_foc.sum()
            # inactive sites carry target_norm == 0 and w == 0, so leave them alone
            ratio = cp.where(
                target_norm > 0,
                (target_norm / cp.maximum(I_norm, cp.float32(1e-20))) ** damping,
                cp.float32(1.0),
            )
            w = w * ratio

        U_tot = (Y_phase * (w * cp.exp(cp.complex64(1j) * theta))[:, None]).T @ X_phase
        return cp.angle(U_tot).astype(cp.float32), w

    def evaluate_traps(self, pm_slm, x_n, y_n, z_n):
        am_slm = cp.asarray(self.generate_source_amplitude())
        pm_slm = cp.asarray(pm_slm)
        x_n_cp = cp.asarray(x_n)
        y_n_cp = cp.asarray(y_n)
        z_n_cp = cp.asarray(z_n)

        X_phase = cp.exp(
            self.j_k * x_n_cp[:, None] * self.x_slm[None, :]
            + self.z_scale * z_n_cp[:, None] * (self.x_slm**2)[None, :]
        )

        Y_phase = cp.exp(
            self.j_k * y_n_cp[:, None] * self.y_slm[None, :]
            + self.z_scale * z_n_cp[:, None] * (self.y_slm**2)[None, :]
        )

        X_phase_conj = cp.conj(X_phase)
        Y_phase_conj = cp.conj(Y_phase)

        U_slm = am_slm * cp.exp(1j * pm_slm)
        U_foc_n = (
            U_slm @ X_phase_conj.T
        )  # Shape: (Ny, Nx) @ (Nx, N_traps) -> (Ny, N_traps)
        U_foc = cp.sum(Y_phase_conj * U_foc_n.T, axis=1)  # Shape: (N_traps,)
        w_foc = cp.abs(U_foc)
        phi_foc = cp.angle(U_foc)
        I_foc = w_foc**2

        return I_foc.get(), phi_foc.get()

    def simulate_focal_plane(
        self,
        pm_slm,
        z_plane=0.0,
        Nx_pad=2048,
        Ny_pad=2048,
        show=False,
        zoom_pixels=100,
        cmap="viridis",
    ):
        X_slm, Y_slm = cp.meshgrid(self.x_slm, self.y_slm)
        defocus = -self.z_scale * z_plane * (X_slm**2 + Y_slm**2)
        am_slm = cp.asarray(self.generate_source_amplitude())
        field_slm = am_slm * cp.exp(1j * cp.asarray(pm_slm) + defocus)
        field_slm_pad = cp.zeros((Ny_pad, Nx_pad), dtype=cp.complex128)
        field_slm_pad[
            (Ny_pad - self.Ny) // 2 : (Ny_pad + self.Ny) // 2,
            (Nx_pad - self.Nx) // 2 : (Nx_pad + self.Nx) // 2,
        ] = field_slm
        field_foc_pad = cp.asnumpy(
            cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(field_slm_pad)))
        )

        if show:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            I_foc = np.abs(field_foc_pad) ** 2
            norm_intensity = I_foc / I_foc.max()
            x_focal_um = np.linspace(
                -self.Nx * self.dx_f / 2, self.Nx * self.dx_f / 2, Nx_pad
            )
            y_focal_um = np.linspace(
                -self.Ny * self.dy_f / 2, self.Ny * self.dy_f / 2, Ny_pad
            )
            im1 = ax.imshow(
                norm_intensity,
                extent=[x_focal_um[0], x_focal_um[-1], y_focal_um[0], y_focal_um[-1]],
                cmap=cmap,
            )

            zoom_range_um = zoom_pixels * self.dx_f
            ax.set_xlim(-zoom_range_um, zoom_range_um)
            ax.set_ylim(-zoom_range_um, zoom_range_um)
            ax.grid(False)
            ax.set_title("Simulated Intensity (Focal Plane)\nZoomed Central Region")
            ax.set_xlabel("x [um]")
            ax.set_ylabel("y [um]")
            cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
            cbar1.set_label("Normalized Intensity")

        return field_foc_pad

    def visualise_algorithm_performance(self, pm_slm, metrics, zoom_pixels, pad=1):
        U_foc = self.simulate_focal_plane(
            pm_slm, Nx_pad=self.Nx * pad, Ny_pad=self.Ny * pad, show=False
        )
        I_foc = np.abs(U_foc) ** 2

        fig, ax = plt.subplots(1, 3, figsize=(21, 6))
        uniformity_history, minmax_history, mse_history, trap_weights = metrics

        ax[0].plot(uniformity_history, "b-o", label="Uniformity")
        ax[0].plot(minmax_history, "r-s", label="Min/Max Ratio")
        ax0_secondary = ax[0].twinx()
        ax0_secondary.plot(mse_history, "g-o", label="MSE")
        ax0_secondary.set_ylabel("MSE Value")
        ax0_secondary.tick_params(axis="y")
        ax0_secondary.legend(loc="lower right")
        ax[0].set_xlabel("Iteration")
        ax[0].set_ylabel("Metric Value")
        ax[0].legend(loc="upper right")
        ax[0].grid(True)

        ax[0].set_title("Optimization Convergence Metrics")

        norm_intensity = I_foc / I_foc.max()
        x_focal_um = np.linspace(
            -self.Nx * self.dx_f / 2, self.Nx * self.dx_f / 2, self.Nx * pad
        )
        y_focal_um = np.linspace(
            -self.Ny * self.dy_f / 2, self.Ny * self.dy_f / 2, self.Ny * pad
        )
        im1 = ax[1].imshow(
            norm_intensity,
            extent=[x_focal_um[0], x_focal_um[-1], y_focal_um[0], y_focal_um[-1]],
            cmap="viridis",
        )

        zoom_range_um = zoom_pixels * self.dx_f
        ax[1].set_xlim(-zoom_range_um, zoom_range_um)
        ax[1].set_ylim(-zoom_range_um, zoom_range_um)
        ax[1].grid(False)
        ax[1].set_title("Simulated Intensity (Focal Plane)\nZoomed Central Region")
        ax[1].set_xlabel("x [um]")
        ax[1].set_ylabel("y [um]")
        cbar1 = plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
        cbar1.set_label("Normalized Intensity")

        # Plot a heatmap of the trap weights with ax[2]
        im2 = ax[2].imshow(trap_weights / trap_weights.max(), cmap="viridis")
        ax[2].grid(False)
        ax[2].set_title("Simulated Trap Weights")
        ax[2].set_xlabel("Trap X Index")
        ax[2].set_ylabel("Trap Y Index")
        cbar2 = plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
        cbar2.set_label("Normalized Trap Weight")

        plt.tight_layout()
        plt.show()

    def superimpose(self, phasemasks):
        return cp.mod(sum(phasemasks), 2 * cp.pi)

    def transform_phase_8bit(self, phasemask):
        return ((phasemask + cp.pi) * self.uint8_scale).astype(cp.uint8)

    def generate_rearrangement_sequence(self, terms1, terms2, occ_mask, d0=0.5):
        """
        Calculates the optimal Hungarian rearrangement path and generates
        the full sequence of interpolated phasemasks efficiently on the GPU.
        """
        start = time.time()

        w1, phi1, x1, y1, z1 = terms1
        w2, phi2, x2, y2, z2 = terms2
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

        # Load steps for MOVING traps
        dw[moving_idx] = (w2_gpu[final_idx] - w1_gpu[moving_idx]) / n_steps
        total_dx[moving_idx] = vec[:, 0].astype(cp.float32)
        total_dy[moving_idx] = vec[:, 1].astype(cp.float32)
        total_dz[moving_idx] = vec[:, 2].astype(cp.float32)

        # Ensure Phase Interpolation takes the shortest angular path to prevent wrapping tears
        # Used cp.pi instead of np.pi to keep the modulo arithmetic entirely on the GPU
        phase_diff = (phi2_gpu[final_idx] - phi1_gpu[moving_idx] + cp.pi) % (
            2 * cp.pi
        ) - cp.pi
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
            update_state_kernel_linear_3D(
                dw,
                dphi,
                total_dx,
                total_dy,
                total_dz,
                ds,
                curr_w,
                curr_phi,
                curr_x,
                curr_y,
                curr_z,
            )

            # Repack the terms and call the generator.
            terms_gpu = (curr_w, curr_phi, curr_x, curr_y, curr_z)
            pm_slm = self.generate_phasemask(terms_gpu)

            # Only superimpose the moving traps with the pre-calculated static background
            composite_pm = self.superimpose([pm_slm, static_background])

            # Store calculated 2D mask directly into pre-allocated VRAM chunk
            gpu_sequence[n] = self.transform_phase_8bit(composite_pm)

        # Batch copy the entire sequence back to the host (CPU) once at the end
        gpu_sequence.get(out=phasemasks_sequence)

        print(f"Time Taken for {n_steps} frames: {(time.time() - start) * 1000:.4f} ms")
        return phasemasks_sequence

    def iter_rearrangement_sequence(
        self,
        terms1,
        terms2,
        occ_mask,
        d0=0.5,
        profile="minimum_jerk",
        to_host=True,
        refine_iters=0,
        refine_damping=0.5,
        init_refine_iters=0,
        init_refine_damping=0.5,
    ):
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

        ``refine_iters`` > 0 spends the GPU idle time inside each frame: instead of one
        superposition GEMM from the interpolated terms, it runs that many weighted-GS
        weight updates with the interpolated trap phases frozen, warm-started from the
        previous frame. The pipeline is board-write bound (~0.59 ms) against ~0.26 ms of
        compute, so 1-2 iterations land inside the existing cycle time.
        """
        if profile not in ("minimum_jerk", "linear"):
            raise ValueError(
                f"profile must be 'minimum_jerk' or 'linear', got {profile!r}"
            )
        w1, phi1, x1, y1, z1 = terms1
        w2, phi2, x2, y2, z2 = terms2
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

        pos_init = cp.stack((x1, y1, z1), axis=-1)[moving_idx]
        pos_final = cp.stack((x2, y2, z2), axis=-1)[final_idx]
        vec = pos_final - pos_init

        max_dist = cp.linalg.norm(vec, axis=1).max()
        steps_scale = 1.0 if profile == "linear" else 1.875
        n_steps = int(cp.ceil(steps_scale * max_dist / d0))

        # .copy() matters: cp.asarray is a no-op on an array already float32 on the GPU,
        # and update_state_kernel writes in place - without it the caller's terms1 is
        # walked to the final positions and a second rearrangement starts wrong.
        curr_w = cp.asarray(w1, dtype=cp.float32).copy()
        curr_phi = cp.asarray(phi1, dtype=cp.float32).copy()
        curr_x = cp.asarray(x1, dtype=cp.float32).copy()
        curr_y = cp.asarray(y1, dtype=cp.float32).copy()
        curr_z = cp.asarray(z1, dtype=cp.float32).copy()

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
        total_dz = cp.zeros_like(curr_z)

        w2_gpu = cp.asarray(w2)
        phi1_gpu, phi2_gpu = cp.asarray(phi1), cp.asarray(phi2)

        dw[off_mask] = 0.0
        curr_w[off_mask] = 0.0

        if init_refine_iters:
            target_norm = cp.zeros_like(curr_w)
            target_norm[moving_idx] = cp.float32(1.0 / len(moving_idx))
            terms_gpu = (curr_w, curr_phi, curr_x, curr_y, curr_z)
            _, curr_w = self.refine_phasemask(
                terms_gpu, target_norm, init_refine_iters, init_refine_damping
            )

        dw[moving_idx] = (w2_gpu[final_idx] - curr_w[moving_idx]) / n_steps
        total_dx[moving_idx] = vec[:, 0].astype(cp.float32)
        total_dy[moving_idx] = vec[:, 1].astype(cp.float32)
        total_dz[moving_idx] = vec[:, 2].astype(cp.float32)

        phase_diff = (phi2_gpu[final_idx] - phi1_gpu[moving_idx] + cp.pi) % (
            2 * cp.pi
        ) - cp.pi
        dphi[moving_idx] = phase_diff / n_steps

        static_background = self.superimpose([self.fresnel, self.blaze, self.zernike])

        # uniform target intensity across the moving traps; parked traps stay at zero
        target_norm = cp.zeros_like(curr_w)
        target_norm[moving_idx] = cp.float32(1.0 / len(moving_idx))
        refine_w = curr_w.copy() if refine_iters else None

        for n in range(n_steps):
            ds = float(ds_profile_cpu[n])
            update_state_kernel_linear_3D(
                dw,
                dphi,
                total_dx,
                total_dy,
                total_dz,
                ds,
                curr_w,
                curr_phi,
                curr_x,
                curr_y,
                curr_z,
            )
            if refine_iters:
                terms_gpu = (refine_w, curr_phi, curr_x, curr_y, curr_z)
                pm_slm, refine_w = self.refine_phasemask(
                    terms_gpu, target_norm, refine_iters, refine_damping
                )
            else:
                terms_gpu = (curr_w, curr_phi, curr_x, curr_y, curr_z)
                pm_slm = self.generate_phasemask(terms_gpu)
            composite_pm = self.superimpose([pm_slm, static_background])
            frame = self.transform_phase_8bit(composite_pm)
            yield frame.get() if to_host else frame

        pm_slm = self.generate_phasemask(terms2)
        composite_pm = self.superimpose([pm_slm, static_background])
        frame = self.transform_phase_8bit(composite_pm)
        yield frame.get() if to_host else frame

    def iter_transport_sequence(
        self,
        terms1,
        terms2,
        d0=0.5,
        d0_z=None,
        profile="minimum_jerk",
        path_fn=linear_path,
        step_norm="max",
        to_host=True,
        refine_iters=0,
        refine_damping=0.5,
        init_refine_iters=0,
        init_refine_damping=0.5,
    ):
        """Transport trap n from terms1[n] to terms2[n] along path_fn, in 3D.

        CHANGED from iter_rearrangement_sequence (3D):
        * no occupancy mask or assignment: traps are already paired one-to-one
            (index n of terms1 goes to index n of terms2), so the LAP pairing and
            the extinguishing of unused traps are removed;
        * new argument ``path_fn`` (default: straight line, the old behaviour);
        * separate step limits: no trap moves more than ``d0`` in the xy plane
            or ``d0_z`` along z in one frame (``d0_z=None`` means ``d0_z = d0``).
            ``step_norm`` sets how they combine for an oblique step: "max" applies
            each limit on its own; "ellipse" makes them share one budget, and with
            d0_z = d0 is the original isotropic 3D distance criterion;
        * the frame count comes from the PATH (its total step cost), then is
            refined until every step is within the limits;
        * ``dw`` is measured from the weights after init_refine, so the weights
            arrive at w2 on the last interpolated frame.
        """
        w1, phi1, x1, y1, z1 = terms1
        w2, phi2, x2, y2, z2 = terms2

        pos_init = cp.stack(
            [cp.asarray(v, dtype=cp.float32) for v in (x1, y1, z1)], axis=-1
        )
        pos_final = cp.stack(
            [cp.asarray(v, dtype=cp.float32) for v in (x2, y2, z2)], axis=-1
        )

        # NEW: whole trajectory up front, shape (n_steps + 1, N, 3)
        traj = build_trajectory(
            path_fn, pos_init, pos_final, d0, d0_z, profile, step_norm
        )
        n_steps = traj.shape[0] - 1
        traj_x = cp.ascontiguousarray(traj[..., 0])
        traj_y = cp.ascontiguousarray(traj[..., 1])
        traj_z = cp.ascontiguousarray(traj[..., 2])

        # unchanged: .copy() so the caller's terms1 is not modified in place
        curr_w = cp.asarray(w1, dtype=cp.float32).copy()
        curr_phi = cp.asarray(phi1, dtype=cp.float32).copy()
        curr_x = cp.asarray(x1, dtype=cp.float32).copy()
        curr_y = cp.asarray(y1, dtype=cp.float32).copy()
        curr_z = cp.asarray(z1, dtype=cp.float32).copy()

        # unchanged in spirit: uniform intensity target, now over all traps
        target_norm = cp.full_like(curr_w, cp.float32(1.0 / len(curr_w)))

        if init_refine_iters:
            terms_gpu = (curr_w, curr_phi, curr_x, curr_y, curr_z)
            _, curr_w = self.refine_phasemask(
                terms_gpu, target_norm, init_refine_iters, init_refine_damping
            )
            curr_w = cp.asarray(curr_w, dtype=cp.float32).copy()

        # CHANGED: step from the (possibly refined) starting weights to w2
        dw = (cp.asarray(w2, dtype=cp.float32) - curr_w) / n_steps
        phase_diff = (cp.asarray(phi2, dtype=cp.float32) - curr_phi + cp.pi) % (
            2 * cp.pi
        ) - cp.pi
        dphi = (phase_diff / n_steps).astype(cp.float32)

        static_background = self.superimpose([self.fresnel, self.blaze, self.zernike])
        refine_w = curr_w.copy() if refine_iters else None

        for n in range(n_steps):
            # CHANGED: pass the positions for frame n+1 instead of (dx, dy, dz, ds)
            update_state_kernel_path_3D(
                dw,
                dphi,
                traj_x[n + 1],
                traj_y[n + 1],
                traj_z[n + 1],
                curr_w,
                curr_phi,
                curr_x,
                curr_y,
                curr_z,
            )
            if refine_iters:
                terms_gpu = (refine_w, curr_phi, curr_x, curr_y, curr_z)
                pm_slm, refine_w = self.refine_phasemask(
                    terms_gpu, target_norm, refine_iters, refine_damping
                )
            else:
                terms_gpu = (curr_w, curr_phi, curr_x, curr_y, curr_z)
                pm_slm = self.generate_phasemask(terms_gpu)
            composite_pm = self.superimpose([pm_slm, static_background])
            frame = self.transform_phase_8bit(composite_pm)
            yield frame.get() if to_host else frame

        # unchanged: finish on the exact target hologram
        pm_slm = self.generate_phasemask(terms2)
        composite_pm = self.superimpose([pm_slm, static_background])
        frame = self.transform_phase_8bit(composite_pm)
        yield frame.get() if to_host else frame


update_state_kernel_linear_3D = cp.ElementwiseKernel(
    "float32 dw, float32 dphi, float32 dx, float32 dy, float32 dz, float32 ds",
    "float32 w, float32 phi, float32 x, float32 y, float32 z",
    """
    w += dw;
    phi += dphi;
    x += dx * ds;
    y += dy * ds;
    z += dz * ds;
    """,
    "update_hologram_state",
)

update_state_kernel_path_3D = cp.ElementwiseKernel(
    "float32 dw, float32 dphi, float32 x_new, float32 y_new, float32 z_new",
    "float32 w, float32 phi, float32 x, float32 y, float32 z",
    """
    w += dw;
    phi += dphi;
    x = x_new;
    y = y_new;
    z = z_new;
    """,
    "update_hologram_state_path_3D",
)

##################################################################################################################################################################################


class OptimisationBasedPhasemaskGeneratorGPUSeidel:
    def __init__(
        self,
        wavelength_um=0.852,
        focal_length_mm=17.3,
        slm_pitch_um=17,
        slm_res=(1024, 1024),
        input_beam_waist_mm=16,
        fresnel_f_mm=1072,
        blaze_dx_dy_um=(46.50, 10.54),
        zernike_coeff_dict=None,
    ):

        self.lam = wavelength_um
        self.f_mm = focal_length_mm
        self.f_um = focal_length_mm * 1000
        self.dx_slm = slm_pitch_um
        self.Ny, self.Nx = slm_res
        self.w0_mm = input_beam_waist_mm
        self.w0_um = input_beam_waist_mm * 1000
        self.Lx_slm = self.Nx * self.dx_slm  # In units of um
        self.Ly_slm = self.Ny * self.dx_slm  # In units of um
        self.dx_f = (self.lam * self.f_um) / self.Lx_slm  # In units of um
        self.dy_f = (self.lam * self.f_um) / self.Ly_slm  # In units of um
        self.Lx_foc = self.Nx * self.dx_f
        self.Ly_foc = self.Ny * self.dy_f

        self.x_slm = cp.linspace(
            -self.Lx_slm / 2, self.Lx_slm / 2, self.Nx, dtype=cp.float32
        )
        self.y_slm = cp.linspace(
            -self.Ly_slm / 2, self.Ly_slm / 2, self.Ny, dtype=cp.float32
        )

        # Precalculations
        self.k = np.float32(2 * np.pi / (self.lam * self.f_um))
        self.j_k = cp.complex64(1j) * self.k
        self.uint8_scale = cp.float32(255.0) / (2 * cp.pi)

        # Additional phasemasks - phase wrapped and in radians
        self.fresnel = self.generate_fresnel_lens_phasemask(
            focal_length_mm=fresnel_f_mm
        )
        self.blaze = self.generate_blazed_grating_phasemask(
            dx_um=blaze_dx_dy_um[0], dy_um=blaze_dx_dy_um[1]
        )
        if zernike_coeff_dict is None:
            zernike_coeff_dict = DEFAULT_ZERNIKE_COEFFS
        self.zernike = self.generate_zernike_phasemask(zernike_coeff_dict, wrap=True)

        print("--- System Configuration ---")
        print(f"SLM Plane Width: {self.Lx_slm / 1000:.2f} mm")
        print(f"SLM Plane Height: {self.Ly_slm / 1000:.2f} mm")
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
        return cp.exp(-R2 / (2 * (self.w0_um / 2.355) ** 2))

    def generate_weighted_array(
        self, weights, spacing, init_phase_randomness=1.0, angle_deg=0
    ):
        """
        Generates a weighted mask indicating where the tweezers should be and
        how strong they should be.
        Returns both the full 2D array and a list of the specific (y, x) indices.
        """
        dim = weights.shape
        xspan = (dim[1] - 1) * spacing
        yspan = (dim[0] - 1) * spacing
        xpos = np.linspace(-(xspan) / 2, (xspan) / 2, dim[1])
        ypos = np.linspace(-(yspan) / 2, (yspan) / 2, dim[0])
        Xpos, Ypos = np.meshgrid(xpos, ypos)
        Xn, Yn = Xpos.flatten(), Ypos.flatten()
        Xn, Yn = rotate_coordinates(Xn, Yn, angle_deg)
        Wn = weights.flatten()
        Thetan = np.random.rand(len(Wn)) * init_phase_randomness * 2 * np.pi

        print("--- Target Generation ---")
        print(f"Grid: {dim[0]}x{dim[1]}")
        print(f"Spacing: {spacing} um")

        return [Wn, Thetan, Xn, Yn]

    def generate_kagome_lattice(self, target_sites, spacing):
        """
        Generates a circularly symmetric Kagome lattice of tweezer sites.

        Parameters:
        - target_sites (int): The desired number of trap sites.
        - spacing (float): The nearest-neighbor site spacing in microns.

        Returns:
        - x_pos (np.ndarray): Array of x coordinates for the trap sites.
        - y_pos (np.ndarray): Array of y coordinates for the trap sites.
        """
        k_max = int(1.5 * np.sqrt(target_sites)) + 2
        m_vals = np.arange(-k_max, k_max + 1)
        n_vals = np.arange(-k_max, k_max + 1)
        m, n = np.meshgrid(m_vals, n_vals)
        m = m.flatten()
        n = n.flatten()
        a1 = np.array([2 * spacing, 0])
        a2 = np.array([spacing, np.sqrt(3) * spacing])
        b1 = np.array([spacing, 0])
        b2 = np.array([spacing / 2, np.sqrt(3) * spacing / 2])
        b3 = np.array([-spacing / 2, np.sqrt(3) * spacing / 2])
        centers = np.outer(m, a1) + np.outer(n, a2)
        sites_b1 = centers + b1
        sites_b2 = centers + b2
        sites_b3 = centers + b3
        all_sites = np.vstack((sites_b1, sites_b2, sites_b3))
        distances = np.linalg.norm(all_sites, axis=1)
        rounded_distances = np.round(distances, decimals=5)
        unique_radii, counts = np.unique(rounded_distances, return_counts=True)
        cumulative_sites = np.cumsum(counts)
        idx_closest = np.argmin(np.abs(cumulative_sites - target_sites))
        optimal_radius = unique_radii[idx_closest]
        actual_sites = cumulative_sites[idx_closest]
        mask = rounded_distances <= optimal_radius
        final_sites = all_sites[mask]
        x_pos = final_sites[:, 0]
        y_pos = final_sites[:, 1]

        print(
            f"Target sites: {target_sites} | Closest symmetric match: {actual_sites} sites"
        )

        return x_pos, y_pos

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

        return (fresnel_phase % (2 * cp.pi) - cp.pi).astype(cp.float32)

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
        return (blazed_phase % (2 * cp.pi) - cp.pi).astype(cp.float32)

    def superposition_optimization(
        self, target, max_iter=30, damping=0.4, verbose=True
    ):
        """
        Optimizes a phasemask for a discrete array of optical tweezers using
        the weighted superposition method, accelerated by the GPU using CuPy.
        """

        print("--- Starting GPU Superposition Phase Retrieval ---")
        start_time = time.time()

        # 3. Setup Illumination Beam (Gaussian)
        am_slm = cp.asarray(self.generate_source_amplitude())

        # 4. Define Target Trap Coordinates
        w_n, theta_n, x_n, y_n = target

        # Move variables to GPU
        w_n_cp = cp.asarray(w_n)
        theta_n_cp = cp.asarray(theta_n)
        x_n_cp = cp.asarray(x_n)
        y_n_cp = cp.asarray(y_n)
        target_0_cp = cp.asarray(target[0])

        rho = cp.sqrt(self.x_slm[None, :] ** 2 + self.y_slm[:, None] ** 2)

        X_phase = cp.exp(
            self.j_k * x_n_cp[:, None] * self.x_slm[None, :]
            + self.Cx * x_n_cp[:, None]
            + get_zernike_polynomial(8, rho)
        )  # Shape: (N_traps, Nx)
        Y_phase = cp.exp(
            self.j_k * y_n_cp[:, None] * self.y_slm[None, :]
            + self.Cy * y_n_cp[:, None]
            + get_zernike_polynomial(7, rho)
        )  # Shape: (N_traps, Ny)
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
            Y_term = Y_phase * C_n[:, None]  # Shape: (N_traps, Ny)
            U_tot = (
                Y_term.T @ X_phase
            )  # Shape: (Ny, N_traps) @ (N_traps, Nx) -> (Ny, Nx)
            pm_slm = cp.angle(U_tot)

            # Step B: Evaluate Exact Intensity at Target Sites (Vectorized)
            U_slm = am_slm * cp.exp(1j * pm_slm)
            U_foc_n = (
                U_slm @ X_phase_conj.T
            )  # Shape: (Ny, Nx) @ (Nx, N_traps) -> (Ny, N_traps)
            U_foc = cp.sum(Y_phase_conj * U_foc_n.T, axis=1)  # Shape: (N_traps,)
            I_foc = cp.abs(U_foc) ** 2

            # Step C: Calculate Metrics
            I_foc_masked = I_foc[mask_cp]
            uniformity = 1.0 - (
                I_foc_masked.std() / I_foc_masked.mean()
            )  # 1.0 is perfect uniformity
            uniformity_history[iteration] = uniformity

            I_foc_sum = I_foc.sum()
            I_foc_norm = I_foc / I_foc_sum
            mse = cp.mean((I_foc_norm - target_norm) ** 2)
            mse_history[iteration] = mse

            minmax_ratio = I_foc_masked.min() / I_foc_masked.max()
            minmax_history[iteration] = minmax_ratio

            # Step D: Update Weights
            update_ratio = (target_norm / I_foc_norm) ** damping
            w_n_cp = w_n_cp * update_ratio

            # Update the target phases to match the simulated arriving field
            theta_n_cp = cp.angle(U_foc)

            if verbose and iteration % 10 == 0:
                print(
                    f"Iteration {iteration:03d} | Mean-Squared Error: {float(mse):.2e} | Uniformity: {float(uniformity) * 100:.2f}% | Min/Max ratio: {float(minmax_ratio):.3f}"
                )

        trap_weights = I_foc
        print(
            f"Iteration {iteration:03d} | Mean-Squared Error: {float(mse):.2e} | Uniformity: {float(uniformity) * 100:.2f}% | Min/Max ratio: {float(minmax_ratio):.3f}"
        )
        print(f"Optimization finished in {time.time() - start_time:.2f} seconds.")

        return (
            pm_slm,
            [w_n_cp, theta_n_cp, x_n_cp, y_n_cp],
            [
                cp.asnumpy(uniformity_history),
                cp.asnumpy(minmax_history),
                cp.asnumpy(mse_history),
                cp.asnumpy(trap_weights),
            ],
        )

    def generate_phasemask(self, trap_terms):
        """
        Generates the phase mask using pure CuPy high-speed matrix multiplication.
        """
        # Unpack once
        w_n, theta_n, x_n, y_n = trap_terms

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

    def refine_phasemask(self, trap_terms, target_norm, iters, damping=0.5):
        """generate_phasemask plus `iters` weighted-GS updates, trap phases held fixed.

        Only the weights move; theta stays at the interpolated value. Letting theta float
        makes the iteration non-contractive, so freezing it is what allows a handful of
        iterations to help rather than wander.

        target_norm is the desired normalised intensity per site (zero on inactive sites).
        Returns the phasemask and the refined weights, for warm-starting the next frame.
        """
        w_n, theta_n, x_n, y_n = trap_terms
        w = cp.asarray(w_n, dtype=cp.float32)
        theta = cp.asarray(theta_n, dtype=cp.float32)

        X_phase = cp.exp(
            self.j_k * cp.asarray(x_n, dtype=cp.float32)[:, None] * self.x_slm[None, :]
        )
        Y_phase = cp.exp(
            self.j_k * cp.asarray(y_n, dtype=cp.float32)[:, None] * self.y_slm[None, :]
        )
        X_conj, Y_conj = cp.conj(X_phase), cp.conj(Y_phase)
        am_slm = self.generate_source_amplitude().astype(cp.complex64)

        for _ in range(iters):
            pm_slm = cp.angle(
                (Y_phase * (w * cp.exp(cp.complex64(1j) * theta))[:, None]).T @ X_phase
            )
            U_slm = am_slm * cp.exp(cp.complex64(1j) * pm_slm)
            I_foc = cp.abs(cp.sum(Y_conj * (U_slm @ X_conj.T).T, axis=1)) ** 2
            I_norm = I_foc / I_foc.sum()
            # inactive sites carry target_norm == 0 and w == 0, so leave them alone
            ratio = cp.where(
                target_norm > 0,
                (target_norm / cp.maximum(I_norm, cp.float32(1e-20))) ** damping,
                cp.float32(1.0),
            )
            w = w * ratio

        U_tot = (Y_phase * (w * cp.exp(cp.complex64(1j) * theta))[:, None]).T @ X_phase
        return cp.angle(U_tot).astype(cp.float32), w

    def evaluate_traps(self, pm_slm, x_n, y_n):
        am_slm = cp.asarray(self.generate_source_amplitude())
        pm_slm = cp.asarray(pm_slm)
        x_n_cp = cp.asarray(x_n)
        y_n_cp = cp.asarray(y_n)

        X_phase = cp.exp(
            self.j_k * x_n_cp[:, None] * self.x_slm[None, :]
        )  # Shape: (N_traps, Nx)
        Y_phase = cp.exp(
            self.j_k * y_n_cp[:, None] * self.y_slm[None, :]
        )  # Shape: (N_traps, Ny)
        X_phase_conj = cp.conj(X_phase)
        Y_phase_conj = cp.conj(Y_phase)

        U_slm = am_slm * cp.exp(1j * pm_slm)
        U_foc_n = (
            U_slm @ X_phase_conj.T
        )  # Shape: (Ny, Nx) @ (Nx, N_traps) -> (Ny, N_traps)
        U_foc = cp.sum(Y_phase_conj * U_foc_n.T, axis=1)  # Shape: (N_traps,)
        w_foc = cp.abs(U_foc)
        phi_foc = cp.angle(U_foc)
        I_foc = w_foc**2

        return I_foc.get(), phi_foc.get()

    def simulate_focal_plane(
        self,
        pm_slm,
        Nx_pad=2048,
        Ny_pad=2048,
        show=False,
        zoom_pixels=100,
        cmap="viridis",
    ):
        am_slm = cp.asarray(self.generate_source_amplitude())
        field_slm = am_slm * cp.exp(1j * cp.asarray(pm_slm))
        field_slm_pad = cp.zeros((Ny_pad, Nx_pad), dtype=cp.complex128)
        field_slm_pad[
            (Ny_pad - self.Ny) // 2 : (Ny_pad + self.Ny) // 2,
            (Nx_pad - self.Nx) // 2 : (Nx_pad + self.Nx) // 2,
        ] = field_slm
        field_foc_pad = cp.asnumpy(
            cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(field_slm_pad)))
        )

        if show:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            I_foc = np.abs(field_foc_pad) ** 2
            norm_intensity = I_foc / I_foc.max()
            x_focal_um = np.linspace(
                -self.Nx * self.dx_f / 2, self.Nx * self.dx_f / 2, Nx_pad
            )
            y_focal_um = np.linspace(
                -self.Ny * self.dy_f / 2, self.Ny * self.dy_f / 2, Ny_pad
            )
            im1 = ax.imshow(
                norm_intensity,
                extent=[x_focal_um[0], x_focal_um[-1], y_focal_um[0], y_focal_um[-1]],
                cmap=cmap,
            )

            zoom_range_um = zoom_pixels * self.dx_f
            ax.set_xlim(-zoom_range_um, zoom_range_um)
            ax.set_ylim(-zoom_range_um, zoom_range_um)
            ax.grid(False)
            ax.set_title("Simulated Intensity (Focal Plane)\nZoomed Central Region")
            ax.set_xlabel("x [um]")
            ax.set_ylabel("y [um]")
            cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
            cbar1.set_label("Normalized Intensity")

        return field_foc_pad

    def visualise_algorithm_performance(self, pm_slm, metrics, zoom_pixels, pad=1):
        U_foc = self.simulate_focal_plane(
            pm_slm, Nx_pad=self.Nx * pad, Ny_pad=self.Ny * pad, show=False
        )
        I_foc = np.abs(U_foc) ** 2

        fig, ax = plt.subplots(1, 3, figsize=(21, 6))
        uniformity_history, minmax_history, mse_history, trap_weights = metrics

        ax[0].plot(uniformity_history, "b-o", label="Uniformity")
        ax[0].plot(minmax_history, "r-s", label="Min/Max Ratio")
        ax0_secondary = ax[0].twinx()
        ax0_secondary.plot(mse_history, "g-o", label="MSE")
        ax0_secondary.set_ylabel("MSE Value")
        ax0_secondary.tick_params(axis="y")
        ax0_secondary.legend(loc="lower right")
        ax[0].set_xlabel("Iteration")
        ax[0].set_ylabel("Metric Value")
        ax[0].legend(loc="upper right")
        ax[0].grid(True)

        ax[0].set_title("Optimization Convergence Metrics")

        norm_intensity = I_foc / I_foc.max()
        x_focal_um = np.linspace(
            -self.Nx * self.dx_f / 2, self.Nx * self.dx_f / 2, self.Nx * pad
        )
        y_focal_um = np.linspace(
            -self.Ny * self.dy_f / 2, self.Ny * self.dy_f / 2, self.Ny * pad
        )
        im1 = ax[1].imshow(
            norm_intensity,
            extent=[x_focal_um[0], x_focal_um[-1], y_focal_um[0], y_focal_um[-1]],
            cmap="viridis",
        )

        zoom_range_um = zoom_pixels * self.dx_f
        ax[1].set_xlim(-zoom_range_um, zoom_range_um)
        ax[1].set_ylim(-zoom_range_um, zoom_range_um)
        ax[1].grid(False)
        ax[1].set_title("Simulated Intensity (Focal Plane)\nZoomed Central Region")
        ax[1].set_xlabel("x [um]")
        ax[1].set_ylabel("y [um]")
        cbar1 = plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
        cbar1.set_label("Normalized Intensity")

        # Plot a heatmap of the trap weights with ax[2]
        im2 = ax[2].imshow(trap_weights / trap_weights.max(), cmap="viridis")
        ax[2].grid(False)
        ax[2].set_title("Simulated Trap Weights")
        ax[2].set_xlabel("Trap X Index")
        ax[2].set_ylabel("Trap Y Index")
        cbar2 = plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
        cbar2.set_label("Normalized Trap Weight")

        plt.tight_layout()
        plt.show()

    def superimpose(self, phasemasks):
        return cp.mod(sum(phasemasks), 2 * cp.pi)

    def transform_phase_8bit(self, phasemask):
        return ((phasemask + cp.pi) * self.uint8_scale).astype(cp.uint8)

    def generate_rearrangement_sequence(self, terms1, terms2, occ_mask, d0=0.5):
        """
        Calculates the optimal Hungarian rearrangement path and generates
        the full sequence of interpolated phasemasks efficiently on the GPU.
        """
        start = time.time()

        w1, phi1, x1, y1 = terms1
        w2, phi2, x2, y2 = terms2
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
        phase_diff = (phi2_gpu[final_idx] - phi1_gpu[moving_idx] + cp.pi) % (
            2 * cp.pi
        ) - cp.pi
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
            update_state_kernel_linear(
                dw, dphi, total_dx, total_dy, ds, curr_w, curr_phi, curr_x, curr_y
            )

            # Repack the terms and call the generator.
            terms_gpu = (curr_w, curr_phi, curr_x, curr_y)
            pm_slm = self.generate_phasemask(terms_gpu)

            # Only superimpose the moving traps with the pre-calculated static background
            composite_pm = self.superimpose([pm_slm, static_background])

            # Store calculated 2D mask directly into pre-allocated VRAM chunk
            gpu_sequence[n] = self.transform_phase_8bit(composite_pm)

        # Batch copy the entire sequence back to the host (CPU) once at the end
        gpu_sequence.get(out=phasemasks_sequence)

        print(f"Time Taken for {n_steps} frames: {(time.time() - start) * 1000:.4f} ms")
        return phasemasks_sequence

    def iter_rearrangement_sequence(
        self,
        terms1,
        terms2,
        occ_mask,
        d0=0.5,
        profile="minimum_jerk",
        to_host=True,
        refine_iters=0,
        refine_damping=0.5,
        init_refine_iters=0,
        init_refine_damping=0.5,
    ):
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

        ``refine_iters`` > 0 spends the GPU idle time inside each frame: instead of one
        superposition GEMM from the interpolated terms, it runs that many weighted-GS
        weight updates with the interpolated trap phases frozen, warm-started from the
        previous frame. The pipeline is board-write bound (~0.59 ms) against ~0.26 ms of
        compute, so 1-2 iterations land inside the existing cycle time.
        """
        if profile not in ("minimum_jerk", "linear"):
            raise ValueError(
                f"profile must be 'minimum_jerk' or 'linear', got {profile!r}"
            )
        w1, phi1, x1, y1 = terms1
        w2, phi2, x2, y2 = terms2
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

        # .copy() matters: cp.asarray is a no-op on an array already float32 on the GPU,
        # and update_state_kernel writes in place - without it the caller's terms1 is
        # walked to the final positions and a second rearrangement starts wrong.
        curr_w = cp.asarray(w1, dtype=cp.float32).copy()
        curr_phi = cp.asarray(phi1, dtype=cp.float32).copy()
        curr_x = cp.asarray(x1, dtype=cp.float32).copy()
        curr_y = cp.asarray(y1, dtype=cp.float32).copy()

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

        dw[off_mask] = 0.0
        curr_w[off_mask] = 0.0

        dw[moving_idx] = (w2_gpu[final_idx] - w1_gpu[moving_idx]) / n_steps
        total_dx[moving_idx] = vec[:, 0].astype(cp.float32)
        total_dy[moving_idx] = vec[:, 1].astype(cp.float32)

        phase_diff = (phi2_gpu[final_idx] - phi1_gpu[moving_idx] + cp.pi) % (
            2 * cp.pi
        ) - cp.pi
        dphi[moving_idx] = phase_diff / n_steps

        static_background = self.superimpose([self.fresnel, self.blaze, self.zernike])

        # uniform target intensity across the moving traps; parked traps stay at zero
        target_norm = cp.zeros_like(curr_w)
        target_norm[moving_idx] = cp.float32(1.0 / len(moving_idx))
        refine_w = curr_w.copy() if refine_iters else None

        for n in range(n_steps):
            ds = float(ds_profile_cpu[n])
            update_state_kernel_linear(
                dw, dphi, total_dx, total_dy, ds, curr_w, curr_phi, curr_x, curr_y
            )
            if refine_iters:
                terms_gpu = (refine_w, curr_phi, curr_x, curr_y)
                pm_slm, refine_w = self.refine_phasemask(
                    terms_gpu, target_norm, refine_iters, refine_damping
                )
            else:
                terms_gpu = (curr_w, curr_phi, curr_x, curr_y)
                pm_slm = self.generate_phasemask(terms_gpu)
            composite_pm = self.superimpose([pm_slm, static_background])
            frame = self.transform_phase_8bit(composite_pm)
            yield frame.get() if to_host else frame

    def plan_rearrangement(
        self, terms1, terms2, occ_mask, d0=0.5, profile="minimum_jerk"
    ):
        if profile not in ("minimum_jerk", "linear"):
            raise ValueError(
                f"profile must be 'minimum_jerk' or 'linear', got {profile!r}"
            )
        w1, phi1, x1, y1 = terms1
        w2, phi2, x2, y2 = terms2
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

        phase_diff = (phi2_gpu[final_idx] - phi1_gpu[moving_idx] + cp.pi) % (
            2 * cp.pi
        ) - cp.pi
        dphi[moving_idx] = phase_diff / n_steps

        dw[off_mask] = 0.0
        curr_w[off_mask] = 0.0

        static_background = self.superimpose([self.fresnel, self.blaze, self.zernike])

        # The trailing fancy-index writes and the background superposition are
        # async launches. Without this sync their GPU tail would land on frame 0
        # and be billed as frame time - exactly the misattribution this split
        # exists to remove. Costs a stall here, which is the point: it belongs
        # to setup.
        cp.cuda.Stream.null.synchronize()

        return {
            "n_steps": n_steps,
            "ds_profile_cpu": ds_profile_cpu,
            "dw": dw,
            "dphi": dphi,
            "total_dx": total_dx,
            "total_dy": total_dy,
            "curr_w": curr_w,
            "curr_phi": curr_phi,
            "curr_x": curr_x,
            "curr_y": curr_y,
            "static_background": static_background,
            "n_moving": int(moving_idx.size),
            "profile": profile,
            # The JV assignment itself, kept on the device. Reading it here would
            # cost a sync inside the timed setup span, so the caller copies it to
            # the host once the run is over.
            "occ_mask": occ_mask,
            "occ_indices": occ_indices,
            "moving_idx": moving_idx,
            "final_idx": final_idx,
            "pos_init": pos_init,
            "pos_final": pos_final,
            "n_targets": int(pos2.shape[0]),
        }

    def iter_rearrangement_sequence_timings(self, plan, to_host=False):
        dw, dphi = plan["dw"], plan["dphi"]
        total_dx, total_dy = plan["total_dx"], plan["total_dy"]
        curr_w, curr_phi = plan["curr_w"], plan["curr_phi"]
        curr_x, curr_y = plan["curr_x"], plan["curr_y"]
        static_background = plan["static_background"]
        ds_profile_cpu = plan["ds_profile_cpu"]

        for n in range(plan["n_steps"]):
            start_ev = cp.cuda.Event()
            end_ev = cp.cuda.Event()
            start_ev.record()
            ds = float(ds_profile_cpu[n])
            update_state_kernel_linear(
                dw, dphi, total_dx, total_dy, ds, curr_w, curr_phi, curr_x, curr_y
            )
            terms_gpu = (curr_w, curr_phi, curr_x, curr_y)
            pm_slm = self.generate_phasemask(terms_gpu)
            composite_pm = self.superimpose([pm_slm, static_background])
            frame = self.transform_phase_8bit(composite_pm)
            end_ev.record()
            yield (frame.get() if to_host else frame), start_ev, end_ev
