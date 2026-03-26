import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import time
import os
os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
from scipy.spatial.distance import cdist
import lap

class PhaseMaskGenerator:
    def __init__(self,
                 wavelength_um=0.852,
                 focal_length_mm=12.5,
                 slm_pitch_um=8,
                 slm_res=(1200,1920),
                 input_beam_waist_mm=9.6):
        
        self.lam = wavelength_um
        self.f_mm = focal_length_mm
        self.f_um = focal_length_mm * 1000
        self.dx_slm = slm_pitch_um
        self.Ny, self.Nx = slm_res
        self.w0_mm = input_beam_waist_mm
        self.w0_um = input_beam_waist_mm * 1000

        self.Lx = self.Nx * self.dx_slm # In units of um
        self.Ly = self.Ny * self.dx_slm # In units of um

        # Focal Plane Resolution
        self.dx_f = (self.lam * self.f_um) / self.Lx # In units of um
        self.dy_f = (self.lam * self.f_um) / self.Ly # In units of um

        print(f"--- System Configuration ---")
        print(f"SLM Plane Width: {self.Lx/1000:.2f} mm")
        print(f"Focal Plane Resolution x (pixel size): {self.dx_f:.4f} um")
        print(f"Focal Plane Resolution y (pixel size): {self.dy_f:.4f} um")
        print(f"Max Field of View in Focal Plane: {(self.dx_f * self.Nx):.0f} um")

    def generate_source_amplitude(self):
        """Generates Gaussian input beam amplitude."""
        x = np.linspace(-self.Lx/2, self.Lx/2, self.Nx)
        y = np.linspace(-self.Ly/2, self.Ly/2, self.Ny)
        X, Y = np.meshgrid(x, y)
        R2 = X**2 + Y**2

        return np.exp(- R2 / (2*(self.w0_um/2.355)**2))

    def generate_uniform_square_array(self, dim, spacing, pad = 1):
        """
        Generates a boolean mask indicating where the tweezers should be.
        Returns both the full 2D array and a list of the specific (y, x) indices.
        """
        mask = np.zeros((self.Ny*pad, self.Nx*pad))
        
        spacing_pix_x = int(spacing*pad / self.dx_f)
        spacing_pix_y = int(spacing*pad / self.dy_f)

        xspan = (dim[1] - 1) * spacing_pix_x
        yspan = (dim[0] - 1) * spacing_pix_y

        xpos = np.linspace((self.Nx*pad - xspan)//2, (self.Nx*pad + xspan)//2, dim[1])
        ypos = np.linspace((self.Ny*pad - yspan)//2, (self.Ny*pad + yspan)//2, dim[0])

        mask[np.array(ypos, dtype=int)[:, None], np.array(xpos, dtype=int)] = 1
            
        print(f"--- Target Generation ---")
        print(f"Grid: {dim[0]}x{dim[1]}")
        print(f"Padding: x{pad}")
        print(f"Spacing x: {spacing} um ({spacing_pix_x:.2f} pixels = {spacing_pix_x*self.dx_f/pad:.4f} um)")
        print(f"Spacing y: {spacing} um ({spacing_pix_y:.2f} pixels = {spacing_pix_y*self.dy_f/pad:.4f} um)")
                    
        return mask, [ypos, xpos]


    def generate_weighted_square_array(self, weights, spacing, pad = 1):
        """
        Generates a weighted mask indicating where the tweezers should be and
        how strong they should be.
        Returns both the full 2D array and a list of the specific (y, x) indices.
        """
        dim = weights.shape
        mask = np.zeros((self.Ny*pad, self.Nx*pad))
        
        spacing_pix_x = int(spacing*pad / self.dx_f)
        spacing_pix_y = int(spacing*pad / self.dy_f)

        xspan = (dim[1] - 1) * spacing_pix_x
        yspan = (dim[0] - 1) * spacing_pix_y

        xpos = np.linspace((self.Nx*pad - xspan)//2, (self.Nx*pad + xspan)//2, dim[1])
        ypos = np.linspace((self.Ny*pad - yspan)//2, (self.Ny*pad + yspan)//2, dim[0])

        mask[np.array(ypos, dtype=int)[:, None], np.array(xpos, dtype=int)] = weights

        print(f"--- Target Generation ---")
        print(f"Grid: {dim[0]}x{dim[1]}")
        print(f"Padding: x{pad}")
        print(f"Spacing x: {spacing} um ({spacing_pix_x:.2f} pixels = {spacing_pix_x*self.dx_f/pad:.4f} um)")
        print(f"Spacing y: {spacing} um ({spacing_pix_y:.2f} pixels = {spacing_pix_y*self.dy_f/pad:.4f} um)")
                    
        return mask, [ypos, xpos]
    
    def generate_uniform_circle_array(self, N, spacing):
        '''
        Generate circular array defined on a square grid. N is an arbitrary size parameter.
        '''
        y_indices, x_indices = np.ogrid[:N, :N]
        center = (N - 1) / 2.0
        radius = N / 2.0
        dist_squared = (y_indices - center)**2 + (x_indices - center)**2
        mask = dist_squared <= radius**2
        circle_array_mask = mask.astype(int)
        circle_array, pos = self.generate_weighted_square_array(circle_array_mask, spacing)
        print(f"Number of trap sites: {circle_array_mask.sum()}")
        return circle_array, pos
    
    def calculate_correlation(self, array1, array2):
        """
        Calculates Pearson correlation coefficient between two 2D arrays.
        Using the flattened arrays for 1D correlation.
        """
        # Flatten arrays
        f1 = array1.flatten()
        f2 = array2.flatten()
        
        # Fast correlation calculation
        # Correlation = Covariance(X,Y) / (Std(X) * Std(Y))
        # np.corrcoef returns the correlation matrix [[1, r], [r, 1]]
        return np.corrcoef(f1, f2)[0, 1]
    
    def calculate_mse(self, array1, array2):
        """
        Calculates mean-squared error between two 2D arrays.
        """
        f1 = array1 / array1.sum()
        f2 = array2 / array2.sum()

        # Fast mean-squared error calculation
        return np.mean((f1 - f2)**2)
    
    def extract_trap_weights(self, img, ypos, xpos):
        """
        Extract trap weights from an array image with given x and y 
        positions of the trap sites.
        """
        return img[np.array(ypos, dtype=int)[:, None], np.array(xpos, dtype=int)]
    
    def calculate_uniformity(self, matrix):
        """
        Return the standard deviation as a percentage of the mean - 
        coefficient of variability or uniformity
        """
        return 1 - np.std(matrix)/matrix.mean()

    def run_wgs(self, target, indices, max_iter = 100, init_phase_sigma = 2*np.pi, pm_slm = 'random', pm_foc = 'random'):
        img_target = target / target.sum()
        beam = self.generate_source_amplitude()

        if isinstance(pm_slm, str):
            pm_slm = np.random.normal(0, init_phase_sigma, size=(self.Ny, self.Nx))  # Phase pattern in SLM plane - unconstrained, let freely evolve
        if isinstance(pm_foc, str):
            pm_foc = np.random.normal(0, init_phase_sigma, size=(self.Ny, self.Nx))  # Phase pattern in Fourier plane - unconstrained, let freely evolve
        am_slm = np.sqrt(beam)                                      # Amplitude pattern in SLM plane - fixed incident Gaussian beam
        am_foc = np.sqrt(target)                                    # Amplitude pattern in Fourier plane - desired pattern, enforced
        field_slm = am_slm * np.exp(pm_slm * 1j)                    # Complex field pattern in SLM plane

        print(f"--- Running FT-WGS Algorithm ({max_iter} iterations) ---")

        corr, mse, cvar = [], [], []
        for i in range(max_iter):
            # Forwards : SLM -> Fourier plane
            field_foc = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field_slm)))    # Complex field pattern in Fourier plane - FT of complex field pattern in SLM plane
            pm_foc = np.angle(field_foc)                                            # Extracted phase pattern in Fourier plane
            field_foc = am_foc * np.exp(pm_foc * 1j)                                # Switch simulated amplitude pattern with desired amplitude pattern

            # Backwards : Fourier plane -> SLM
            field_slm = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(field_foc))) # Complex field pattern in SLM plane - inv FT of complex field pattern in Fourier plane
            pm_slm = np.angle(field_slm)                                            # Extracted phase pattern in SLM plane
            field_slm = am_slm * np.exp(pm_slm * 1j)                                # Complex field pattern in SLM plane

            # Forwards 
            img_foc = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field_slm)))) ** 2
            img_foc = img_foc / img_foc.sum()
            am_foc = np.multiply(np.sqrt(np.divide(img_target/img_target.sum(), img_foc/img_foc.sum())), am_foc)

            it_corr = self.calculate_correlation(img_target, img_foc)
            it_mse = self.calculate_mse(img_foc, img_target)
            it_cvar = self.calculate_uniformity(self.extract_trap_weights(img_foc, indices[0], indices[1]))

            corr.append(it_corr)
            mse.append(it_mse)
            cvar.append(it_cvar)

            print(f"Iteration {i}: Correlation = {it_corr*100:.4f} %    Uniformity = {it_cvar*100:.4f} %")

        print(f"Final Correlation: {corr[-1]*100:.4f} %    Final Uniformity: {cvar[-1]*100:.4f} %")
        return pm_slm, pm_foc, [corr, mse, cvar]
    
    def run_wgs_pad(self, target, indices, max_iter = 100, init_phase_sigma = 2*np.pi, pm_slm = 'random', pm_foc = 'random'):
        Ny_pad, Nx_pad = target.shape
        img_target = target / target.sum()
        beam = self.generate_source_amplitude()

        if isinstance(pm_slm, str):
            pm_slm = np.random.normal(0, init_phase_sigma, size=(self.Ny, self.Nx))  # Phase pattern in SLM plane - unconstrained, let freely evolve
        if isinstance(pm_foc, str):
            pm_foc = np.random.normal(0, init_phase_sigma, size=(self.Ny, self.Nx))  # Phase pattern in Fourier plane - unconstrained, let freely evolve
        am_slm = np.sqrt(beam)                                      # Amplitude pattern in SLM plane - fixed incident Gaussian beam
        am_foc = np.sqrt(target)                                    # Amplitude pattern in Fourier plane - desired pattern, enforced
        field_slm = am_slm * np.exp(pm_slm * 1j)                    # Complex field pattern in SLM plane

        field_slm_pad = np.zeros((Ny_pad, Nx_pad), dtype=np.complex128)
        field_slm_pad[(Ny_pad - self.Ny)//2:(Ny_pad + self.Ny)//2, (Nx_pad - self.Nx)//2:(Nx_pad + self.Nx)//2] = field_slm

        print(f"--- Running FT-WGS Zero-Padding Algorithm ({max_iter} iterations) ---")

        corr, mse, cvar = [], [], []
        for i in range(max_iter):
            # Forwards : SLM -> Fourier plane
            field_foc_pad = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field_slm_pad)))    # Complex field pattern in Fourier plane - FT of complex field pattern in SLM plane
            pm_foc_pad = np.angle(field_foc_pad)                                            # Extracted phase pattern in Fourier plane
            field_foc_pad = am_foc * np.exp(pm_foc_pad * 1j)                                # Switch simulated amplitude pattern with desired amplitude pattern

            # Backwards : Fourier plane -> SLM
            field_slm_pad = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(field_foc_pad))) # Complex field pattern in SLM plane - inv FT of complex field pattern in Fourier plane
            pm_slm_pad = np.angle(field_slm_pad)                                            # Extracted phase pattern in SLM plane
            pm_slm = pm_slm_pad[(Ny_pad - self.Ny)//2:(Ny_pad + self.Ny)//2, (Nx_pad - self.Nx)//2:(Nx_pad + self.Nx)//2]
            field_slm = am_slm * np.exp(pm_slm * 1j)                                        # Complex field pattern in SLM plane
            field_slm_pad[(Ny_pad - self.Ny)//2:(Ny_pad + self.Ny)//2, (Nx_pad - self.Nx)//2:(Nx_pad + self.Nx)//2] = field_slm 

            # Forwards 
            img_foc = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field_slm_pad)))) ** 2
            img_foc = img_foc / img_foc.sum()
            am_foc = np.multiply(np.sqrt(np.divide(img_target/img_target.sum(), img_foc/img_foc.sum())), am_foc)

            it_corr = self.calculate_correlation(img_target, img_foc)
            it_mse = self.calculate_mse(img_foc, img_target)
            it_cvar = self.calculate_uniformity(self.extract_trap_weights(img_foc, indices[0], indices[1]))

            corr.append(it_corr)
            mse.append(it_mse)
            cvar.append(it_cvar)

            print(f"Iteration {i}: Correlation = {it_corr*100:.4f} %    Uniformity = {it_cvar*100:.4f} %")

        print(f"Final Correlation: {corr[-1]*100:.4f} %    Final Uniformity: {cvar[-1]*100:.4f} %")
        return pm_slm, pm_foc, [corr, mse, cvar]
    
    def recover_fourier_field_wgs(self, phasemask, pad_factor=2):
        '''
        Forward propagate (ASM) phasemask to simulate the
        complex field in the Fourier plane.
        '''
        Ny_pad, Nx_pad = self.Ny * pad_factor, self.Nx * pad_factor
        am_slm = np.sqrt(self.generate_source_amplitude())

        field_slm = am_slm * np.exp(1j * phasemask)
        field_slm_pad = np.zeros((Ny_pad, Nx_pad), dtype=np.complex128)
        field_slm_pad[(Ny_pad - self.Ny)//2:(Ny_pad + self.Ny)//2, (Nx_pad - self.Nx)//2:(Nx_pad + self.Nx)//2] = field_slm

        field_foc_pad = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field_slm_pad)))
        field_foc = field_foc_pad[(Ny_pad - self.Ny)//2:(Ny_pad + self.Ny)//2, (Nx_pad - self.Nx)//2:(Nx_pad + self.Nx)//2]
        return field_foc_pad

    def generate_fresnel_lens(self, f_mm):
        f_um = - f_mm * 1000
        x_slm = np.linspace(-self.Lx/2, self.Lx/2, self.Nx)
        y_slm = np.linspace(-self.Ly/2, self.Ly/2, self.Ny)
        X_slm, Y_slm = np.meshgrid(x_slm, y_slm)
        r2 = X_slm**2 + Y_slm**2
        pm_lens = -(np.pi * r2) / (self.lam * f_um) % (2*np.pi) - np.pi
        return pm_lens

    def save_phasemask(self, phasemask):
        phasemask_bmp = Image.fromarray(phasemask)
        phasemask_bmp.save('C:\\Users\\CaFMOT\\OneDrive - Imperial College London\\caftweezers\\MeadowController\\phasemasks\\phasemask.bmp')
        print('Phasemask generated and saved as an 8bit.bmp')

    def superimpose_8bit(self, phasemasks):
        '''
        Superimpose 8-bit phasemasks.
        '''
        return (sum(phasemasks) % 256).astype(np.uint8)
    
    def transform_phase_8bit(self, phasemask):
        '''
        Converts phasemask to 8-bit bitmaps.
        '''
        return np.uint8(((phasemask/(2*np.pi))*256) + 128)
    
    def show_algorithm_performance(self, metrics):
        plt.figure(figsize=(8, 6))

        # Plot 1: Convergence
        ax1 = plt.subplot(1, 1, 1)
        ax1.plot(metrics[0], 'b-o', label="Correlation")
        ax1.plot(metrics[2], 'g-o', label='Uniformity')
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Correlation", color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(metrics[1], 'r-s', label="MSE")
        ax2.set_ylabel("MSE", color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.legend(loc='upper right')
        plt.title("Algorithm Convergence")

        plt.tight_layout()
        plt.show()
    
    def show_algorithm_output(self, target, indices, pm_slm, zoom_pixels=50, pad=2, pad_sim=1):
        
        field_foc = self.recover_fourier_field_wgs(pm_slm, pad_factor=pad*pad_sim)
        img_foc = np.abs(field_foc) ** 2

        # 5. Visualization
        # Prepare spatial axes for plotting
        x_slm_mm = np.linspace(-self.Lx/2, self.Lx/2, self.Nx) / 1e3
        y_slm_mm = np.linspace(-self.Ly/2, self.Ly/2, self.Ny) / 1e3

        x_focal_um = np.linspace(-self.Nx*self.dx_f/2, self.Nx*self.dx_f/2, self.Nx)
        y_focal_um = np.linspace(-self.Ny*self.dy_f/2, self.Ny*self.dy_f/2, self.Ny)

        # Create figure
        fig, ax = plt.subplots(1, 3, figsize=(21, 12))

        # Plot 2: The Resulting Intensity (What the atoms see)
        # Zoom in to the center to see the array
        center = self.Ny // 2

        # Target intensity
        norm_intensity = img_foc / img_foc.max()

        im1 = ax[0].imshow(target, 
                            extent=[x_focal_um[0], x_focal_um[-1], y_focal_um[0], y_focal_um[-1]],
                            cmap='viridis')

        # Set zoom limits
        zoom_range_um = zoom_pixels * self.dx_f
        ax[0].set_xlim(-zoom_range_um, zoom_range_um)
        ax[0].set_ylim(-zoom_range_um, zoom_range_um)
        ax[0].grid(False)
        ax[0].set_title(f"Target Intensity (Focal Plane)\nZoomed Central Region")
        ax[0].set_xlabel("x [um]")
        ax[0].set_ylabel("y [um]")
        cbar1 = plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
        cbar1.set_label("Target Intensity")

        # Normalize intensity
        norm_intensity = img_foc / img_foc.max()

        x_focal_um = np.linspace(-self.Nx*self.dx_f/2, self.Nx*self.dx_f/2, self.Nx*pad*pad_sim)
        y_focal_um = np.linspace(-self.Ny*self.dy_f/2, self.Ny*self.dy_f/2, self.Ny*pad*pad_sim)

        im2 = ax[1].imshow(norm_intensity, 
                            extent=[x_focal_um[0], x_focal_um[-1], y_focal_um[0], y_focal_um[-1]],
                            cmap='viridis')

        # Set zoom limits
        zoom_range_um = zoom_pixels * self.dx_f
        ax[1].set_xlim(-zoom_range_um, zoom_range_um)
        ax[1].set_ylim(-zoom_range_um, zoom_range_um)
        ax[1].grid(False)
        ax[1].set_title(f"Simulated Intensity (Focal Plane)\nZoomed Central Region")
        ax[1].set_xlabel("x [um]")
        ax[1].set_ylabel("y [um]")
        cbar2 = plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)
        cbar2.set_label("Normalized Intensity")


        W_meas = self.extract_trap_weights(norm_intensity, indices[0]*pad_sim, indices[1]*pad_sim)
        img3 = ax[2].matshow(W_meas)
        cbar3 = plt.colorbar(img3, ax=ax[2], fraction=0.046, pad=0.04)
        cbar3.set_label("Normalized Trap Weights")
        ax[2].set_title(f"Simulated Normalised Trap Weights")

        plt.tight_layout()
        plt.show()

    def generate_iso_weights_matrix(self, n_rows, n_cols, isotype='iso'):
        W = np.ones((n_rows, 2*n_cols+1))
        for i, j in np.ndindex(W.shape):
            if isotype=='iso':
                W[i, j] = (-1)**(i+j)
                W[W == -1] = 0
            elif isotype=='hex':
                W[i, j] = (-1)**((i+j))
                if ((j+3*i))%(6)==0: W[i,j] = -1
                W[W == -1] = 0
            elif isotype=='kagome':
                W[i, j] = (-1)**((i+j))
                if i%2==0 and ((j+i)/2)%2!=0: W[i,j] = -1
                W[W == -1] = 0
        string = f"W = np.array" + f"({W.astype('i')})".replace(' ', ', ')
        print(f'Total Number of Atoms = {W.sum()}')
        return string, W

    def generate_weighted_isometric_array(self, W, spacing_um):
        xspacing = round(spacing_um / self.dx_f)
        yspacing = round(spacing_um / self.dy_f)
        xspan, yspan = (W.shape[1] - 1) * xspacing * 1, (W.shape[0] - 1) * yspacing * 0.866 * 2
        x_pos = np.linspace(-xspan//2, + xspan//2, W.shape[1]) + self.dx//2
        y_pos = np.linspace(-yspan//2, + yspan//2, W.shape[0]) + self.dy//2
        img = np.zeros((self.dx, self.dy))
        img[np.array(y_pos, dtype=int)[:, None], np.array(x_pos, dtype=int)] = W
        return img, x_pos, y_pos

    def phasemask_interpolation(self, phasemask_init, phasemask_final, n_steps):
        # 1. Create the array of step indices: [1, 2, ... n_steps-1]
        # 2. Reshape to (N, 1, 1) so it broadcasts against the (H, W) images
        steps = np.arange(n_steps)[:, None, None]
        
        Dphi = phasemask_final - phasemask_init
        dphi = Dphi / n_steps
        
        # Broadcasting magic:
        # (H,W) + (H,W) * (N,1,1) -> (N,H,W)
        return phasemask_init + dphi * steps
    
############################################################################################################################################################################################################

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


class OptimisationBasedPhasemaskGenerator:
    def __init__(self,
                 wavelength_um=0.852,
                 focal_length_mm=10.0,
                 slm_pitch_um=8,
                 slm_res=(1200,1920),
                 input_beam_waist_mm=9.6):
        
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

        print(f"--- System Configuration ---")
        print(f"SLM Plane Width: {self.Lx_slm/1000:.2f} mm")
        print(f"SLM Plane Height: {self.Ly_slm/1000:.2f} mm")
        print(f"Focal Plane Resolution x (pixel size): {self.dx_f:.4f} um")
        print(f"Focal Plane Resolution y (pixel size): {self.dy_f:.4f} um")
        print(f"Focal Plane Width: {self.Lx_foc:.2f} um")
        print(f"Focal Plane Height: {self.Ly_foc:.2f} um")

    def generate_source_amplitude(self):
        """Generates Gaussian input beam amplitude."""
        x = np.linspace(-self.Lx_slm/2, self.Lx_slm/2, self.Nx)
        y = np.linspace(-self.Ly_slm/2, self.Ly_slm/2, self.Ny)
        X, Y = np.meshgrid(x, y)
        R2 = X**2 + Y**2
        return np.exp(- R2 / (2*(self.w0_um/2.355)**2))
    
    def generate_weighted_array(self, weights, spacing, init_phase_randomness=1.0):
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
        x_slm = np.linspace(-self.Lx_slm/2, self.Lx_slm/2, self.Nx)
        y_slm = np.linspace(-self.Ly_slm/2, self.Ly_slm/2, self.Ny)
        X_slm, Y_slm = np.meshgrid(x_slm, y_slm)
        R_max = np.sqrt(self.Lx_slm**2 + self.Ly_slm**2) / 2.0
        rho = np.sqrt(X_slm**2 + Y_slm**2) / R_max
        theta = np.arctan2(Y_slm, X_slm)
        
        # 3. Define the unit circle mask
        mask = (rho <= 1.0).astype(float)
        
        # Initialize an empty phase array
        total_phase = np.zeros((self.Ny, self.Nx))
        
        # 4. Superpose the requested Zernike modes
        for noll_index, coeff in zernike_coeffs.items():
            if coeff != 0.0:
                Z_mode = get_zernike_polynomial(noll_index, rho, theta, mask)
                total_phase += coeff * Z_mode
                
        if wrap:
            total_phase = total_phase % (2 * np.pi) - np.pi

        return total_phase
    
    def generate_fresnel_lens_phasemask(self, focal_length_mm):
        """
        Generates a Fresnel lens phase mask to focus the beam at a specific focal length.
        This is useful for correcting defocus or creating a virtual focus plane.
        """
        # 1. Setup SLM Coordinates (Centered at 0)
        x_slm = np.linspace(-self.Lx_slm/2, self.Lx_slm/2, self.Nx)
        y_slm = np.linspace(-self.Ly_slm/2, self.Ly_slm/2, self.Ny)
        X_slm, Y_slm = np.meshgrid(x_slm, y_slm)
        
        # 2. Calculate the Fresnel lens phase profile
        f_um = focal_length_mm * 1000
        k = 2 * np.pi / self.lam
        R_squared = X_slm**2 + Y_slm**2
        fresnel_phase = (k / (2 * f_um)) * R_squared
        
        return fresnel_phase % (2*np.pi) - np.pi
    
    def generate_blazed_grating_phasemask(self, dx_um, dy_um):
        """
        Generates a blazed grating phase mask to steer the beam by (dx_um, dy_um) in the focal plane.
        The steering angles are calculated based on the desired displacement and the system's focal length.
        """
        # 1. Setup SLM Coordinates (Centered at 0)
        x_slm = np.linspace(-self.Lx_slm/2, self.Lx_slm/2, self.Nx)
        y_slm = np.linspace(-self.Ly_slm/2, self.Ly_slm/2, self.Ny)
        X_slm, Y_slm = np.meshgrid(x_slm, y_slm)
        
        # 2. Calculate steering angles
        theta_x = np.arctan(dx_um / self.f_um)  # Steering angle in radians for x
        theta_y = np.arctan(dy_um / self.f_um)  # Steering angle in radians for y
        
        # 3. Calculate the blazed grating phase profile
        k = 2 * np.pi / self.lam
        blazed_phase = k * (X_slm * np.sin(theta_x) + Y_slm * np.sin(theta_y))
        return blazed_phase % (2*np.pi) - np.pi

    def superposition_optimization(self, target, max_iter=30, damping=0.4, verbose=True):
        """
        Optimizes a phasemask for a discrete array of optical tweezers using 
        the weighted superposition method.
        """
        print("--- Starting Superposition Phase Retrieval ---")
        start_time = time.time()

        # 2. Setup SLM Physical Coordinates
        x_slm = np.linspace(-self.Lx_slm/2, self.Lx_slm/2, self.Nx)
        y_slm = np.linspace(-self.Ly_slm/2, self.Ly_slm/2, self.Ny)

        # 3. Setup Illumination Beam (Gaussian)
        am_slm = self.generate_source_amplitude()

        # 4. Define Target Trap Coordinates
        w_n, theta_n, x_n, y_n, array_shape = target
        N_traps = len(w_n)
        
        # Pre-calculations
        k = 2 * np.pi / (self.lam * self.f_um)

        X_phase = np.exp(1j * k * x_n[:, None] * x_slm[None, :]) # Shape: (N_traps, Nx)
        Y_phase = np.exp(1j * k * y_n[:, None] * y_slm[None, :]) # Shape: (N_traps, Ny)
        X_phase_conj = np.conj(X_phase)
        Y_phase_conj = np.conj(Y_phase)

        uniformity_history = []
        minmax_history = []
        mse_history = []
        for iteration in range(max_iter):
            
            # Step A: Generate the Superposition Complex Field (Vectorized)
            C_n = w_n * np.exp(1j * theta_n)
            Y_term = Y_phase * C_n[:, None]   # Shape: (N_traps, Ny)
            U_tot = Y_term.T @ X_phase      # Shape: (Ny, N_traps) @ (N_traps, Nx) -> (Ny, Nx)
            pm_slm = np.angle(U_tot)
            
            # Step B: Evaluate Exact Intensity at Target Sites (Vectorized)
            U_slm = am_slm * np.exp(1j * pm_slm)
            U_foc_n = U_slm @ X_phase_conj.T # Shape: (Ny, Nx) @ (Nx, N_traps) -> (Ny, N_traps)
            U_foc = np.sum(Y_phase_conj * U_foc_n.T, axis=1) # Shape: (N_traps,)
            I_foc = np.abs(U_foc)**2
                
            # Step C: Calculate Metrics
            uniformity = 1 - (I_foc[target[0] > 0].std() / I_foc[target[0] > 0].mean()) # 1.0 is perfect uniformity
            uniformity_history.append(uniformity)
            mse = np.mean((I_foc/I_foc.sum() - target[0]/target[0].sum())**2)
            mse_history.append(mse)
            minmax_ratio = I_foc[target[0] > 0].min() / I_foc[target[0] > 0].max()
            minmax_history.append(minmax_ratio)
            
            # Step D: Update Weights
            # We dampen the update to prevent oscillations
            update_ratio = ((target[0]/target[0].sum()) / (I_foc/I_foc.sum()))**damping
            w_n = w_n * update_ratio
            
            # Update the target phases to match the simulated arriving field
            theta_n = np.angle(U_foc)
            
            if verbose:
                if iteration % 10 == 0:
                    print(f"Iteration {iteration:03d} | Mean-Squared Error: {mse:.2e} | Uniformity: {uniformity*100:.2f}% | Min/Max ratio: {minmax_ratio:.3f}")

        trap_weights = I_foc.reshape(array_shape)
        print(f"Iteration {iteration:03d} | Mean-Squared Error: {mse:.2e} | Uniformity: {uniformity*100:.2f}% | Min/Max ratio: {minmax_ratio:.3f}")
        print(f"Optimization finished in {time.time() - start_time:.2f} seconds.")
        
        return pm_slm, [w_n, theta_n, x_n, y_n, array_shape], [uniformity_history, minmax_history, mse_history, trap_weights]

    def zernike_superposition_optimisation_v0(self, target, target_zernike, max_iter=30, damping=0.4):
        """
        Optimizes a phasemask for a discrete array of optical tweezers using
        the weighted superposition method.
        """
        print("--- Starting Superposition Phase Retrieval ---")
 
        # 2. Setup SLM Physical Coordinates
        x_slm = np.linspace(-self.Lx_slm/2, self.Lx_slm/2, self.Nx)
        y_slm = np.linspace(-self.Ly_slm/2, self.Ly_slm/2, self.Ny)
        X_slm, Y_slm = np.meshgrid(x_slm, y_slm)
 
        # 3. Setup Illumination Beam (Gaussian)
        am_slm = self.generate_source_amplitude()
 
        # 4. Define Target Trap Coordinates
        w_n, theta_n, x_n, y_n, array_shape = target
        N_traps = len(w_n)
        
        # Pre-calculations
        print("Performing pre-calculations...")
        k = 2 * np.pi / (self.lam * self.f_um)
        dXY = np.exp(1j * k * np.einsum('n,ij->nij', x_n, X_slm) + 1j * k * np.einsum('n,ij->nij', y_n, Y_slm))
        dXYZ = dXY * np.array([np.exp(1j * self.generate_zernike_phasemask(target_zernike[n])) for n in range(N_traps)])
        dXYZ_conj = np.conj(dXYZ)
 
        uniformity_history = np.zeros(max_iter)
        mse_history = np.zeros(max_iter)
        minmax_history = np.zeros(max_iter)
        start_time = time.time()
        for it in range(max_iter):
            # Calculate SLM field
            """
            U_slm = np.zeros((self.Ny, self.Nx), dtype=np.complex128)
            for n in range(N_traps): U_slm += w_n[n] * dX[n] * dY[n] * np.exp(1j*theta_n[n])"""
            # Pre-calculate the complex coefficients (1D array)
            opt_params = w_n * np.exp(1j * theta_n)
            U_slm = np.einsum('n,nij->ij', opt_params, dXYZ)
            pm_slm = np.angle(U_slm)
            U_slm = am_slm * np.exp(1j * pm_slm)
 
            # Evaluate Fourier field
            U_n = np.einsum('ij,nij->n', U_slm, dXYZ_conj)
            I_n = np.abs(U_n) ** 2
 
            # Heuristic updates
            w_n = w_n * ((target[0]/target[0].sum()) / (I_n/I_n.sum()))**damping
            theta_n = np.angle(U_n)
 
            # Calculate metrics
            uniformity = 1 - (I_n[target[0] > 0].std() / I_n[target[0] > 0].mean()) # 1.0 is perfect uniformity
            uniformity_history[it] = uniformity
            mse = np.mean((I_n/I_n.sum() - target[0]/target[0].sum())**2)
            mse_history[it] = mse
            minmax_ratio = I_n[target[0] > 0].min() / I_n[target[0] > 0].max()
            minmax_history[it] = minmax_ratio
 
            print(f"Iteration {it+1:03d} | Mean-Squared Error: {mse:.2e} | Uniformity: {uniformity*100:.2f}% | Min/Max ratio: {minmax_ratio:.3f}")
 
        print(f"Optimization finished in {time.time() - start_time:.2f} s | Algorithm speed: {max_iter/(time.time() - start_time):.3f} it/s")
        trap_weights = I_n.reshape(array_shape)
 
        return pm_slm, [w_n, theta_n, x_n, y_n], [uniformity_history, minmax_history, mse_history, trap_weights]

    def generate_phasemask(self, trap_terms, zernike_terms=None):
        w_n, theta_n, x_n, y_n = trap_terms
        z_n = zernike_terms if zernike_terms is not None else [{} for _ in range(len(w_n))]
        N_traps = len(w_n)

        x_slm = np.linspace(-self.Lx_slm/2, self.Lx_slm/2, self.Nx)
        y_slm = np.linspace(-self.Ly_slm/2, self.Ly_slm/2, self.Ny)
        X_slm, Y_slm = np.meshgrid(x_slm, y_slm)

        k = 2 * np.pi / (self.lam * self.f_um)
        U_tot = np.zeros((self.Ny, self.Nx), dtype=np.complex128)
        
        for n in range(N_traps):
            delta = k * (x_n[n] * X_slm + y_n[n] * Y_slm)
            zernike = self.generate_zernike_phasemask(z_n[n])
            U_tot += w_n[n] * np.exp(1j * (delta + theta_n[n] + zernike))

        pm_slm = np.angle(U_tot)
        return pm_slm
 
    def simulate_focal_plane(self, pm_slm, Nx_pad=2048, Ny_pad=2048, show=False, zoom_pixels=100, cmap='viridis'):
        am_slm = self.generate_source_amplitude()
        field_slm = am_slm * np.exp(1j * pm_slm)
        field_slm_pad = np.zeros((Ny_pad, Nx_pad), dtype=np.complex128)
        field_slm_pad[(Ny_pad - self.Ny)//2:(Ny_pad + self.Ny)//2, (Nx_pad - self.Nx)//2:(Nx_pad + self.Nx)//2] = field_slm
        field_foc_pad = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field_slm_pad)))

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

    def superimpose_8bit(self, phasemasks):
        '''
        Superimpose 8-bit phasemasks.
        '''
        return (sum(phasemasks) % 256).astype(np.uint8)
    
    def transform_phase_8bit(self, phasemask):
        '''
        Converts phasemask to 8-bit bitmaps.
        '''
        return np.uint8(((phasemask/(2*np.pi))*256) + 128)
 
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
                 focal_length_mm=10.0,
                 slm_pitch_um=8,
                 slm_res=(1200,1920),
                 input_beam_waist_mm=9.6,
                 fresnel_f_mm=600.0,
                 blaze_dx_dy_um=(63.0,7.0),
                 zernike_coeff_dict={5:-0.858, 6:0.282, 7:-1.725, 8:-0.434, 9:1.033, 10:-0.123, 11:1.048}):
        
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
    
    def generate_weighted_array(self, weights, spacing, init_phase_randomness=1.0):
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
        w_n, theta_n, x_n, y_n, _ = trap_terms
        
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
    
        w1, phi1, x1, y1, _ = terms1
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
        
        max_dist = cp.sqrt((vec**2).sum(axis=1)).max()
        n_steps = int(max_dist // d0)
            
        # 3. INITIALIZE VRAM STATE MACHINE
        curr_w = cp.asarray(w1, dtype=cp.float32)
        curr_phi = cp.asarray(phi1, dtype=cp.float32)
        curr_x = cp.asarray(x1, dtype=cp.float32)
        curr_y = cp.asarray(y1, dtype=cp.float32)
        
        # Initialize step vectors with zeros
        dw = cp.zeros_like(curr_w)
        dphi = cp.zeros_like(curr_phi)
        dx = cp.zeros_like(curr_x)
        dy = cp.zeros_like(curr_y)
        
        # Load steps for MOVING traps
        dw[moving_idx] = cp.asarray((w2[final_idx] - w1[moving_idx]) / n_steps)
        dx[moving_idx] = cp.asarray(vec[:, 0] / n_steps)
        dy[moving_idx] = cp.asarray(vec[:, 1] / n_steps)
        
        # Ensure Phase Interpolation takes the shortest angular path to prevent 2*pi wrapping tears
        phase_diff = (phi2[final_idx] - phi1[moving_idx] + np.pi) % (2 * np.pi) - np.pi
        dphi[moving_idx] = cp.asarray(phase_diff / n_steps)
        
        # Load steps for OFF traps (Ramp down weights to 0)
        dw[off_mask] = cp.asarray(-w1[off_mask] / n_steps)
        
        phasemasks_sequence = np.empty((n_steps, self.Ny, self.Nx), dtype=np.uint8)
        gpu_sequence = cp.empty((n_steps, self.Ny, self.Nx), dtype=cp.uint8)
        
        
        # 4. THE ULTRA-FAST GPU LOOP
        for n in range(n_steps):
            # Advance state seamlessly in VRAM
            curr_w += dw
            curr_phi += dphi
            curr_x += dx
            curr_y += dy
            
            # Repack the terms and call the generator.
            terms_gpu = (curr_w, curr_phi, curr_x, curr_y, 0)
            pm_slm = self.generate_phasemask(terms_gpu)
            composite_pm = self.superimpose([pm_slm, self.fresnel, self.blaze, self.zernike])
            composite_pm_uint8 = self.transform_phase_8bit(composite_pm)
            
            # Pull the calculated 2D mask back to host memory (CPU)
            gpu_sequence[n] = composite_pm_uint8

        gpu_sequence.get(out=phasemasks_sequence)
            
        print(f"Time Taken for {n_steps} frames: {(time.time() - start)*1000:.4f} ms")
        return phasemasks_sequence