import numpy as np
import matplotlib.pyplot as plt
from diffractio import mm, um, np, plt, degrees
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
from PIL import Image


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

        if pm_slm == 'random':
            pm_slm = np.random.normal(0, init_phase_sigma, size=(self.Ny, self.Nx))  # Phase pattern in SLM plane - unconstrained, let freely evolve
        if pm_foc == 'random':
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
    
    def run_wgs_pad(self, target, indices, max_iter = 100, init_phase_sigma = 2*np.pi):
        Ny_pad, Nx_pad = target.shape
        img_target = target / target.sum()
        beam = self.generate_source_amplitude()

        pm_slm = np.random.normal(0, init_phase_sigma, size=(self.Ny, self.Nx))  # Phase pattern in SLM plane - unconstrained, let freely evolve
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
    
    def generate_zernike_polynomials(self, zernike_data, scale=1.0):
        '''
        Generates Zernike polynomial phasemask based on a list of
        (n, m, cnm) Zernike modes (n,m) and coefficients (cnm).
        '''
        n, m, cnm = zernike_data[:, 0].astype(int), zernike_data[:, 1].astype(int), zernike_data[:, 2]
        h, w = self.Ny, self.Nx
        M = self.dx_slm * 1e6 * um                      # SLM pixel size in um
        wl = self.lam * 1e6 * um                        # Wavelength of source
        diameter= scale * w * M                         # Diameter of Zernike mode

        x0 = np.arange(0,w) * M
        y0 = np.arange(0,h) * M
        zernike = Scalar_source_XY(x=x0, y=y0, wavelength=wl)
        zernike.zernike_beam(A=1, r0=(w/2*M, h/2*M), radius=diameter/2, n=n, m=m, c_nm=cnm)
        zernike_pm = np.angle(zernike.u)
        return zernike_pm
    
    def generate_blazed_grating(self, period, amp=2*np.pi, angle=0):
        '''
        Generates a blazed grating with some period, angle, and amplitude.
        Use angle = 90 for a vertical blazed grating.
        '''
        h, w = self.Ny, self.Nx
        M = self.dx_slm * 1e6 * um                      # SLM pixel size in um
        wl = self.lam * 1e6 * um                        # Wavelength of source
        x0 = np.arange(0, w) * M
        y0 = np.arange(0, h) * M
        period = period * um
        grating = Scalar_mask_XY(x0, y0, wl)
        grating.blazed_grating(period=period, phase_max=amp, x0=0, angle=angle * degrees)
        grating_pm = np.angle(grating.u)
        return grating_pm

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

    def generate_binary_grating(self, period_px, minval, maxval):
        # Not in phase units - output is in 8bit 0->255
        x0 = np.arange(0, self.Nx) * self.dx_slm
        y0 = np.arange(0, self.Ny) * self.dx_slm
        grating = Scalar_mask_XY(x0, y0, self.lam)
        grating.binary_grating(
            period=period_px*self.dx_slm,
            a_min=minval,
            a_max=maxval,
            phase=np.pi,
            x0=0,
            fill_factor=0.5,
            angle=0 * degrees,
        )
        grating = np.angle(grating.u)
        grating = (grating - grating.min()) / (grating.max() - grating.min()) * (maxval - minval) + minval
        return grating
    
    def generate_blazed_grating(self, period, amp=2*np.pi, angle=0):  
        x0 = np.arange(0, self.Nx) * self.dx_slm
        y0 = np.arange(0, self.Ny) * self.dx_slm
        period = period * um
        grating = Scalar_mask_XY(x0, y0, self.lam)
        grating.blazed_grating(period=period, phase_max=amp, x0=0, angle=angle * degrees)
        grating_pm = np.angle(grating.u)
        return grating_pm

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
    

