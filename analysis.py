import numpy as np
from zipfile import ZipFile
from PIL import Image
import re
from typing import Any, Dict
from scipy.optimize import curve_fit
from scipy.ndimage import center_of_mass, label
import matplotlib.pyplot as plt
import scipy.constants as cn
import os
import shutil
from time import sleep
import matplotlib.patches as patches
from rich.progress import track
from scipy.special import erf

cloudpath = "C:\\Users\\CaFMOT\\OneDrive - Imperial College London\\"
tweezer_img_source_dir = "C:\\Users\\CaFMOT\\OneDrive - Imperial College London\\caftweezers\\HamCamImages\\"
root = cloudpath + "caftweezers\\mot_master_data"
remote_path = "C:\\Users\\cafmot\\OneDrive - Imperial College London (1)\\Desktop\\MOTCamSave\\ThorCam Images\\"
RbMassAMU = 86.909184

def cool_vco_to_detuning(vco):
    return (vco - 4.55)*12.26

def detuning_to_cool_vco(detuning):
    return detuning/12.26 + 4.55

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def EllipticalGaussian2D(pos, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = pos
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    gaussian = offset + amplitude * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo)
                            + c * ((y - yo) ** 2)))
    return gaussian.ravel()

def gaussian2D(pos, amplitude, xo, yo, sigma_x, sigma_y, offset):
    x, y = pos
    xo = float(xo)
    yo = float(yo)
    gaussian = offset + amplitude * np.exp(- ( ((x-xo)**2)/(2*sigma_x**2) + ((y-yo)**2)/(2*sigma_y**2) ))
    return gaussian.ravel()

def spot_sharpness(img):
    pixels = img.ravel()
    M = np.sum(pixels)**2 / np.sum(np.square(pixels))
    return M

def detect_bright_points(image_array, threshold=200):
    # Step 1: Convert image to binary (bright spots = True, background = False)
    binary_image = image_array > threshold  # Apply threshold to create a binary mask

    # Step 2: Label connected components (each bright spot gets a unique label)
    labeled_array, num_features = label(binary_image)

    # Step 3: Find the brightest pixel in each region (brightest pixel corresponds to the highest intensity)
    centers = []
    for region_id in range(1, num_features + 1):
        # Get the coordinates of the pixels in the current region (bright spot)
        coordinates = np.argwhere(labeled_array == region_id)

        # Find the coordinates of the brightest pixel in this region
        brightest_pixel = None
        max_intensity = -1
        for y, x in coordinates:
            if image_array[y, x] > max_intensity:
                max_intensity = image_array[y, x]
                brightest_pixel = (y, x)

        # Append the brightest pixel's coordinates (y, x) as a tuple
        centers.append(brightest_pixel)
    
    centers = np.array(centers)
    
    return centers.astype(int)  # Return centers as an array of (y, x) coordinates

# Sort detected points into a 2D grid and return (i, j) labels
def sort_into_grid(centers, grid_shape = [2,2]):
    num_row, num_col = grid_shape
    if len(centers) != num_row*num_col:
        #print("Looking for array sites...")
        return 'error'
    else:
        sorted_centers = np.array(sorted(centers, key=lambda p: p[0]))
        y_sorted_array = sorted_centers.reshape(tuple([num_row, num_col, 2]))
        sorted_array = np.array([np.array(sorted(row, key=lambda p: p[1])) for row in y_sorted_array])
        grid_positions = {}
        for index in np.ndindex(sorted_array.shape[:-1]):
            grid_positions[index] = tuple(sorted_array[index])
        return grid_positions, num_row, num_col

def detect_trap_sites(img_array, grid_shape, detection_step = 100):
    print('Looking for trap sites...')
    detection_threshold = img_array.max()
    while 1:
        if detection_threshold < 1: 
            print('Could not detect.')
            break
            
        centers = detect_bright_points(img_array, threshold=detection_threshold)
        output = sort_into_grid(centers, grid_shape)
        if output == 'error':
            detection_threshold -= detection_step
        else:
            grid_positions, num_rows, num_cols = output
            print(f'{grid_shape[0]}x{grid_shape[1]} array detected.')
            return grid_positions, detection_threshold

# Sum pixel values in a 5x5 region around each detected center
def sum_pixel_values(image_array, grid_positions, grid_shape, window_size=10):
    half_size = window_size // 2
    pixel_sums = np.zeros(grid_shape, dtype=int)  # Create empty 2D array

    for (i, j),(y, x) in grid_positions.items():
        # Extract 5x5 region and sum pixel values
        region = image_array[max(y-half_size, 0):min(y+half_size+1, image_array.shape[0]),
                             max(x-half_size, 0):min(x+half_size+1, image_array.shape[1])]
        pixel_sums[i, j] = np.sum(region)

    return pixel_sums

# Function to visualize results with cropping and zooming
def visualize_results(image_array, grid_positions, margin=50, window_size=5, threshold=150, vmaxfactor = 0.8):
    # Get bounding box around detected points
    y_vals, x_vals = zip(*grid_positions.values())  # Extract y and x coordinates
    min_y, max_y = min(y_vals), max(y_vals)
    min_x, max_x = min(x_vals), max(x_vals)

    # Define crop boundaries with a margin of 50 pixels
    y1 = max(min_y - margin, 0)
    y2 = min(max_y + margin, image_array.shape[0])
    x1 = max(min_x - margin, 0)
    x2 = min(max_x + margin, image_array.shape[1])

    # Crop the image
    cropped_image = image_array[y1:y2, x1:x2]
    cropped_bin_image = (cropped_image > threshold)

    # Adjust positions of grid labels for cropped view
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(cropped_image, cmap="gray", extent=[x1, x2, y2, y1], vmax=vmaxfactor*cropped_image.max())
    ax[0].grid(False)
    ax[1].imshow(cropped_image, cmap="gray", extent=[x1, x2, y2, y1], vmax=vmaxfactor*cropped_image.max())  # Use extent to maintain coordinates
    ax[2].imshow(cropped_bin_image, cmap='viridis', extent=[x1, x2, y2, y1])

    # Draw 5x5 squares and labels
    half_size = window_size // 2
    for (i, j), (y, x) in grid_positions.items():
        # Draw grid label
        ax[1].text(x+5, y, f"({i},{j})", color='white', fontsize=12, weight='bold')

        # Draw a 5x5 square centered on (x, y)
        rect0 = patches.Rectangle((x - half_size, y - half_size), window_size, window_size,
                                 linewidth=1, edgecolor='red', facecolor='none')
        rect1 = patches.Rectangle((x - half_size, y - half_size), window_size, window_size,
                                 linewidth=1, edgecolor='red', facecolor='none')
        ax[1].add_patch(rect0)
        ax[2].add_patch(rect1)
    plt.show()

def detect_loading_threshold(photon_rates):
    photon_rates = np.array(photon_rates)
    entries, bin_edges = np.histogram(photon_rates.ravel(), bins=30, range=(photon_rates.min(),photon_rates.max()))
    entries = entries/sum(entries)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    bg_gauss_fit = lambda x, a, b, c: a*np.exp(-(x-b)**2 / (2*c**2))
    params, _ = curve_fit(bg_gauss_fit, bin_centers, entries, p0=[0.2, bin_centers[0]*1.05, bin_centers[0]*0.1])

    xx = np.linspace(bin_centers[0], bin_centers[-1], 1000)
    yy = bg_gauss_fit(xx, *params)

    A, mu, sigma = params
    loading_threshold = mu + 4*sigma
    
    return loading_threshold

def detect_trap_sites_general(img_array, atom_number, detection_step=100):
    print('Looking for trap sites...')
    detection_threshold = img_array.max()
    while 1:
        if detection_threshold < 1: 
            print('Could not detect.')
            break
            
        centers = detect_bright_points(img_array, threshold=detection_threshold)
        if len(centers) != atom_number:
            detection_threshold -= detection_step
        else:
            print(f'{len(centers)} Atoms Detected.')
            break
    grid_positions = {}
    for i in range(len(centers)):
        grid_positions[i] = tuple(centers[i])
    return grid_positions, detection_threshold

def visualize_array_detection(image_array, grid_positions, margin=50, window_size=5, threshold=150, vmaxfactor = 0.8):
    # Get bounding box around detected points
    y_vals, x_vals = zip(*grid_positions.values())  # Extract y and x coordinates
    min_y, max_y = min(y_vals), max(y_vals)
    min_x, max_x = min(x_vals), max(x_vals)

    # Define crop boundaries with a margin of 50 pixels
    y1 = max(min_y - margin, 0)
    y2 = min(max_y + margin, image_array.shape[0])
    x1 = max(min_x - margin, 0)
    x2 = min(max_x + margin, image_array.shape[1])

    # Crop the image
    cropped_image = image_array[y1:y2, x1:x2]
    cropped_bin_image = (cropped_image > threshold)

    # Adjust positions of grid labels for cropped view
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(cropped_image, cmap="gray", extent=[x1, x2, y2, y1], vmax=vmaxfactor*cropped_image.max())  # Use extent to maintain coordinates
    ax[1].imshow(cropped_bin_image, cmap='viridis', extent=[x1, x2, y2, y1])

    # Draw 5x5 squares and labels
    half_size = window_size // 2
    for (i), (y, x) in grid_positions.items():
        # Draw grid label
        ax[0].text(x+5, y, f"({i})", color='white', fontsize=12, weight='bold')

        # Draw a 5x5 square centered on (x, y)
        rect0 = patches.Rectangle((x - half_size, y - half_size), window_size, window_size,
                                 linewidth=1, edgecolor='red', facecolor='none')
        rect1 = patches.Rectangle((x - half_size, y - half_size), window_size, window_size,
                                 linewidth=1, edgecolor='red', facecolor='none')
        ax[0].add_patch(rect0)
        ax[1].add_patch(rect1)
    plt.show()

def sum_pixel_values_general(image_array, grid_positions, window_size=10):
    half_size = window_size // 2
    pixel_sums = np.zeros(len(grid_positions), dtype=int)  # Create empty 2D array

    for (i),(y, x) in grid_positions.items():
        # Extract 5x5 region and sum pixel values
        region = image_array[max(y-half_size, 0):min(y+half_size+1, image_array.shape[0]),
                             max(x-half_size, 0):min(x+half_size+1, image_array.shape[1])]
        pixel_sums[i] = np.sum(region)
    return pixel_sums

def maxwell_boltzmann_cdf(P, Amp, Pc, P_offset):
    """
    Fits the loading probability curve assuming a thermal ensemble.
    
    Args:
        P: Tweezer Power (x-axis)
        Amp: Saturation Amplitude (Max loading prob, e.g. ~0.55)
        Pc: Characteristic Power (Proportional to Temperature)
        P_offset: Shift in power (e.g. AOM turn-on threshold or background offset)
    """
    # Shift P by offset and clip negative values to 0 (physics requires P >= 0)
    P_eff = np.maximum(P - P_offset, 1e-10) 
    
    # Calculate ratio U/kT ~ P/Pc
    ratio = P_eff / Pc
    sqrt_ratio = np.sqrt(ratio)
    
    # The CDF formula
    term1 = erf(sqrt_ratio)
    term2 = (2.0 / np.sqrt(np.pi)) * sqrt_ratio * np.exp(-ratio)
    
    return Amp * (term1 - term2)


########################################################################################################################################################################################

class TweezerExperimentAnalysis:
    def __init__(self,
                 year='26',
                 month='01Jan',
                 day='00'):
        self.dirPath = root + "\\{}\\{}\\{}".format(year, month, day)
        self.fileNameString = f"Tweezer{day}{month[2:]}{year[2:]}00"
        print(f"---- {day}/{month}/20{year} Tweezer Experiment Analysis Initialised ----")

    def get_next_zipno(self):
        zip_numbers = [int(f[17:-4]) for f in os.listdir(self.dirPath)]
        lastZip = zip_numbers[-1]
        startZip = lastZip + 1
        print(f"Initial zip no: {startZip}")
        return startZip

    def read_images_from_zip(self, zipNo, close: bool = True) -> np.ndarray:
        zipFileName = "{}//{}_{}.zip".format(self.dirPath, self.fileNameString, str(zipNo).zfill(3))
        archive = ZipFile(zipFileName)
        images = []
        filenames = archive.namelist()
        filenames.sort(key=natural_keys)
        for filename in filenames:
            if filename[-3:] == "tif":
                with archive.open(filename) as image_file:
                    images.append(
                        np.array(
                            Image.open(image_file),
                            dtype=float
                        )
                    )
        if close:
            archive.close()
        return np.array(images)
    
    def read_parameters_from_zip(self, zipNo, close: bool = True) -> Dict[str, Any]:
        zipFileName = "{}//{}_{}.zip".format(self.dirPath, self.fileNameString, str(zipNo).zfill(3))
        archive = ZipFile(zipFileName)
        parameters = {}
        for filename in archive.namelist():
            if filename[-14:] == "parameters.txt":
                with archive.open(filename) as parameter_file:
                    script_parameters = parameter_file.readlines()
                    for line in script_parameters:
                        name, value, _ = line.split(b"\t")
                        parameters[name.decode("utf-8")] = float(value)
            elif filename[-18:] == "hardwareReport.txt":
                with archive.open(filename) as hardware_file:
                    hardware_parameters = hardware_file.readlines()
                    for line in hardware_parameters:
                        name, value, _ = line.split(b"\t")
                        if value.isdigit():
                            parameters[name.decode("utf-8")] = float(value)
        if close:
            archive.close()
        return parameters

    def getTemperature(self, startZipNo, endZipNo, bgZipNo, variable="TOF", show_images=False, incl_init_velocity=False):
        x0, x0_err, y0, y0_err = [], [], [], []
        sx, sx_err, sy, sy_err = [], [], [], []
        t = []
        M = (0.50/0.65) * 3.45 * 1e-6 # pixel -> metre conversion factor

        for zipNo in range(startZipNo, endZipNo + 1):
            imgs, params_dict = self.read_images_from_zip(zipNo), self.read_parameters_from_zip(zipNo)
            bg_img = self.read_images_from_zip(bgZipNo)
            bg_subtracted = (imgs.mean(axis=0) - bg_img.mean(axis=0))
            x = np.arange(0, bg_subtracted.shape[1])
            y = np.arange(0, bg_subtracted.shape[0])
            x, y = np.meshgrid(x, y)

            # Fit gaussian to cloud
            com_y, com_x = center_of_mass(bg_subtracted)
            p0 = [bg_subtracted.ravel().max() - bg_subtracted.ravel().min(), com_x, com_y, 70, 70, bg_subtracted.ravel().min()]
            params, pcov = curve_fit(gaussian2D, (x, y), bg_subtracted.ravel(), p0)
            perr = np.sqrt(np.diag(pcov))

            if show_images:
                plt.figure()
                plt.imshow(bg_subtracted)
                plt.scatter(params[1], params[2], marker="+", color="white", alpha=0.5)

            # Convert cloud positions and uncertainties to metres
            x0.append(params[1]*M)
            y0.append(params[2]*M)
            x0_err.append(perr[1]*M)
            y0_err.append(perr[2]*M)

            # Convert cloud sigma widths and uncertainties to metres
            sx.append(min(params[3], params[4]) * M)
            sy.append(max(params[3], params[4]) * M)
            
            sx_err.append(perr[np.where(params == min(params[3], params[4]))[0]] * M)
            sy_err.append(perr[np.where(params == max(params[3], params[4]))[0]] * M)

            t.append(params_dict[variable] * 1e-5) # TOF variable convert to seconds
            print(f'A = {round(params[0], 2)},  x0 = {round(params[1], 2)},  y0 = {round(params[2], 2)},  dx = {round(params[3], 2)},  dy = {round(params[4], 2)}')
            
        sx2 = np.array(sx) ** 2 # Squared width m^2 units
        sy2 = np.array(sy) ** 2 # Squared width m^2 units
        x0 = np.array(x0)
        y0 = np.array(y0)
        t = np.array(t) # Time in s units
        t2 = t ** 2 # Squared time s^2 units

        # Fitting a straight line to the cloud expansion - slope units m^2 / s^2
        lin = lambda x, m, c: m * x + c
        popt_x, cov_x = curve_fit(lin, t2, sx2)  
        popt_y, cov_y = curve_fit(lin, t2, sy2)
        perr_lin_x = np.sqrt(np.diag(cov_x))
        perr_lin_y = np.sqrt(np.diag(cov_y))

        # Calculating temperatures - units of uK
        Tx = round(popt_x[0] * (RbMassAMU * cn.u / cn.k) * 1E6, 2) 
        Tx_err = round(perr_lin_x[0] * (RbMassAMU * cn.u / cn.k) * 1E6, 2)
        Ty = round(popt_y[0] * (RbMassAMU * cn.u / cn.k) * 1e6, 2)
        Ty_err = round(perr_lin_y[0] * (RbMassAMU * cn.u / cn.k) * 1e6, 2)

        # Fitting a parabola to the cloud freefall - accel units m / s^2, init velocity units m / s^-1, init pos units m
        vx, vy = 0, 0
        if incl_init_velocity:
            grav = lambda tau, a, b, c: 0.5*a*tau**2 + b*tau + c
        else:
            grav = lambda tau, a, c: 0.5*a*tau**2 + c
        post_y, covs_y = curve_fit(grav, t, y0, p0=[9.81, 0.0, y0[0]]) 
        post_x, covs_x = curve_fit(grav, t, x0)
        if incl_init_velocity:
            vx = post_x[1]
            vy = post_y[1]
        else:
            vx, vy = 0, 0

        # Graphing
        fig, ax = plt.subplots(1, 4, figsize=(12, 3))
        plt.tight_layout()
        tt = np.linspace(min(t), max(t), 100)
        tt2 = tt ** 2
        
        ax[0].errorbar(t2 * 1E6, sx2 * 1E6, yerr=np.array(sx_err).flatten()**2*1E6, fmt='o', color='C0', ecolor='C0')
        ax[0].plot(tt2* 1E6, lin(tt2, *popt_x)* 1E6, color='C0')
        ax[0].set_xlabel('$t^{2}$ (ms$^{2}$)')
        ax[0].set_ylabel('$\sigma^{2}$ (mm$^{2}$)')
        ax[0].set_title(f'$T_x$ = {Tx} uK, Error = {Tx_err} uK')
        
        ax[1].errorbar(t2 * 1E6, sy2 * 1E6, yerr=np.array(sy_err).flatten()**2*1E6, fmt='o', color='C3', ecolor='C3')
        ax[1].plot(tt2* 1E6, lin(tt2, *popt_y)* 1E6, color='C3')
        ax[1].set_xlabel('$t^{2}$ (ms$^{2}$)')
        ax[1].set_ylabel('$\sigma^{2}$ (mm$^{2}$)')
        ax[1].set_title(f'$T_y$ = {Ty} uK, Error = {Ty_err} uK')
        
        ax[2].errorbar(t * 1E3, x0 * 1E3, yerr=np.array(x0_err).flatten()*1E3, fmt='o', color='C0', ecolor='C0')
        ax[2].plot(tt * 1E3, grav(tt, *post_x) * 1E3, color='C0')
        ax[2].set_xlabel('$t$ (ms)')
        ax[2].set_ylabel('$x_0$ (mm)')
        ax[2].set_title(f'$a_x$ = {round(post_x[0]/9.81, 2)} g')
        
        ax[3].errorbar(t * 1E3, y0 * 1E3, yerr=np.array(y0_err).flatten()*1E3, fmt='o', color='C3', ecolor='C3')
        ax[3].plot(tt * 1E3, grav(tt, *post_y) * 1E3, color='C3')
        ax[3].set_xlabel('$t$ (ms)')
        ax[3].set_ylabel('$y_0$ (mm)')
        ax[3].set_title(f'$a_y$ = {round(post_y[0]/9.81, 2)} g')

        return Tx, Tx_err, Ty, Ty_err, post_x[0]/9.81, post_y[0]/9.81, vy
    
    def getPos(self, startZipNo, endZipNo, bgZipNo, variable="TOF"):
        pos_x, pos_y = [], []
        sigma_x, sigma_y = [], []
        t = []
        M = (0.65/0.50)*6.45*1e-6

        for zipNo in range(startZipNo, endZipNo + 1):
            imgs, params_dict = self.read_images_from_zip(zipNo), self.read_parameters_from_zip(zipNo)
            bg_img = self.read_images_from_zip(bgZipNo)
            
            bg_subtracted = (imgs.mean(axis=0) - bg_img.mean(axis=0))
            x = np.arange(0, bg_subtracted.shape[1])
            y = np.arange(0, bg_subtracted.shape[0])
            x, y = np.meshgrid(x, y)
            p0 = [bg_subtracted.ravel().max() - bg_subtracted.ravel().min(), 500, 700, 100, 100, 0, bg_subtracted.ravel().min()]
            x = x.ravel()
            y = y.ravel()
            params, pcov = curve_fit(gaussian2D, (x, y), bg_subtracted.ravel(), p0)

            fitGaussian = gaussian2D((x,y), *params).reshape(bg_subtracted.shape)
            plt.figure()
            plt.imshow(bg_subtracted)
            plt.contour(fitGaussian, levels=[0.4*params[0],0.6*params[0],0.8*params[0]], colors=['white'], alpha=0.5)
            plt.scatter(params[1], params[2], marker="+", color="white", alpha=0.5)

            # Multiply positions and sigmas by the imaging magnification factor M
            pos_x.append(params[1]*M)
            pos_y.append(params[2]*M)
            sigma_x.append(params[3]*M)
            sigma_y.append(params[4]*M)
            t.append(params_dict[variable])
            print(params)

        sigma_x = np.array(sigma_x) ** 2
        sigma_y = np.array(sigma_y) ** 2 
        t_pos=(np.array(t) * 1e-5)  # Multiply times to convert units to s
        t = (np.array(t) * 1e-5) ** 2 # Square times for expansion fitting

        lin = lambda x, m, c: m * x + c
        popt_x, cov_x = curve_fit(lin, t, sigma_x)
        popt_y, cov_y = curve_fit(lin, t, sigma_y)
        
        #grav = lambda x, a, b, c: 0.5*a*x**2 + b*x + c
        def grav(tau, a, b, c): return  0.5*a*tau**2  + c+ b*tau #+ b*tau
        post_y, covs_y = curve_fit(grav, t_pos, pos_y) # Use the y position to calc freefall accel.

        fig, ax = plt.subplots(1, 2, figsize=(12, 3))
        ax[0].scatter(t_pos, pos_x)
        #ax[1].plot(t_pos, grav(t_pos, *post_y))
        ax[0].set_xlabel('$t$ (s)')
        ax[0].set_ylabel('$x_0$ (m)')
        ax[0].set_title("Pos x")
        
        ax[1].scatter(t_pos, pos_y)
        #ax[2].plot(t_pos, grav(t_pos, *post_y))
        ax[1].set_xlabel('$t$ (s)')
        ax[1].set_ylabel('$y_0$ (m)')
        ax[1].set_title("Pos y")
        
        return t_pos, pos_x, pos_y
    
    def get_cloud_width(self, startZipNo, endZipNo, bgZipNo, variable="TOF"):
        pos_x, pos_y = [], []
        sigma_x, sigma_y = [], []
        x_err, y_err = [], []
        t = []
        M = 0.96 * (0.50/0.65) * 3.45*1e-6

        for zipNo in range(startZipNo, endZipNo + 1):
            imgs, params_dict = self.read_images_from_zip(zipNo), self.read_parameters_from_zip(zipNo)
            bg_img = self.read_images_from_zip(bgZipNo)
            bg_subtracted = (imgs.mean(axis=0) - bg_img.mean(axis=0))
            
            x = np.arange(0, bg_subtracted.shape[1])
            y = np.arange(0, bg_subtracted.shape[0])
            x, y = np.meshgrid(x, y)
            
            x_flat = x.ravel()
            y_flat = y.ravel()
            
            #com_y, com_x = center_of_mass(bg_subtracted)
            #p0 = [bg_subtracted.ravel().max() - bg_subtracted.ravel().min(), com_x, com_y, 100, 100, 0, bg_subtracted.ravel().min()]
            #params, pcov = curve_fit(gaussian2D, (x_flat, y_flat), bg_subtracted.ravel(), p0)
            #perr = np.sqrt(np.diag(pcov))
            #fitGaussian = gaussian2D((x, y), *params).reshape(bg_subtracted.shape)

            com_y, com_x = center_of_mass(bg_subtracted)
            p0 = [bg_subtracted.ravel().max() - bg_subtracted.ravel().min(), com_x, com_y, 70, 70, 0, bg_subtracted.ravel().min()]
            params, pcov = curve_fit(gaussian2D, (x, y), bg_subtracted.ravel(), p0)
            perr = np.sqrt(np.diag(pcov))
            
            plt.figure()
            plt.imshow(bg_subtracted)
            #plt.contour(fitGaussian, levels=[0.4 * params[0], 0.6 * params[0], 0.8 * params[0]], colors=['white'], alpha=0.5)
            plt.scatter(params[1], params[2], marker="+", color="white", alpha=0.5)

            pos_x.append(params[1] * M)
            pos_y.append(params[2] * M)
            x_err.append(perr[1] * M)
            y_err.append(perr[2] * M)
            sigma_x.append( min(params[3]*M, params[4]*M))
            sigma_y.append( max(params[3]*M, params[4]*M))
            t.append(params_dict[variable])
            print(f'A = {round(params[0], 4)},  x0 = {round(params[1], 4)},  y0 = {round(params[2], 4)},  dx = {round(params[3], 4)},  dy = {round(params[4], 4)}')

        sigma_x = np.array(sigma_x) ** 2
        sigma_y = np.array(sigma_y) ** 2
        x_err = np.array(x_err) ** 2
        y_err = np.array(y_err) ** 2

        t_pos = np.array(t) 
        t = np.array(t) 

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].errorbar(t, sigma_x * 1e6, x_err * 1e6, fmt='ob', label='Horizontal')
        ax[0].set_xlabel(variable)
        ax[0].set_ylabel('$\sigma^{2}_x$ (m$^{2}$)')
        
        ax[1].errorbar(t, sigma_y * 1e6, y_err * 1e6, fmt='or', label='Vertical')
        ax[1].set_xlabel(variable)
        ax[1].set_ylabel('$\sigma^{2}_y$ (m$^{2}$)')

        return sigma_x, x_err, sigma_y, y_err
    
    def get_cloud_size(self, startZipNo, endZipNo, bgZipNo, variable="TOF"):
        pos_x, pos_y = [], []
        sigma_x, sigma_y = [], []
        t = []
        M = 0.96 * (0.50/0.65) * 3.45*1e-6

        for zipNo in range(startZipNo, endZipNo + 1):
            imgs, params_dict = self.read_images_from_zip(zipNo), self.read_parameters_from_zip(zipNo)
            bg_img = self.read_images_from_zip(bgZipNo)
            bg_subtracted = (imgs.mean(axis=0) - bg_img.mean(axis=0))
            x = np.arange(0, bg_subtracted.shape[1])
            y = np.arange(0, bg_subtracted.shape[0])
            x, y = np.meshgrid(x, y)
            p0 = [bg_subtracted.ravel().max() - bg_subtracted.ravel().min(), 600, 600, 100, 100, 0, bg_subtracted.ravel().min()]
            params, pcov = curve_fit(gaussian2D, (x, y), bg_subtracted.ravel(), p0)
            fitGaussian = gaussian2D((x,y), *params).reshape(bg_subtracted.shape)
            plt.figure()
            plt.imshow(bg_subtracted)
            plt.contour(fitGaussian, levels=[0.4*params[0],0.6*params[0],0.8*params[0]], colors=['white'], alpha=0.5)
            plt.scatter(params[1], params[2], marker="+", color="white", alpha=0.5)

            # Multiply positions and sigmas by the imaging magnification factor M
            pos_x.append(params[1]*M)
            pos_y.append(params[2]*M)
            sigma_x.append(params[3]*M)
            sigma_y.append(params[4]*M)
            t.append(params_dict[variable])
            #print(params)

        sigma_x = np.array(sigma_x) ** 2
        sigma_y = np.array(sigma_y) ** 2 
    
        return t, sigma_x, sigma_y
    
    def get_cloud_size_den(self, startZipNo, endZipNo, bgZipNo, variable="TOF"):
        pos_x, pos_y = [], []
        sigma_x, sigma_y = [], []
        t = []
        M = (0.65/0.50)*6.45*1e-6

        for zipNo in range(startZipNo, endZipNo + 1):
            imgs, params_dict = self.read_images_from_zip(zipNo), self.read_parameters_from_zip(zipNo)
            bg_img = self.read_images_from_zip(bgZipNo)
            bg_subtracted = (imgs.mean(axis=0) - bg_img.mean(axis=0))
            x = np.arange(0, bg_subtracted.shape[1])
            y = np.arange(0, bg_subtracted.shape[0])
            x, y = np.meshgrid(x, y)
            p0 = [bg_subtracted.ravel().max() - bg_subtracted.ravel().min(), 600, 600, 100, 100, 0, bg_subtracted.ravel().min()]
            params, pcov = curve_fit(gaussian2D, (x, y), bg_subtracted.ravel(), p0)
            fitGaussian = gaussian2D((x,y), *params).reshape(bg_subtracted.shape)
            plt.figure()
            plt.imshow(bg_subtracted)
            plt.contour(fitGaussian, levels=[0.4*params[0],0.6*params[0],0.8*params[0]], colors=['white'], alpha=0.5)
            plt.scatter(params[1], params[2], marker="+", color="white", alpha=0.5)

            # Multiply positions and sigmas by the imaging magnification factor M
            pos_x.append(params[1]*M)
            pos_y.append(params[2]*M)
            sigma_x.append(min(params[3]*M, params[4]*M))
            sigma_y.append(max(params[3]*M, params[4]*M))
            t.append(params_dict[variable])
            #print(params)

        sigma_x = (np.array(sigma_x)*100*np.sqrt(2*3.14)) ** 2
        sigma_y = (np.array(sigma_y)*100*np.sqrt(2*3.14)) ** 2 
        return t, sigma_x, sigma_y
    
    def get_fluorescence_atom_no(self, file_start, file_end, bg_fileno, parameter_name, crop, to_crop=False, show_images=True):
        crop_x1, crop_x2, crop_y1, crop_y2 = crop
        n_container = []
        n_err_container = []
        t = []
        CamRate = 10.5
        for fileno in range(file_start, file_end + 1):
            images, parameters_dict = self.read_images_from_zip(fileno), self.read_parameters_from_zip(fileno)
            bg, pp = self.read_images_from_zip(bg_fileno), self.read_parameters_from_zip(bg_fileno)
            
            im = images - bg.mean(axis=0)
            if to_crop:
                im = im[:,crop_x1:crop_x2, crop_y1:crop_y2]
            n = im.sum(axis=(1,2))
            mean = n.mean()
            err = n.std() / np.sqrt(im.shape[0])
            n_container.append(mean) 
            n_err_container.append(err)
            t.append(parameters_dict[parameter_name])
            if show_images:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.imshow(im.mean(axis=0), cmap="coolwarm")
        n_container = np.array(n_container)*CamRate
        n_err_container = np.array(n_err_container)*CamRate
        t = np.array(t)
        return t, n_container, n_err_container
    
    def get_temp_v_parameter(self, startZip, endZip, bgZip, no_zips_per_datapoint, parameter='tMolCool'):
        total_zips = len(range(startZip, endZip + 1)) 
        no_datapoints = total_zips / no_zips_per_datapoint
        Tx, Txe, Ty, Tye, P, Gx, Gy, Vy, N, Ne = [], [], [], [], [], [], [], [], [], []
        for i in range(0, int(no_datapoints)):
            tx, txe, ty, tye, gx, gy, vy = self.getTemperature(startZip + i * no_zips_per_datapoint, startZip + (i+1) * no_zips_per_datapoint - 1, bgZip, variable="tImgTOF");
            _, n, ne = self.get_fluorescence_atom_no(startZip + i * no_zips_per_datapoint, startZip + i * no_zips_per_datapoint, bgZip, parameter, [0, -1, 0, -1], to_crop=False, show_images=False);
            p = self.read_parameters_from_zip(startZip + i * no_zips_per_datapoint)[parameter]
        
            Tx.append(tx)
            Txe.append(txe)
            Ty.append(ty)
            Tye.append(tye)
            P.append(p)
            Gx.append(gx)
            Gy.append(gy)
            Vy.append(vy)
            N.append(n)
            Ne.append(ne)
            
        return Tx, Txe, Ty, Tye, P, Gx, Gy, Vy, N, Ne
    
    def get_cloud_density(self, file_start, file_end, bg_fileno, parameter_name, crop, to_crop=False, show_images=True):
        t, n_container, n_err_container = self.get_fl_n(file_start, file_end, bg_fileno, parameter_name, crop, to_crop=False, show_images=False)
        t, sigma_x, sigma_y = self.get_cloud_size_den(file_start, file_end, bg_fileno, variable=parameter_name)
        
        return t, n_container/(sigma_x * np.sqrt(sigma_y)), n_err_container/(sigma_x * np.sqrt(sigma_y))

    def tweezer_inject(self, zipNo):
        images_list = [f for f in os.listdir(tweezer_img_source_dir) if f.endswith('.tif')]
        zipPathIMG = self.dirPath + '\\' + self.fileNameString + '_' + str(zipNo).zfill(3) + '.zip'
        for img_name in images_list:
            image_path = tweezer_img_source_dir + img_name
            archive =  ZipFile(zipPathIMG, 'a')
            archive.write(image_path, os.path.basename(image_path))
            archive.close()
            sleep(0.1)
            os.remove(image_path)

    def tweezer_inject_double(self, zipNo):
        images_list = [f for f in os.listdir(tweezer_img_source_dir) if f.endswith('.tif')]
        zipPathIMG = self.dirPath + '\\' + self.fileNameString + '_' + str(zipNo).zfill(3) + '.zip'
        zipPathBG = self.dirPath + '\\' + self.fileNameString + '_' + str(zipNo+1).zfill(3) + '.zip'
        shutil.copyfile(zipPathIMG, zipPathBG)
        for img_name in images_list:
            image_no = int(img_name[15:-4])
            image_path = tweezer_img_source_dir + img_name
            if image_no % 2 == 0:
                archive =  ZipFile(zipPathBG, 'a')
                archive.write(image_path, os.path.basename(image_path))
                archive.close()
            else:
                archive =  ZipFile(zipPathIMG, 'a')
                archive.write(image_path, os.path.basename(image_path))
                archive.close()
            sleep(0.1)
            os.remove(image_path)

    def tweezer_show_bg_subtracted(self, zipNo, reg=[25, 31, 22, 28], cmap='coolwarm', show=True, vmaxfactor=0.8, show_grid=True):
        images = self.read_images_from_zip(zipNo)
        backgrounds = self.read_images_from_zip(zipNo + 1)
        bg_sub_img = images - backgrounds.mean(axis=0)
        img_average = bg_sub_img.mean(axis=0)
        trap_average = img_average[reg[0]:reg[1], reg[2]:reg[3]]
        vmin = img_average.min()
        vmax = vmaxfactor*img_average.max()
        if show:
            fig, ax = plt.subplots(1, 2, figsize=(10, 20))
            ax[0].imshow(img_average, cmap=cmap, vmin=vmin, vmax=vmax)
            ax[1].imshow(trap_average, cmap=cmap, vmin=vmin, vmax=vmax)
            if not show_grid:
                ax[0].grid()
                ax[1].grid()
        return img_average
    
    def tweezer_show(self, zipNo, reg=[25, 31, 22, 28], cmap='coolwarm', show=True, vmaxfactor=0.8, show_grid=True):
        images = self.read_images_from_zip(zipNo)
        img_average = images.mean(axis=0)
        trap_average = img_average[reg[0]:reg[1], reg[2]:reg[3]]
        vmin = img_average.min()
        vmax = vmaxfactor*img_average.max()
        if show:
            fig, ax = plt.subplots(1, 2, figsize=(10, 20))
            ax[0].imshow(img_average, cmap=cmap, vmin=vmin, vmax=vmax)
            ax[1].imshow(trap_average, cmap=cmap, vmin=vmin, vmax=vmax)
            if not show_grid:
                ax[0].grid()
                ax[1].grid()
        return img_average
    
    def get_array_loading_probability(self, imgFileNo, grid_positions, grid_shape, threshold=6.85, window_size=5, binning=20, show_histogram=True):
        n_row, n_col = grid_shape
        a500 = 0.00483372
        b500 = 1828.38
        c500 = 13
        images = self.read_images_from_zip(imgFileNo)
        photon_rates, count_rates, atom_counter = [], [], np.zeros(grid_shape)
        for image in images:
            counts = sum_pixel_values(image, grid_positions, [n_row, n_col], window_size=window_size)
            electrons = (counts - b500) * a500
            photons = electrons/0.7
            photon_rate = (photons/80e-3)/1000
            photon_rates.append(photon_rate)
        if threshold == 0:
            threshold = detect_loading_threshold(photon_rates)
        for image in images:
            counts = sum_pixel_values(image, grid_positions, [n_row, n_col], window_size=window_size)
            electrons = (counts - b500) * a500
            photons = electrons/0.7
            photon_rate = (photons/80e-3)/1000
            atoms = np.zeros_like(photon_rate)
            atoms[photon_rate > threshold] = 1
            photon_rates.append(photon_rate)
            count_rates.append(photon_rate*0.7)
            atom_counter += atoms
        loading_probabilities = atom_counter / len(images)
        if show_histogram:
            fig, ax = plt.subplots(1, 3, figsize = (21,5))
            for n in range(n_row):
                for m in range(n_col):
                    ax[0].hist(np.array(photon_rates)[:, n, m], bins=binning, alpha=0.6)
                    ax[0].set_xlabel('Photon Rate / kHz')
                    ax[0].set_ylabel('Measurements')
                    print(f'Trap ({n}, {m}) Loading Probability : {loading_probabilities[n, m]*100} %')
            ax[1].hist(np.array(photon_rates).ravel(), bins=binning, alpha=0.8)
            ax[1].set_xlabel('Photon Rate / kHz')
            ax[1].set_ylabel('Measurements')
            cax = ax[2].matshow(loading_probabilities, cmap='viridis', vmin=0.3)
            fig.colorbar(cax, ax=ax[2])
        return photon_rates, loading_probabilities, threshold
    
    def get_array_loading_probability_general(self, imgFileNo, grid_positions, threshold=6.85, window_size=5, binning=20):
        a500 = 0.00483372
        b500 = 1828.38
        c500 = 13
        images = self.read_images_from_zip(imgFileNo)
        photon_rates, count_rates, atom_counter = [], [], np.zeros(len(grid_positions))
        for image in images:
            counts = sum_pixel_values_general(image, grid_positions, window_size=window_size)
            electrons = (counts - b500) * a500
            photons = electrons/0.7
            photon_rate = (photons/80e-3)/1000
            photon_rates.append(photon_rate)
        if threshold == 0:
            threshold = detect_loading_threshold(photon_rates)
        for image in images:
            counts = sum_pixel_values_general(image, grid_positions, window_size=window_size)
            electrons = (counts - b500) * a500
            photons = electrons/0.7
            photon_rate = (photons/80e-3)/1000
            atoms = np.zeros_like(photon_rate)
            atoms[photon_rate > threshold] = 1
            photon_rates.append(photon_rate)
            count_rates.append(photon_rate*0.7)
            atom_counter += atoms
        loading_probabilities = atom_counter / len(images)
        fig, ax = plt.subplots(1, 3, figsize = (21,5))
        for i in range(len(grid_positions)):
            ax[0].hist(np.array(photon_rates)[:, i], bins=binning, alpha=0.6)
            ax[0].set_xlabel('Photon Rate / kHz')
            ax[0].set_ylabel('Measurements')
            print(f'Trap ({i}) Loading Probability : {loading_probabilities[i]*100} %')
        ax[1].hist(np.array(photon_rates).ravel(), bins=binning, alpha=0.6)
        ax[1].set_xlabel('Photon Rate / kHz')
        ax[1].set_ylabel('Measurements')
        ax[2].bar(grid_positions.keys(), loading_probabilities)
        ax[2].set_xlabel('Trap Site')
        ax[2].set_ylabel('Loading Probability')
        print(f'Overall Loading Probability = {loading_probabilities.mean()*100} %')
        return photon_rates, loading_probabilities
    
    def extract_survival_probabilities(self, startZip, num_datapoints, grid_positions, grid_shape=[8,8], trap_size=3, loading_threshold=6):
        individual_survival_prob_list = []
        survival_prob_list = []
        survival_prob_err_list = []
        
        for zipNo in track(np.arange(startZip, startZip + 2*num_datapoints, 2)):
            photon_rates_1, _, threshold_1 = self.get_array_loading_probability(zipNo, grid_positions, grid_shape, threshold=loading_threshold, window_size=trap_size, binning=20, show_histogram=False)
            photon_rates_2, _, threshold_2 = self.get_array_loading_probability(zipNo + 1, grid_positions, grid_shape, threshold=threshold_1, window_size=trap_size, binning=20, show_histogram=False)
        
            photon_rates_1 = np.array(photon_rates_1)
            photon_rates_2 = np.array(photon_rates_2)
        
            survival_probabilities = np.zeros((grid_shape[0],grid_shape[1]))
        
            for i, j in np.ndindex(grid_shape[0], grid_shape[1]):
                trap_photon_rates_1 = photon_rates_1[:, i, j]
                trap_photon_rates_2 = photon_rates_2[:, i, j]
                
                trap_photon_rates_2_postselected = trap_photon_rates_2[trap_photon_rates_1 > threshold_1]
                if len(trap_photon_rates_2_postselected) != 0:
                    survival_probabilities[i, j] = len(trap_photon_rates_2_postselected[trap_photon_rates_2_postselected > threshold_2]) / len(trap_photon_rates_2_postselected)
                else:
                    survival_probabilities[i, j] = np.nan
            survival_probabilities = survival_probabilities.ravel()
            survival_probabilities = survival_probabilities[~np.isnan(survival_probabilities)]
            
            individual_survival_prob_list.append(survival_probabilities)
            survival_prob_list.append(survival_probabilities.mean())
            survival_prob_err_list.append(survival_probabilities.std())

        return survival_prob_list, survival_prob_err_list, individual_survival_prob_list
    
    def array_baseline_measurement(self, zipNo, grid_shape = [8, 8], trap_size = 3, detection_step=100):
        img_array = self.tweezer_show_bg_subtracted(zipNo, reg=[0, -1, 0, -1], show=False, vmaxfactor=0.6, cmap='gray', show_grid=False)
        grid_positions, detection_threshold = detect_trap_sites(img_array, grid_shape, detection_step=detection_step)
        visualize_results(img_array, grid_positions, margin=20, window_size=trap_size, threshold=detection_threshold)
        photon_rates, loading_probabilities, threshold = self.get_array_loading_probability(zipNo, grid_positions, grid_shape, threshold=0, window_size=trap_size, binning=20)
        cvar = np.std(loading_probabilities) / loading_probabilities.mean()
        print(f"Loading Threshold = {threshold:.6f} kHz")
        print(f"Standard Deviation = {cvar*100} %")
        print(f"Average Loading Prob = {loading_probabilities.mean()*100} %")
        return grid_positions, photon_rates, loading_probabilities, threshold
    
    def array_loading_threshold_measurement(self, zipNo, powers, datapoints, grid_positions, grid_shape, threshold, trap_size=3):
        ld_probs = []
        ld_probs_err = []
        for z in np.arange(zipNo, zipNo + datapoints):
            photon_rates, ld_prob, _ = self.get_array_loading_probability(z, grid_positions, grid_shape, threshold=threshold, window_size=trap_size, show_histogram=False)
            N = np.array(photon_rates).size

            p = ld_prob.mean()
            ld_probs.append(p)
            ld_probs_err.append(np.sqrt(p * (1 - p) / N))
        
        p0 = [0.55, 0.5, 0.2] 

        # Bounds: Amp [0, 1], Pc > 0, P_offset usually >= 0
        bounds = ([0, 0, 0], [1.0, 5.0, 1.0])
        params, pcov = curve_fit(maxwell_boltzmann_cdf, powers, ld_probs, p0=p0, bounds=bounds)

        xx = np.linspace(0, powers.max(), 1000)
        yy = maxwell_boltzmann_cdf(xx, *params)

        power_thresh = params[1]+ params[2]

        plt.errorbar(powers, np.array(ld_probs), yerr=np.array(ld_probs_err), fmt='o', color='C0')
        plt.plot(xx, yy, label=f'Loading Threshold = {power_thresh:.4f}')
        plt.legend()
        plt.xlabel('Tweezer Power')
        plt.ylabel('Loading Probability')