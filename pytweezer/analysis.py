import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import center_of_mass, label, gaussian_filter, white_tophat
import matplotlib.pyplot as plt
from time import sleep
import matplotlib.patches as patches
from rich.progress import track
from scipy.special import erf
from sklearn.mixture import GaussianMixture
from scipy import integrate
from scipy.stats import norm
from pytweezer.experiment.experiment_parameter_manager import ExpParameterManager
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment



####################################################################################################

# Fundamental Constants
motcam_mm_per_px = 1e3 / 194902.8581 # mm / px
hamcam_px_per_um = 2.0714285714285716
hamcam_um_per_px = 1 / hamcam_px_per_um


MRb = 1.44316e-25  # kg
kB = 1.380649e-23  # J/K
c = 299792458
e0 = 8.854e-12
e = 1.6e-19
me = 9.109e-31
hbar = 1.05e-34

# D2 Line
wD2 = 2 * np.pi * 384.230484e12
muD2 = 2.069e-29
gammaD2 = 2*np.pi*6.065*1e6

# D1 Line
wD1 = 2 * np.pi * 377.107463e12
muD1 = 1.46e-29
gammaD1 = 2*np.pi*5.746*1e6

# System Parameters
wavelen = 852e-9 # m
w = 2 * np.pi * c / (wavelen)

####################################################################################################

exp_params = ExpParameterManager()
cool_vco_resonance = exp_params.get_parameter("cool_vco_resonance")

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

def pair_coordinates(x_pos, y_pos, u_pos, v_pos):
    # Convert the inputs into 2D numpy arrays of coordinates
    # shape will be (N, 2)
    pts1 = np.column_stack((x_pos, y_pos))
    pts2 = np.column_stack((u_pos, v_pos))
    
    # Calculate the distance matrix between all points in pts1 and pts2
    # distance_matrix[i, j] is the distance between pts1[i] and pts2[j]
    distance_matrix = cdist(pts1, pts2)
    
    # Use the Hungarian algorithm to find the optimal 1-to-1 matching.
    row_indices, col_indices = linear_sum_assignment(distance_matrix)
    
    # Reorder the u and v arrays based on the optimal matching
    u_pos_sorted = np.array(u_pos)[col_indices]
    v_pos_sorted = np.array(v_pos)[col_indices]
    
    return u_pos_sorted, v_pos_sorted

def set_array_centre(grid_positions_img):
    exp_params = ExpParameterManager()
    y0 = np.mean([pos[0] for pos in grid_positions_img.values()])
    x0 = np.mean([pos[1] for pos in grid_positions_img.values()])
    exp_params.set_parameter("x_centre", x0)
    exp_params.set_parameter("y_centre", y0)
    exp_params.save_parameters()

def index_grid_positions(grid_positions, x_n, y_n):
    exp_params = ExpParameterManager()
    y_img = np.array([pos[0] for pos in grid_positions.values()])
    x_img = np.array([pos[1] for pos in grid_positions.values()])
    y_0 = exp_params.get_parameter("y_centre")
    x_0 = exp_params.get_parameter("x_centre")
    y_pos = (y_img - y_0) * hamcam_um_per_px
    x_pos = (x_img - x_0) * -hamcam_um_per_px
    x_n_rot, y_n_rot = rotate_coordinates(x_n, y_n, angle_deg=-2.5)
    x_pos_sorted, y_pos_sorted = pair_coordinates(x_n_rot, y_n_rot, x_pos, y_pos)
    x_img_sorted = (x_pos_sorted * -hamcam_px_per_um + x_0).astype('int64')
    y_img_sorted = (y_pos_sorted * hamcam_px_per_um + y_0).astype('int64')
    grid_positions_sorted = dict(enumerate(zip(y_img_sorted, x_img_sorted)))
    return grid_positions_sorted

def convert_vco_detuning(vco_array):
    detuning_array = (vco_array - cool_vco_resonance) * 12.24 # Convert to MHz
    return detuning_array

def gaussian2D(x, y, A, x0, y0, sx, sy, theta, offset):
    x_rot = (x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)
    y_rot = -(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)
    return A * np.exp(-((x_rot**2 / (2 * sx**2)) + (y_rot**2 / (2 * sy**2)))) + offset

def fit_gaussian(img):
    x = np.arange(img.shape[1])
    y = np.arange(img.shape[0])
    x, y = np.meshgrid(x, y)
    com_y, com_x = center_of_mass(img)
    p0 = (img.max() - img.min(), com_x, com_y, 70, 70, 0, img.min())  # Initial guess for parameters
    popt, pcov = curve_fit(lambda xy, A, x0, y0, sx, sy, theta, offset: gaussian2D(xy[0], xy[1], A, x0, y0, sx, sy, theta, offset).ravel(), (x.ravel(), y.ravel()), img.ravel(), p0=p0)
    A_fit, x0_fit, y0_fit, sx_fit, sy_fit, theta_fit, offset_fit = popt
    sx_fit, sy_fit = sorted([abs(sx_fit), abs(sy_fit)], reverse=True)
    return {
        "A": A_fit, "x0": x0_fit, "y0": y0_fit, "sx": abs(sx_fit), "sy": abs(sy_fit), "theta": theta_fit, "offset": offset_fit
    }

def get_total_counts(img, x, y, window_size):
    half_window = window_size // 2
    x_min = max(x - half_window, 0)
    x_max = min(x + half_window + 1, img.shape[1])
    y_min = max(y - half_window, 0)
    y_max = min(y + half_window + 1, img.shape[0])
    return np.sum(img[y_min:y_max, x_min:x_max])

def detect_bright_points(image_array, threshold=200):
    # Step 1: Convert image to binary (bright spots = True, background = False)
    binary_image = image_array > threshold  # Apply threshold to create a binary mask

    # Step 2: Label connected components (each bright spot gets a unique label)
    labeled_array, num_features = label(binary_image)

    # Step 3: Find the centre of mass in each region
    centers = []
    for region_id in range(1, num_features + 1):
        # Calculate the centre of mass for this region
        com = center_of_mass(image_array, labeled_array, region_id)
        
        # Append the centre of mass coordinates (y, x) as a tuple
        centers.append(com)
    
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

def detect_trap_sites_grid(img_array, grid_shape, detection_step = 100):
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

def detect_trap_sites(img_array, atom_number, detection_step=100):
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

def sum_pixel_values(image_array, grid_positions, window_size=10):
    half_size = window_size // 2
    pixel_sums = np.zeros(len(grid_positions), dtype=int)  # Create empty 2D array

    for (i),(y, x) in grid_positions.items():
        # Extract 5x5 region and sum pixel values
        region = image_array[max(y-half_size, 0):min(y+half_size+1, image_array.shape[0]),
                             max(x-half_size, 0):min(x+half_size+1, image_array.shape[1])]
        pixel_sums[i] = np.sum(region)
    return pixel_sums

def extract_crops(images, grid_positions, window_size=9):
    """Window around every site in every frame: (n_frames, n_sites, w, w), keys."""
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 2:
        images = images[None]
    keys = sorted(grid_positions)
    half = window_size // 2
    rows = np.array([grid_positions[k][0] for k in keys])
    cols = np.array([grid_positions[k][1] for k in keys])
    offsets = np.arange(-half, half + 1)
    row_index = rows[:, None, None] + offsets[None, :, None]
    col_index = cols[:, None, None] + offsets[None, None, :]

    height, width = images.shape[1:]
    if row_index.min() < 0 or col_index.min() < 0 or \
            row_index.max() >= height or col_index.max() >= width:
        raise ValueError(f"window_size={window_size} runs off the {height}x{width} frame.")
    return images[:, row_index, col_index], keys


def split_threshold(values, max_iter=50):
    """Two-means split of a bimodal distribution."""
    cut = np.median(values)
    for _ in range(max_iter):
        low, high = values[values <= cut], values[values > cut]
        if low.size == 0 or high.size == 0:
            break
        new_cut = 0.5 * (low.mean() + high.mean())
        if np.isclose(new_cut, cut):
            break
        cut = new_cut
    return float(cut)


def build_psf_templates(images, grid_positions, window_size=9, min_samples=10):
    """Per-site PSF template: mean occupied crop minus mean empty crop, sum-normalised.

    Occupancy is called per site, so a dim site is not dragged below an array-wide
    cut. Sites with too few frames either way fall back to the mean template.
    """
    crops, keys = extract_crops(images, grid_positions, window_size)
    box_scores = crops.sum(axis=(2, 3))

    templates, counts = {}, {}
    for index, site in enumerate(keys):
        scores = box_scores[:, index]
        occupied = scores > split_threshold(scores)
        counts[site] = (int(occupied.sum()), int((~occupied).sum()))
        if min(counts[site]) < min_samples:
            templates[site] = None
            continue
        psf = np.clip(crops[occupied, index].mean(axis=0)
                      - crops[~occupied, index].mean(axis=0), 0, None)
        templates[site] = (psf / psf.sum()).astype(np.float32) if psf.sum() > 0 else None

    usable = [t for t in templates.values() if t is not None]
    if not usable:
        raise ValueError(f"No site had {min_samples} clear frames out of {len(crops)}.")
    fallback = np.mean(usable, axis=0)
    fallback = (fallback / fallback.sum()).astype(np.float32)
    missing = [s for s, t in templates.items() if t is None]
    if missing:
        print(f"{len(missing)} site(s) used the mean template: {missing}")
    return {s: (fallback if t is None else t) for s, t in templates.items()}


class SiteScorer:
    """Per-site photon rates on a fixed grid.

    ``method="box"``  top-hat, then sum each site's window - the original.
    ``method="psf"``  weight each window by that site's template. Weights are
    mean-subtracted, so flat background scores zero and no top-hat is needed.
    """

    def __init__(self, grid_positions, method="box", window_size=None,
                 templates=None, feature_size=10, threshold=None):
        if method not in ("box", "psf"):
            raise ValueError(f"method must be 'box' or 'psf', got {method!r}")
        if method == "psf" and not templates:
            raise ValueError("method='psf' needs templates; see build_psf_templates.")

        self.method = method
        self.grid_positions = dict(grid_positions)
        self.templates = templates
        self.feature_size = feature_size
        self.threshold = threshold
        self.keys = sorted(self.grid_positions)

        if window_size is None:
            window_size = next(iter(templates.values())).shape[0] if method == "psf" else 5
        self.window_size = int(window_size)

        half = self.window_size // 2
        offsets = np.arange(-half, half + 1)
        rows = np.array([self.grid_positions[k][0] for k in self.keys])
        cols = np.array([self.grid_positions[k][1] for k in self.keys])
        self._rows = rows[:, None, None] + offsets[None, :, None]
        self._cols = cols[:, None, None] + offsets[None, None, :]

        # Cached: the hot path should not re-read these per frame.
        exp_params = ExpParameterManager()
        self._offset = exp_params.get_parameter("conversion_offset")
        self._factor = exp_params.get_parameter("conversion_factor")

        if method == "psf":
            stack = np.stack([templates[k] for k in self.keys]).astype(np.float32)
            weights = stack - stack.mean(axis=(1, 2), keepdims=True)
            self._weights = weights / np.einsum("swh,swh->s", weights, stack)[:, None, None]

    @property
    def name(self):
        return "PSF" if self.method == "psf" else "box sum"

    @classmethod
    def from_images(cls, images, grid_positions, window_size=9,
                    min_samples=10, **kwargs):
        """Build PSF templates from a stack, then a scorer that uses them."""
        templates = build_psf_templates(images, grid_positions, window_size, min_samples)
        return cls(grid_positions, method="psf", window_size=window_size,
                   templates=templates, **kwargs)

    def site_scores(self, image):
        """One photon rate per site, in ``self.keys`` order."""
        if self.method == "box":
            if self.feature_size:
                image = white_tophat(image, size=self.feature_size)
            counts = np.asarray(image, dtype=np.float32)[self._rows, self._cols].sum(axis=(1, 2))
        else:
            crops = np.asarray(image, dtype=np.float32)[self._rows, self._cols]
            counts = np.einsum("swh,swh->s", crops, self._weights)
        return (counts - self._offset) * self._factor / 0.7

    def score_stack(self, images):
        return np.stack([self.site_scores(image) for image in images])

    def occupancy(self, image, threshold=None):
        """Flat boolean mask in trap order, for the rearrangement coordinator.

        ``threshold`` is in photons - takes ``threshold_1`` from
        :func:`get_array_loading_statistics` directly.
        """
        cut = self.threshold if threshold is None else threshold
        if cut is None:
            raise ValueError(f"{self.name} scorer has no threshold; calibrate it first.")
        return self.site_scores(image) > cut


# Function to visualize results with cropping and zooming
def visualize_results(image_array, grid_positions, margin=50, window_size=5, threshold=150, vmaxfactor=0.8, index_labels=False, bin_sharpness=20, bin_thresh_factor=0.8):
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
    cropped_bin_image = (cropped_image - image_array.min()) / (image_array.max() - image_array.min())  
    threshold_bin = (threshold - image_array.min()) / (image_array.max() - image_array.min())

    # Apply a sigmoid function to binarize the image, change parameters to adjust the threshold and sharpness of the sigmoid
    sigmoid = lambda x, a, b: 1 / (1 + np.exp(-a * (x - b)))
    img_bin_sigmoid = sigmoid(cropped_bin_image, bin_sharpness, bin_thresh_factor*threshold_bin)

    # Adjust positions of grid labels for cropped view
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(cropped_image, cmap="gray", extent=[x1, x2, y2, y1], vmax=vmaxfactor*cropped_image.max())
    ax[0].grid(False)
    ax[0].set_title("Raw Image")
    ax[1].imshow(cropped_image, cmap="gray", extent=[x1, x2, y2, y1], vmax=vmaxfactor*cropped_image.max())  # Use extent to maintain coordinates
    ax[1].set_title("Trap Site Detection")
    ax[2].imshow(img_bin_sigmoid, cmap='hot', extent=[x1, x2, y2, y1])
    ax[2].set_title("Binarized Image (Sigmoid)")

    # Draw 5x5 squares and labels
    half_size = window_size // 2
    for (i), (y, x) in grid_positions.items():
        # Draw grid label
        if index_labels:
            ax[1].text(x+5, y, f"({i},{j})", color='white', fontsize=6, weight='bold')

        # Draw a 5x5 square centered on (x, y)
        rect0 = patches.Rectangle((x - half_size, y - half_size), window_size, window_size,
                                 linewidth=1, edgecolor='red', facecolor='none')
        rect1 = patches.Rectangle((x - half_size, y - half_size), window_size, window_size,
                                 linewidth=1, edgecolor='red', facecolor='none')
        ax[1].add_patch(rect0)
    plt.show()

def detect_loading_threshold(counts):
    counts_2d = counts.reshape(-1, 1)
    # Initialize and fit the GMM for 2 components
    gmm = GaussianMixture(n_components=2, covariance_type='spherical')
    gmm.fit(counts_2d)

    # Extract the fitted parameters
    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten() # We need these for accurate plotting!

    # Identify which mean is the background and which is the signal
    bg_idx = np.argmin(means)
    sig_idx = np.argmax(means)

    mu_bg, var_bg = means[bg_idx], variances[bg_idx]
    mu_sig, var_sig = means[sig_idx], variances[sig_idx]
    weight_bg, weight_sig = weights[bg_idx], weights[sig_idx]

    # Calculate the intersection point of the two Gaussians (the threshold)
    a = 1/(2*var_bg) - 1/(2*var_sig)
    b = mu_sig/var_sig - mu_bg/var_bg
    c = mu_bg**2 / (2*var_bg) - mu_sig**2 / (2*var_sig) - np.log(np.sqrt(var_sig/var_bg))

    # The quadratic formula yields two roots; we want the one between the means
    roots = np.roots([a, b, c])
    threshold = [r for r in roots if mu_bg < r < mu_sig][0]
    
    return threshold, [mu_bg, var_bg, weight_bg], [mu_sig, var_sig, weight_sig]

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

def gaussian_high_pass(image, sigma_blur):
    """
    Standard linear high-pass filter.
    Subtracts a Gaussian-blurred version of the image from itself.
    """
    # Create the low-pass background
    low_pass = gaussian_filter(image, sigma=sigma_blur)
    
    # Subtract to get the high-pass, clipping at 0 to prevent negative counts
    high_pass = image - low_pass
    return np.clip(high_pass, 0, None)

def morphological_tophat_high_pass(image, feature_size):
    """
    Non-linear high-pass filter (Recommended for Tweezer Arrays).
    Extracts bright features smaller than the feature_size.
    """
    # white_tophat requires a footprint (kernel size). 
    # This should be larger than your atom spot, but smaller than the background glow.
    return white_tophat(image, size=feature_size)

def extract_cloud_temperature(images, tau_list, show_plots=True):
    x0_list, y0_list, sx_list, sy_list = [], [], [], []
    for it, img in enumerate(images):
        gaussian_params = fit_gaussian(img)
        amp, centre_x, centre_y, sx, sy = gaussian_params["A"], int(gaussian_params["x0"]), int(gaussian_params["y0"]), gaussian_params["sx"], gaussian_params["sy"]
        print(f"Iteration {str(it+1).zfill(2)}/{len(tau_list)}  |  TOF: {tau_list[it]/100:.6g} ms  |  Amplitude: {amp:.6g}  |  Centre: ({centre_x}, {centre_y})  |  Widths: (sx: {sx:.6g}, sy: {sy:.6g})")
        x0_list.append(centre_x)
        y0_list.append(centre_y)
        sx_list.append(sx)
        sy_list.append(sy)

    tau_list_ms = tau_list * 1 / 100
    sx_array = np.array(sx_list) * motcam_mm_per_px
    sy_array = np.array(sy_list) * motcam_mm_per_px
    x0_array = np.array(x0_list) * motcam_mm_per_px
    y0_array = np.array(y0_list) * motcam_mm_per_px

    lin_fit = lambda x, m, c: m * x + c
    px, pcovx = curve_fit(lin_fit, tau_list_ms**2, sx_array**2)
    py, pcovy = curve_fit(lin_fit, tau_list_ms**2, sy_array**2)
    t2_fit = np.linspace(0, tau_list_ms[-1]**2, 100)
    sx2_fit = lin_fit(t2_fit, *px)
    sy2_fit = lin_fit(t2_fit, *py)
    mx = px[0]  # slope for sx^2 vs tau^2
    my = py[0]  # slope for sy^2 vs tau^2
    Tx = mx * MRb / kB * 1e6
    Ty = my * MRb / kB * 1e6
    mx_err = np.sqrt(np.diag(pcovx))[0]
    my_err = np.sqrt(np.diag(pcovy))[0]
    Tx_err = mx_err * MRb / kB * 1e6
    Ty_err = my_err * MRb / kB * 1e6

    free_fall = lambda t, a, x0: 1/2 * a * t**2 + x0
    param1, pcov1 = curve_fit(free_fall, tau_list_ms, x0_array)
    t_fit = np.linspace(0, tau_list_ms[-1], 100)
    x0_fit = free_fall(t_fit, *param1)
    a = param1[0] # acceleration in pixels/ms²
    a_err = np.sqrt(np.diag(pcov1))[0]

    if show_plots:
        # Plot sx_array, sy_array, x0_array all three side by side with respect to tau_list_ms
        plt.figure(figsize=(10, 3))
        plt.subplot(1, 3, 1)
        plt.scatter(tau_list_ms**2, sx_array**2)
        plt.plot(t2_fit, sx2_fit, color='red', label=f"Tx = {Tx:.2f} ± {Tx_err:.2f} μK")
        plt.legend()
        plt.xlabel('τ² (ms²)')
        plt.ylabel('σx² (mm²)')
        plt.title('σx² vs τ²')

        plt.subplot(1, 3, 2)
        plt.scatter(tau_list_ms**2, sy_array**2)
        plt.plot(t2_fit, sy2_fit, color='red', label=f"Ty = {Ty:.2f} ± {Ty_err:.2f} μK")
        plt.legend()
        plt.xlabel('τ² (ms²)')
        plt.ylabel('σy² (mm²)')
        plt.title('σy² vs τ²')

        plt.subplot(1, 3, 3)
        plt.scatter(tau_list_ms, x0_array)
        plt.plot(t_fit, x0_fit, color='red', label=f"a = {a/(9.81/1000):.2f} ± {a_err/(9.81/1000):.2f} g")
        plt.legend()
        plt.xlabel('τ (ms)')
        plt.ylabel('x0 (mm)')
        plt.title('x0 vs τ')

        plt.tight_layout()
        plt.show()

    return {
        "Tx": Tx, "Tx_err": Tx_err, "Ty": Ty, "Ty_err": Ty_err, "a": a, "a_err": a_err
    }

def tweezer_show_bg_subtracted(images, backgrounds, cmap='gray', show=True, vmaxfactor=0.8, show_grid=True):
    images = np.array(images)
    backgrounds = np.array(backgrounds)
    bg_sub_img = images - backgrounds.mean(axis=0)
    img_average = bg_sub_img.mean(axis=0)
    vmin = img_average.min()
    vmax = vmaxfactor*img_average.max()
    if show:
        plt.imshow(img_average, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar()
        if not show_grid:
            plt.grid()
    return img_average

def convert_photons_to_counts(photons):
    exp_params = ExpParameterManager()
    conversion_factor = exp_params.get_parameter("conversion_factor")
    conversion_offset = exp_params.get_parameter("conversion_offset")
    counts_array = photons * 0.7 / conversion_factor + conversion_offset
    return counts_array

def convert_counts_to_photons(counts):
    exp_params = ExpParameterManager()
    conversion_factor = exp_params.get_parameter("conversion_factor")
    conversion_offset = exp_params.get_parameter("conversion_offset")
    photons_array = (counts - conversion_offset) * conversion_factor / 0.7
    return photons_array

def get_array_loading_statistics_grid(images, grid_positions, grid_shape, threshold=1.0, window_size=5, binning=20, show_histogram=True, threshold_detection=True, verbose=True, method="box", scorer=None, psf_window=9):
    """Loading statistics per site, scored by box sum or PSF matched filter.

    ``method="box"`` expects top-hat filtered ``images``; ``method="psf"`` builds a
    template per site and takes the raw frames. Pass ``scorer`` to reuse templates
    built elsewhere.
    """
    n_row, n_col = grid_shape
    exp_params = ExpParameterManager()
    conversion_factor = exp_params.get_parameter("conversion_factor")
    conversion_offset = exp_params.get_parameter("conversion_offset")

    if scorer is None and method == "psf":
        scorer = SiteScorer.from_images(images, grid_positions, grid_shape,
                                        window_size=psf_window)

    # Extract photon counts for each image and each trap site
    if scorer is not None:
        photon_array = scorer.score_stack(images)   # SiteScorer already returns photons
    else:
        raw_counts = np.array([sum_pixel_values(image, grid_positions, grid_shape, window_size=window_size) for image in images])
        photon_array = (raw_counts - conversion_offset) * conversion_factor / 0.7
    tot_photon_array = photon_array.flatten()

    # Threshold detection and fidelity calculation
    threshold_detection_success = False
    if threshold_detection:
        try:
            threshold, bg_params, sig_params = detect_loading_threshold(tot_photon_array)
            mu_bg, var_bg, weight_bg  = bg_params
            mu_sig, var_sig, weight_sig = sig_params
            prob_false_negative = norm.cdf(threshold, loc=mu_sig, scale=np.sqrt(var_sig))
            prob_false_positive = 1.0 - norm.cdf(threshold, loc=mu_bg, scale=np.sqrt(var_bg))
            total_error = (weight_bg * prob_false_positive) + (weight_sig * prob_false_negative)
            fidelity = 1.0 - total_error
            threshold_detection_success = True
        except Exception as e:
            print(f"Threshold detection failed: {e}. Trying default threshold of {threshold:.2f} kHz.")
            try:
                mu_bg = tot_photon_array[tot_photon_array < threshold].mean()
                mu_sig = tot_photon_array[tot_photon_array >= threshold].mean()
                var_bg = tot_photon_array[tot_photon_array < threshold].var()
                var_sig = tot_photon_array[tot_photon_array >= threshold].var()
                fidelity = norm.cdf((threshold - mu_sig) / np.sqrt(var_sig)) + (1 - norm.cdf((threshold - mu_bg) / np.sqrt(var_bg)))
            except Exception as e:
                print(f"Fallback threshold calculation failed: {e}. Setting fidelity to 0.")
                fidelity = np.nan      
    else:
        try:
            mu_bg = tot_photon_array[tot_photon_array < threshold].mean()
            mu_sig = tot_photon_array[tot_photon_array >= threshold].mean()
            var_bg = tot_photon_array[tot_photon_array < threshold].var()
            var_sig = tot_photon_array[tot_photon_array >= threshold].var()
            fidelity = norm.cdf((threshold - mu_sig) / np.sqrt(var_sig)) + (1 - norm.cdf((threshold - mu_bg) / np.sqrt(var_bg)))
        except Exception as e:
            print(f"Fallback threshold calculation failed: {e}. Setting fidelity to 0.")
            fidelity = np.nan

    # Atom counting and loading probabilities
    atom_counter = (photon_array > threshold).astype(int).sum(axis=0)
    loading_probabilities = atom_counter / len(images)
    
    if show_histogram:
        fig, ax = plt.subplots(1, 3, figsize = (16,5), constrained_layout=True)
        for n in range(n_row):
            for m in range(n_col):
                ax[0].hist(photon_array[:, n, m], bins=40, density=True, alpha=0.6, range=(min(tot_photon_array), max(tot_photon_array)))
                ax[0].set_xlabel('Photons')
                ax[0].set_ylabel('Probability Density')
                if verbose:
                    print(f'Trap ({n}, {m}) Loading Probability : {loading_probabilities[n, m]*100:.2f} %')
        if threshold_detection and threshold_detection_success:
            x_fit = np.linspace(min(tot_photon_array), max(tot_photon_array), 1000)
            pdf_bg = weight_bg * norm.pdf(x_fit, mu_bg, np.sqrt(var_bg))
            pdf_sig = weight_sig * norm.pdf(x_fit, mu_sig, np.sqrt(var_sig))
            ax[1].plot(x_fit, pdf_bg + pdf_sig, 'k-', lw=2,)
            ax[1].plot(x_fit, pdf_bg, 'b--', lw=2, label=f'Background Fit ($\mu$={mu_bg:.1f})')
            ax[1].plot(x_fit, pdf_sig, 'r--', lw=2, label=f'Signal Fit ($\mu$={mu_sig:.1f})')
        ax[1].hist(tot_photon_array, bins=binning, density=True, alpha=0.5, color='gray', edgecolor='black', label='Raw Data')
        ax[1].axvline(x=threshold, color='green', linestyle='--', lw=2, label=f'Threshold ({threshold:.1f})')
        ax[1].set_xlabel('Photons')
        ax[1].set_ylabel('Probability Density')
        ax[1].legend()
        cax = ax[2].matshow(loading_probabilities, cmap='viridis', vmin=0.3)
        fig.colorbar(cax, ax=ax[2])

    if verbose:
        print(f"Detected Loading Threshold: {threshold:.2f} photons")
        print(f"Std dev of Loading Probabilities: {np.std(loading_probabilities) / loading_probabilities.mean()*100:.2f} %")
        print(f"Mean Loading Probability: {loading_probabilities.mean()*100:.2f} %")
        if fidelity is not np.nan:
            print(f"Detection Fidelity: {fidelity*100:.2f} %")
            print(f"Mean Background Photon Count: {mu_bg:.2f} photons, Variance: {var_bg:.2f} photons^2")
            print(f"Mean Signal Photon Count: {mu_sig:.2f} photons, Variance: {var_sig:.2f} photons^2")

    return photon_array, loading_probabilities, threshold, fidelity

def get_array_loading_statistics(images, grid_positions, threshold=1.0, window_size=5, binning=20, show_histogram=True, threshold_detection=True, verbose=True, method="box", scorer=None, psf_window=9, show_site_labels=False):
    """Loading statistics per site, scored by box sum or PSF matched filter.

    ``method="box"`` expects top-hat filtered ``images``; ``method="psf"`` builds a
    template per site and takes the raw frames. Pass ``scorer`` to reuse templates
    built elsewhere.
    """
    exp_params = ExpParameterManager()
    conversion_factor = exp_params.get_parameter("conversion_factor")
    conversion_offset = exp_params.get_parameter("conversion_offset")

    if scorer is None and method == "psf":
        scorer = SiteScorer.from_images(images, grid_positions, window_size=psf_window)

    # Extract photon counts for each image and each trap site
    if scorer is not None:
        photon_array = scorer.score_stack(images)   # SiteScorer already returns photons
    else:
        raw_counts = np.array([sum_pixel_values(image, grid_positions, window_size=window_size) for image in images])
        photon_array = (raw_counts - conversion_offset) * conversion_factor / 0.7
    tot_photon_array = photon_array.flatten()

    # Threshold detection and fidelity calculation
    threshold_detection_success = False
    if threshold_detection:
        try:
            threshold, bg_params, sig_params = detect_loading_threshold(tot_photon_array)
            mu_bg, var_bg, weight_bg  = bg_params
            mu_sig, var_sig, weight_sig = sig_params
            prob_false_negative = norm.cdf(threshold, loc=mu_sig, scale=np.sqrt(var_sig))
            prob_false_positive = 1.0 - norm.cdf(threshold, loc=mu_bg, scale=np.sqrt(var_bg))
            total_error = (weight_bg * prob_false_positive) + (weight_sig * prob_false_negative)
            fidelity = 1.0 - total_error
            threshold_detection_success = True
        except Exception as e:
            print(f"Threshold detection failed: {e}. Trying default threshold of {threshold:.2f} kHz.")
            try:
                mu_bg = tot_photon_array[tot_photon_array < threshold].mean()
                mu_sig = tot_photon_array[tot_photon_array >= threshold].mean()
                var_bg = tot_photon_array[tot_photon_array < threshold].var()
                var_sig = tot_photon_array[tot_photon_array >= threshold].var()
                fidelity = norm.cdf((threshold - mu_sig) / np.sqrt(var_sig)) + (1 - norm.cdf((threshold - mu_bg) / np.sqrt(var_bg)))
            except Exception as e:
                print(f"Fallback threshold calculation failed: {e}. Setting fidelity to 0.")
                fidelity = np.nan      
    else:
        try:
            mu_bg = tot_photon_array[tot_photon_array < threshold].mean()
            mu_sig = tot_photon_array[tot_photon_array >= threshold].mean()
            var_bg = tot_photon_array[tot_photon_array < threshold].var()
            var_sig = tot_photon_array[tot_photon_array >= threshold].var()
            fidelity = norm.cdf((threshold - mu_sig) / np.sqrt(var_sig)) + (1 - norm.cdf((threshold - mu_bg) / np.sqrt(var_bg)))
        except Exception as e:
            print(f"Fallback threshold calculation failed: {e}. Setting fidelity to 0.")
            fidelity = np.nan

    # Atom counting and loading probabilities
    atom_counter = (photon_array > threshold).astype(int).sum(axis=0)
    loading_probabilities = atom_counter / len(images)
    
    if show_histogram:
        fig, ax = plt.subplots(1, 3, figsize = (17,5), constrained_layout=True)
        for i in range(len(grid_positions)):
            ax[0].hist(photon_array[:, i], bins=40, density=True, alpha=0.6, range=(min(tot_photon_array), max(tot_photon_array)))
            ax[0].set_xlabel('Photons')
            ax[0].set_ylabel('Probability Density')
            if verbose:
                print(f'Trap ({i}) Loading Probability : {loading_probabilities[i]*100:.2f} %')
        if threshold_detection and threshold_detection_success:
            x_fit = np.linspace(min(tot_photon_array), max(tot_photon_array), 1000)
            pdf_bg = weight_bg * norm.pdf(x_fit, mu_bg, np.sqrt(var_bg))
            pdf_sig = weight_sig * norm.pdf(x_fit, mu_sig, np.sqrt(var_sig))
            ax[1].plot(x_fit, pdf_bg + pdf_sig, 'k-', lw=2,)
            ax[1].plot(x_fit, pdf_bg, 'b--', lw=2, label=f'Background Fit ($\mu$={mu_bg:.1f})')
            ax[1].plot(x_fit, pdf_sig, 'r--', lw=2, label=f'Signal Fit ($\mu$={mu_sig:.1f})')
        ax[1].hist(tot_photon_array, bins=binning, density=True, alpha=0.5, color='gray', edgecolor='black', label='Raw Data')
        ax[1].axvline(x=threshold, color='green', linestyle='--', lw=2, label=f'Threshold ({threshold:.1f})')
        ax[1].set_xlabel('Photons')
        ax[1].set_ylabel('Probability Density')
        ax[1].legend()
        X, Y = np.array(list(grid_positions.values())).T
        ax[2].scatter(Y, X, c=loading_probabilities, cmap='viridis', s=200)
        ax[2].invert_yaxis()
        if show_site_labels:
            for i, (x, y) in enumerate(zip(X, Y)):
                ax[2].text(y, x, str(i), color='white', fontsize=8, ha='center', va='center')
        cbar = plt.colorbar(ax[2].collections[0], ax=ax[2])
        cbar.set_label('Loading Probability')

    if verbose:
        print(f"Detected Loading Threshold: {threshold:.2f} photons")
        print(f"Std dev of Loading Probabilities: {np.std(loading_probabilities) / loading_probabilities.mean()*100:.2f} %")
        print(f"Mean Loading Probability: {loading_probabilities.mean()*100:.2f} %")
        if fidelity is not np.nan:
            print(f"Detection Fidelity: {fidelity*100:.2f} %")
            print(f"Mean Background Photon Count: {mu_bg:.2f} photons, Variance: {var_bg:.2f} photons^2")
            print(f"Mean Signal Photon Count: {mu_sig:.2f} photons, Variance: {var_sig:.2f} photons^2")

    return photon_array, loading_probabilities, threshold, fidelity

def extract_survival_probability(imgs1, imgs2, grid_positions, threshold='auto', window_size=5):
    if threshold == 'auto':
        pr_1, eta_1, thresh_1, fidelity_1 = get_array_loading_statistics(imgs1, grid_positions, threshold_detection=True, window_size=window_size, binning=60, show_histogram=False, verbose=False)
        pr_2, eta_2, thresh_2, fidelity_2 = get_array_loading_statistics(imgs2, grid_positions, threshold_detection=True, window_size=window_size, binning=60, show_histogram=False, verbose=False)
    else:
        pr_1, eta_1, thresh_1, fidelity_1 = get_array_loading_statistics(imgs1, grid_positions, threshold_detection=False, threshold=threshold, window_size=window_size, binning=60, show_histogram=False, verbose=False)
        pr_2, eta_2, thresh_2, fidelity_2 = get_array_loading_statistics(imgs2, grid_positions, threshold_detection=False, threshold=threshold, window_size=window_size, binning=60, show_histogram=False, verbose=False)
    survival_fractions = []
    for i in range(pr_1.shape[0]):

        pr_mat_1 = pr_1[i]
        pr_mat_2 = pr_2[i]

        pr_mat_1_binary = (pr_mat_1 > thresh_1).astype(int)
        pr_mat_2_binary = (pr_mat_2 > thresh_2).astype(int)
        survival_matrix = pr_mat_1_binary * pr_mat_2_binary

        init_occupation = pr_mat_1_binary.sum()
        final_occupation = pr_mat_2_binary.sum()
        survival_count = survival_matrix.sum()
        survival_fraction = survival_count / init_occupation if init_occupation > 0 else 0
        survival_fractions.append(survival_fraction)

    survival_probability = np.mean(survival_fractions)
    return survival_probability

def recapture_prob_fit(tarray, T, a, b):
    exp_params = ExpParameterManager()
    P = exp_params.get_parameter("mean_tweezer_power_mW") * 1E-3
    wr = exp_params.get_parameter("radial_trap_freq_kHz") * 1E3 * 2 * np.pi

    C = 1/(2*c*e0) * muD2**2/hbar * (1/(wD2 - w) + 1/(wD2 + w)) + 1/(2*c*e0) * muD1**2/hbar * (1/(wD1 - w) + 1/(wD1 + w)) # This is the combined polarizability term U0 / I0
    U0 = np.sqrt(C * P * MRb * wr**2 / (2 * np.pi))
    w0 = np.sqrt(4*U0 / (MRb * wr**2))

    probs = []
    for t in tarray:
        escape_v_ineq = lambda ve, t: 0.5 * MRb * ve**2 - U0 * np.exp(- 2 * ve**2 * t**2 / w0**2)
        maxwell_p = lambda v, T: (MRb*v)/(kB*T) * np.exp(-MRb*v**2 / (2*kB*T))
        xx = np.linspace(0, 1, 100000)
        y = escape_v_ineq(xx, t)
        ve = xx[np.argmin(np.abs(y))]
        prob = integrate.quad(maxwell_p, 0, ve, args=(T))[0]
        probs.append(prob)
    return a * np.array(probs) + b

def extract_temperature(dropTimeList, survival_prob_list, plot=True):
    dropt = dropTimeList*10*1e-6
    survival_prob_list = np.array(survival_prob_list)
    survival_prob_list = survival_prob_list[~np.isnan(survival_prob_list)]
    dropt = dropt[~np.isnan(survival_prob_list)]
    try:
        params, pcov = curve_fit(recapture_prob_fit, dropt, survival_prob_list, p0 = [50e-6, 1.5, 0.1])
        temperature = params[0] * 1e6
    except:
        temperature = 100
    if plot:
        xfit = np.linspace(dropt.min(), dropt.max(), 100)
        yfit = recapture_prob_fit(xfit, temperature*1e-6, params[1], params[2])
        plt.figure()
        plt.plot(dropt*1e6, survival_prob_list, 'o', label='Data', color='blue')
        plt.plot(xfit*1e6, yfit, '-', color='red', label=f'Temperature = {temperature:.2f} uK')
        plt.grid()
        plt.xlabel('Drop Time (us)')
        plt.ylabel('Survival Probability')
        plt.legend()
        plt.show()

    return temperature