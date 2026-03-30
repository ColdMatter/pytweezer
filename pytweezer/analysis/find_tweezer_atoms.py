import numpy as np
from scipy import ndimage
from pytweezer.servers import DataClient, ImageClient
from pytweezer.servers import Properties, PropertyAttribute
import argparse
import importlib
from skimage.feature import blob_log

# def _get_blob_log():
#     try:
#         feature = importlib.import_module('skimage.feature')
#         return getattr(feature, 'blob_log', None)
#     except Exception:
#         return None

# filepath: /home/twalker/RbCaF/pytweezer/pytweezer/analysis/find_tweezer_atoms.py
''' Identify atom locations in a tweezer image and stream their positions.

Input:
    Image stream with atoms in tweezers

Output:
    Data stream with atom positions (coordinates)

Properties:
    *   imagestreams: ([str]) input image streams
    *   threshold: (int) minimum raw intensity at a detected blob center
    *   atom_fwhm_px: (float) expected atom gaussian full-width at half maximum in pixels
    *   blob_min_sigma: (float) minimum gaussian sigma for blob detection (pixels)
    *   blob_max_sigma: (float) maximum gaussian sigma for blob detection (pixels)
    *   blob_num_sigma: (int) number of sigma steps in blob detection
    *   blob_threshold: (float) detection threshold on normalized response, 0..1
    *   blob_overlap: (float) overlap threshold used by blob suppression

'''


class TweezerGridAnalyzer():

    _imagestreams = PropertyAttribute('imagestreams', ['None'])
    _threshold = PropertyAttribute('threshold', 40)
    _atom_fwhm_px = PropertyAttribute('atom_fwhm_px', 5.0)
    _blob_min_sigma = PropertyAttribute('blob_min_sigma', 1.2)
    _blob_max_sigma = PropertyAttribute('blob_max_sigma', 4.0)
    _blob_num_sigma = PropertyAttribute('blob_num_sigma', 8)
    _blob_threshold = PropertyAttribute('blob_threshold', 0.12)
    _blob_overlap = PropertyAttribute('blob_overlap', 0.5)

    def __init__(self, name):
        self._props = Properties(name)
        self._name = name

        self.imageq = ImageClient(name.split('/')[-1])
        self.imageq.subscribe(self._imagestreams)
        self.dataq = DataClient(name.split('/')[-1])
        print('find_tweezer_atoms.py subscriptions: ', self._imagestreams)

    @staticmethod
    def _normalize_response(response):
        lo = np.percentile(response, 10.0)
        hi = np.percentile(response, 99.5)
        scale = max(hi - lo, 1e-9)
        out = (response - lo) / scale
        return np.clip(out, 0.0, 1.0)

    def _blob_detect_fallback(self, response_norm, min_sigma):
        window = max(3, int(round(2.0 * min_sigma)) | 1)
        local_max = response_norm == ndimage.maximum_filter(response_norm, size=window, mode='nearest')
        peak_mask = local_max & (response_norm >= float(self._blob_threshold))
        coords = np.argwhere(peak_mask)
        if coords.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        return coords.astype(np.float64)

    def find_atoms(self, img):
        """Detect atom centers via Laplacian-of-Gaussian blob detection."""
        image = np.asarray(img, dtype=np.float64)

        atom_sigma = max(float(self._atom_fwhm_px) / 2.35482, 0.8)
        smooth = ndimage.gaussian_filter(image, sigma=max(0.6, 0.5 * atom_sigma))
        background = ndimage.gaussian_filter(smooth, sigma=max(4.0, 3.0 * atom_sigma))
        response = smooth - background
        response_norm = self._normalize_response(response)

        min_sigma = max(0.5, float(self._blob_min_sigma))
        max_sigma = max(min_sigma, float(self._blob_max_sigma))
        num_sigma = max(2, int(self._blob_num_sigma))

        blobs = blob_log(
            response_norm,
            min_sigma=min_sigma,
        
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=float(self._blob_threshold),
            overlap=float(self._blob_overlap),
        )
        if blobs.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        peak_coords = blobs[:, :2].astype(np.float64)

        # Keep only blobs that are bright enough in the original/smoothed image.
        adaptive_floor = np.mean(smooth) + 1.8 * np.std(smooth)
        intensity_floor = max(float(self._threshold), adaptive_floor)
        h, w = smooth.shape
        keep = []
        for y, x in peak_coords:
            iy = int(np.clip(round(y), 0, h - 1))
            ix = int(np.clip(round(x), 0, w - 1))
            keep.append(smooth[iy, ix] >= intensity_floor)

        keep = np.asarray(keep, dtype=bool)
        if not np.any(keep):
            return np.empty((0, 2), dtype=np.float64)

        return peak_coords[keep]

    def run(self):
        while True:
            msg = self.imageq.recv()
            if msg is not None:
                msgstr, head, img = msg
                
                atom_positions = self.find_atoms(img)
                
                output_data = {
                    'positions': atom_positions,
                    'count': len(atom_positions),
                    'timestamp': head.get('timestamp', 0)
                }

                self.dataq.send(output_data, channel='_')


def main_run(name):
    analyzer = TweezerGridAnalyzer(name)
    analyzer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs=1, help='name of this program instance')
    args = parser.parse_args()
    name = args.name[0]
    main_run(name)