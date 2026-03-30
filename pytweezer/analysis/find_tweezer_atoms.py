import numpy as np
from scipy import ndimage
from pytweezer.servers import DataClient, ImageClient
from pytweezer.servers import Properties, PropertyAttribute
import argparse

# filepath: /home/twalker/RbCaF/pytweezer/pytweezer/analysis/find_tweezer_atoms.py
''' Identify atom locations in a tweezer image and stream their positions.

Input:
    Image stream with atoms in tweezers

Output:
    Data stream with atom positions (coordinates)

Properties:
    *   imagestreams: ([str]) input image streams
    *   threshold: (float) minimum raw intensity at a detected atom center
    *   atom_fwhm_px: (float) expected atom gaussian full-width at half maximum in pixels
    *   expected_sites: (int) expected number of available tweezer sites for filling-fraction calculation

'''


class TweezerGridAnalyzer():

    _imagestreams = PropertyAttribute('imagestreams', ['None'])
    _threshold = PropertyAttribute('threshold', 1.0)
    _atom_fwhm_px = PropertyAttribute('atom_fwhm_px', 5.0)
    _expected_sites = PropertyAttribute('expected_sites', 64)

    def __init__(self, name):
        self._props = Properties(name)
        self._name = name

        self.imageq = ImageClient(name.split('/')[-1])
        self.imageq.subscribe(self._imagestreams)
        self.dataq = DataClient(name.split('/')[-1])
        print('find_tweezer_atoms.py subscriptions: ', self._imagestreams)

    @staticmethod
    def _robust_sigma(x):
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        return max(mad / 0.67448975, 1e-6), med

    def _refine_centroid(self, highpass, y, x, atom_sigma):
        h, w = highpass.shape
        r = max(1, int(round(atom_sigma)))
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)

        patch = highpass[y0:y1, x0:x1]
        weights = np.clip(patch, 0.0, None)
        wsum = weights.sum()
        if wsum <= 0:
            return float(y), float(x)

        yy, xx = np.mgrid[y0:y1, x0:x1]
        cy = float((weights * yy).sum() / wsum)
        cx = float((weights * xx).sum() / wsum)
        return cy, cx

    def find_atoms(self, img):
        """Detect atom centers with minimal tuning parameters."""
        image = np.asarray(img, dtype=np.float64)

        atom_sigma = max(float(self._atom_fwhm_px) / 2.35482, 0.8)

        # One-scale background suppression + matched filtering for Gaussian spots.
        smooth = ndimage.gaussian_filter(image, sigma=max(0.6, 0.5 * atom_sigma))
        background = ndimage.gaussian_filter(smooth, sigma=max(6.0, 4.0 * atom_sigma))
        matched = ndimage.gaussian_filter(smooth - background, sigma=atom_sigma)

        noise_sigma, noise_med = self._robust_sigma(matched)
        snr = (matched - noise_med) / noise_sigma

        # Derive peak separation from expected spot size.
        min_distance = max(3.0, 1.8 * atom_sigma)
        win = max(3, int(round(min_distance)))
        if win % 2 == 0:
            win += 1

        local_max = matched == ndimage.maximum_filter(matched, size=win, mode='nearest')
        candidate_mask = local_max & (snr >= 2.8) & (matched > 0.0)

        raw_floor = float(self._threshold)
        if raw_floor > 0:
            candidate_mask &= (image >= raw_floor)

        peak_coords = np.argwhere(candidate_mask).astype(np.float64)
        if peak_coords.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float64)

        refined = []
        for y, x in peak_coords:
            iy, ix = int(round(y)), int(round(x))
            cy, cx = self._refine_centroid(matched, iy, ix, atom_sigma)
            refined.append([cy, cx])

        return np.asarray(refined, dtype=np.float64)

    def run(self):
        while True:
            msg = self.imageq.recv()
            if msg is not None:
                msgstr, head, img = msg
                
                atom_positions = self.find_atoms(img)
                atom_count = int(len(atom_positions))

                expected_sites = int(self._expected_sites) if int(self._expected_sites) > 0 else 0
                filling_fraction = float(atom_count) / float(expected_sites) if expected_sites > 0 else 0.0
                
                output_data = {
                    'positions': atom_positions,
                    'count': atom_count,
                    'n_atoms': atom_count,
                    'filling_fraction': filling_fraction,
                    'timestamp': head.get('timestamp', 0)
                }

                stats_data = {
                    'n_atoms': atom_count,
                    'filling_fraction': filling_fraction,
                    'expected_sites': expected_sites,
                    'timestamp': head.get('timestamp', 0),
                }

                self.dataq.send(output_data, channel='_')
                self.dataq.send(stats_data, channel='_stats')


def main_run(name):
    analyzer = TweezerGridAnalyzer(name)
    analyzer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs=1, help='name of this program instance')
    args = parser.parse_args()
    name = args.name[0]
    main_run(name)