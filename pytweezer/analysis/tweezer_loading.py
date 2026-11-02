"""Tweezer-array loading statistics over a rolling window of shots.

The live counterpart of ``analysis.get_array_loading_statistics`` as run in
the optimisation notebooks: each frame is top-hat filtered, a box is summed at
every trap site and converted to photons, and a two-Gaussian fit to all the
sites' photon counts gives the occupancy threshold and detection fidelity.

Input:
    One image stream: the tweezer camera, two frames per shot.

Output:
    One data message per accepted frame on ``<name>``. The array is the photon
    count of every site in every shot of the window, ``(shots, sites)``; the
    header carries ``shots``, ``sites``, ``threshold`` (photons), ``fidelity``,
    ``loading`` (mean over sites), ``loading_spread`` (std/mean over sites),
    ``site_loading``, ``site_rows``/``site_cols`` (pixels in the frame) and the
    fit components ``background``/``signal`` as ``[mean, variance, weight]``.
    ``sites_from`` says where the number of sites came from: ``"rig"`` or
    ``"property"``. Until the sites are found the message is header-only, with
    ``shots``, ``sites`` and ``sites_from``.

Properties:
    *   imagestreams: ([str]) input image streams.
    *   image_index: (int) which frame of a shot to score, by its ``index``
        header; 0 is the loading image.
    *   shots: (int) frames in the rolling window.
    *   rig: (str) device that knows which array is on the SLM, asked for
        the number of sites; empty to use ``sites`` alone.
    *   sites: (int) trap sites in the array, when the rig does not know.
    *   feature_size: (int) top-hat filter size in pixels.
    *   window_size: (int) side of the box summed at each site, in pixels.
    *   detection_step: (float) step in counts by which the site-finding level
        is lowered from the brightest pixel of the mean image.
    *   threshold: (float) occupancy threshold in photons, used only when the
        two-Gaussian fit fails.

The rig is asked at most every ``RIG_REFRESH_S`` seconds, and its last answer
is kept while it cannot be reached. It does not know the array when the SLM was
last written by anything but the rig, and ``sites`` is used then.

The sites are re-found in the mean of the window on every frame and kept from
the last success, so after the array is moved they follow once the old frames
have left the window. A change of frame size empties the window.
"""

import time
from collections import deque

import numpy as np
from scipy.stats import norm

from pytweezer.analysis import analysis as an
from pytweezer.analysis.analysis_base import ImageAnalysis, run_analysis
from pytweezer.logging_utils import get_logger
from pytweezer.servers import DataClient, PropertyAttribute
from pytweezer.servers.device_client import get_device

LOGGER = get_logger("tweezer_loading")

RIG_REFRESH_S = 10.0
RIG_TIMEOUT_S = 5.0


def find_sites(image, n_sites, step):
    """``n_sites`` bright spots in ``image`` as ``{index: (row, col)}``, or ``None``."""
    level = float(image.max())
    while level >= 1:
        centres = an.detect_bright_points(image, threshold=level)
        if len(centres) == n_sites:
            return {i: (int(row), int(col)) for i, (row, col) in enumerate(centres)}
        level -= step
    return None


class LoadingStatistics:
    """A rolling window of filtered frames and the array's loading statistics over it."""

    def __init__(self):
        self.frames = deque()
        self.frame_sum = None
        self.sites = None

    def add(self, frame, *, shots, feature_size):
        frame = an.morphological_tophat_high_pass(
            np.asarray(frame, dtype=np.float64), feature_size=feature_size
        )
        if self.frame_sum is None or self.frame_sum.shape != frame.shape:
            self.frames.clear()
            self.frame_sum = np.zeros_like(frame)
            self.sites = None
        self.frames.append(frame)
        self.frame_sum += frame
        while len(self.frames) > shots:
            self.frame_sum -= self.frames.popleft()

    def summary(self, *, n_sites, window_size, detection_step, fallback_threshold):
        """``(header, photons)`` for the window; ``photons`` is ``None`` until the sites are found."""
        found = find_sites(self.frame_sum / len(self.frames), n_sites, detection_step)
        if found is not None:
            self.sites = found
        elif self.sites is not None and len(self.sites) != n_sites:
            self.sites = None
        head = {"shots": len(self.frames), "sites": n_sites}
        if self.sites is None:
            return head, None

        counts = np.array(
            [
                an.sum_pixel_values(frame, self.sites, window_size=window_size)
                for frame in self.frames
            ]
        )
        photons = an.convert_counts_to_photons(counts).astype(np.float64)

        try:
            threshold, background, signal = an.detect_loading_threshold(photons.ravel())
        except (ValueError, TypeError, StopIteration):
            threshold, background, signal, fidelity = (
                fallback_threshold,
                None,
                None,
                None,
            )
        else:
            (mu_bg, var_bg, weight_bg), (mu_sig, var_sig, weight_sig) = (
                background,
                signal,
            )
            false_positive = 1.0 - norm.cdf(threshold, mu_bg, np.sqrt(var_bg))
            false_negative = norm.cdf(threshold, mu_sig, np.sqrt(var_sig))
            fidelity = 1.0 - (weight_bg * false_positive + weight_sig * false_negative)
            fidelity = float(fidelity) if np.isfinite(fidelity) else None
            background = [float(value) for value in background]
            signal = [float(value) for value in signal]

        site_loading = (photons > threshold).mean(axis=0)
        mean_loading = float(site_loading.mean())
        rows, cols = zip(*(self.sites[i] for i in range(n_sites)), strict=True)
        head |= {
            "threshold": float(threshold),
            "fidelity": fidelity,
            "loading": mean_loading,
            "loading_spread": float(site_loading.std() / mean_loading)
            if mean_loading > 0
            else None,
            "site_loading": [float(value) for value in site_loading],
            "site_rows": list(rows),
            "site_cols": list(cols),
            "background": background,
            "signal": signal,
        }
        return head, photons


class TweezerLoading(ImageAnalysis):
    _image_index = PropertyAttribute("image_index", 0)
    _shots = PropertyAttribute("shots", 100)
    _rig = PropertyAttribute("rig", "Rb Rearrangement Rig")
    _sites = PropertyAttribute("sites", 64)
    _feature_size = PropertyAttribute("feature_size", 10)
    _window_size = PropertyAttribute("window_size", 5)
    _detection_step = PropertyAttribute("detection_step", 100.0)
    _threshold = PropertyAttribute("threshold", 100.0)

    _rig_client = None
    _rig_array = None
    _rig_asked_at = -np.inf

    def __init__(self, name):
        super().__init__(name)
        # Statistics, not a frame, so they go out on the Data hub.
        self.dataq = DataClient(name.split("/")[-1])
        self.statistics = LoadingStatistics()

    def _array_on_rig(self):
        """The rig's answer to which array is on the SLM, or ``None`` if it does not know."""
        rig = self._rig
        if not rig:
            return None
        now = time.monotonic()
        if now - self._rig_asked_at < RIG_REFRESH_S:
            return self._rig_array
        self._rig_asked_at = now
        try:
            if self._rig_client is None:
                self._rig_client = get_device(rig, timeout=RIG_TIMEOUT_S)
            self._rig_array = self._rig_client.get_array()
        except Exception as error:  # an unreachable rig must not stop the analysis
            LOGGER.warning("Could not ask %s for its array: %r", rig, error)
            self._rig_client = None
        return self._rig_array

    def process(self, head, data):
        image_index = int(self._image_index)
        if data.ndim != 2 or head.get("index", image_index) != image_index:
            return
        self.statistics.add(
            data, shots=max(1, int(self._shots)), feature_size=int(self._feature_size)
        )
        array = self._array_on_rig()
        out_head, photons = self.statistics.summary(
            n_sites=array["sites"] if array else int(self._sites),
            window_size=int(self._window_size),
            detection_step=max(1e-3, float(self._detection_step)),
            fallback_threshold=float(self._threshold),
        )
        out_head["sites_from"] = "rig" if array else "property"
        if "timestamp" in head:
            out_head["timestamp"] = head["timestamp"]
        self.dataq.send(out_head, photons)
        return


if __name__ == "__main__":
    run_analysis(TweezerLoading)
