"""Tweezer-array loading statistics over the shots of the current task.

The live counterpart of ``analysis.get_array_loading_statistics`` as run in
the optimisation notebooks: each frame is top-hat filtered, a box is summed at
every trap site and converted to photons, and a two-Gaussian fit to all the
sites' photon counts gives the occupancy threshold and detection fidelity.

The trap sites, box and top-hat sizes are the saved parameters accepted from
``FindTrapSites`` (``trap_sites`` in full-sensor pixels, ``trap_window_size``,
``trap_feature_size``), placed in each frame by the ROI origin in its header
(``roi_x0``/``roi_y0``; 0 when missing).

Input:
    One image stream: the tweezer camera, two frames per shot.

Output:
    One data message per accepted frame on ``<name>``. The array is the photon
    count of every site in every shot of the window, ``(shots, sites)``; the
    header carries ``shots``, ``sites``, ``threshold`` (photons), ``fidelity``,
    ``loading`` (mean over sites), ``loading_spread`` (std/mean over sites),
    ``site_loading``, ``site_rows``/``site_cols`` (pixels in the frame) and the
    fit components ``background``/``signal`` as ``[mean, variance, weight]``.
    ``sites_match`` is ``False`` when the SLM has another array on it than
    the sites were fitted to, and ``rid`` is the task the frames came from.
    Without usable sites the message is header-only, with ``shots``,
    ``sites``, ``rid`` and an ``error`` saying why.

Properties:
    *   imagestreams: ([str]) input image streams.
    *   image_index: (int) which frame of a shot to score, by its ``index``
        header; 0 is the loading image.
    *   shots: (int) most recent frames of the task kept in the window.
    *   slm: (str) SLM device that knows which array is on it, asked whether
        it is the one the sites were fitted to; empty not to ask.
    *   threshold: (float) occupancy threshold in photons, used only when the
        two-Gaussian fit fails.

A frame whose ``rid`` header differs from the last one starts the analysis
afresh, so each task is analysed on its own shots. Frames without a ``rid``
(from a notebook, say) count as one run of their own.

The SLM and the saved parameters are each asked at most every ``REFRESH_S``
seconds and again on a new task, and their last answers are kept while they
cannot be reached.
"""

import time
from collections import deque

import numpy as np

from pytweezer.analysis import analysis as an
from pytweezer.analysis.analysis_base import ImageAnalysis, run_analysis
from pytweezer.analysis.array_loading import loading_statistics, photons_at_sites
from pytweezer.analysis.trap_sites import same_geometry, to_roi
from pytweezer.experiment.client import (
    ManagerError,
    ManagerUnavailable,
    get_saved_parameters,
)
from pytweezer.logging_utils import get_logger
from pytweezer.servers import DataClient, PropertyAttribute
from pytweezer.servers.device_client import get_device

LOGGER = get_logger("tweezer_loading")

REFRESH_S = 10.0
SLM_TIMEOUT_S = 5.0
TRAP_SITE_KEYS = ("trap_sites", "trap_window_size", "trap_feature_size")
NO_SITES = "No trap sites are saved: run FindTrapSites and accept its proposal"


class LoadingStatistics:
    """A rolling window of filtered frames and the array's loading statistics over it."""

    def __init__(self):
        self.frames = deque()
        self.frame_sum = None

    def add(self, frame, *, shots, feature_size):
        frame = an.morphological_tophat_high_pass(
            np.asarray(frame, dtype=np.float64), feature_size=feature_size
        )
        if self.frame_sum is None or self.frame_sum.shape != frame.shape:
            self.frames.clear()
            self.frame_sum = np.zeros_like(frame)
        self.frames.append(frame)
        self.frame_sum += frame
        while len(self.frames) > shots:
            self.frame_sum -= self.frames.popleft()

    def summary(self, *, sites, window_size, fallback_threshold):
        """``(header, photons)`` for the window at ``sites`` (``{index: (row, col)}``)."""
        photons = photons_at_sites(self.frames, sites, window_size)
        head = {"shots": len(self.frames), "sites": len(sites)}
        return head | loading_statistics(photons, sites, fallback_threshold), photons


class TweezerLoading(ImageAnalysis):
    _image_index = PropertyAttribute("image_index", 0)
    _shots = PropertyAttribute("shots", 100)
    _slm = PropertyAttribute("slm", "Rb SLM")
    _threshold = PropertyAttribute("threshold", 100.0)

    _slm_client = None
    _slm_array = None
    _slm_asked_at = -np.inf
    _saved = None
    _saved_asked_at = -np.inf
    _rid = None

    def __init__(self, name):
        super().__init__(name)
        # Statistics, not a frame, so they go out on the Data hub.
        self.dataq = DataClient(name.split("/")[-1])
        self.statistics = LoadingStatistics()

    def _array_on_slm(self):
        """The SLM's answer to which array is on it, or ``None`` if it does not know."""
        slm = self._slm
        if not slm:
            return None
        now = time.monotonic()
        if now - self._slm_asked_at < REFRESH_S:
            return self._slm_array
        self._slm_asked_at = now
        try:
            if self._slm_client is None:
                self._slm_client = get_device(slm, timeout=SLM_TIMEOUT_S)
            self._slm_array = self._slm_client.get_array()
        except Exception as error:  # an unreachable SLM must not stop the analysis
            LOGGER.warning("Could not ask %s for its array: %r", slm, error)
            self._slm_client = None
        return self._slm_array

    def _saved_parameters(self):
        now = time.monotonic()
        if now - self._saved_asked_at < REFRESH_S:
            return self._saved
        self._saved_asked_at = now
        try:
            self._saved = get_saved_parameters()
        except (ManagerUnavailable, ManagerError) as error:
            LOGGER.warning("Could not read the saved trap sites: %r", error)
        return self._saved

    def _sites_in(self, head, shape):
        """The saved sites as ``{index: (row, col)}`` in this frame, or ``None`` and why."""
        saved = self._saved_parameters() or {}
        if any(key not in saved for key in TRAP_SITE_KEYS):
            return None, NO_SITES
        sites = np.rint(
            to_roi(saved["trap_sites"], head.get("roi_y0", 0), head.get("roi_x0", 0))
        ).astype(int)
        half = int(saved["trap_window_size"]) // 2
        rows, cols = sites.T
        if (
            min(rows.min(), cols.min()) < half
            or rows.max() + half >= shape[0]
            or cols.max() + half >= shape[1]
        ):
            return None, "The saved trap sites lie outside this camera frame"
        return {i: (int(row), int(col)) for i, (row, col) in enumerate(sites)}, ""

    def process(self, head, data):
        image_index = int(self._image_index)
        if data.ndim != 2 or head.get("index", image_index) != image_index:
            return
        rid = head.get("rid")
        if rid != self._rid:
            self._rid = rid
            self.statistics = LoadingStatistics()
            # A new task may have just put a new array on the SLM, or new sites.
            self._slm_asked_at = self._saved_asked_at = -np.inf
        sites, why_not = self._sites_in(head, data.shape)
        if sites is None:
            self.dataq.send(
                {"shots": len(self.statistics.frames), "sites": 0, "rid": rid}
                | {"error": why_not},
                None,
            )
            return
        saved = self._saved
        self.statistics.add(
            data,
            shots=max(1, int(self._shots)),
            feature_size=int(saved["trap_feature_size"]),
        )
        out_head, photons = self.statistics.summary(
            sites=sites,
            window_size=int(saved["trap_window_size"]),
            fallback_threshold=float(self._threshold),
        )
        array = self._array_on_slm()
        out_head["sites_match"] = not array or (
            "trap_geometry" in saved
            and same_geometry(saved["trap_geometry"], array["x_um"], array["y_um"])
        )
        out_head["rid"] = rid
        if "timestamp" in head:
            out_head["timestamp"] = head["timestamp"]
        self.dataq.send(out_head, photons)


if __name__ == "__main__":
    run_analysis(TweezerLoading)
