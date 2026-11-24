"""Total camera counts at the tweezer array's trap sites, and in the rest of the frame.

The sites are the boxes the Tweezer Monitor draws: a ``trap_window_size``
square around each of the saved ``trap_sites`` (full-sensor pixels), placed in
each frame by the ROI origin in its header (``roi_x0``/``roi_y0``; 0 when
missing). Counts are the raw pixel values, with no filtering, offset
subtraction or conversion to photons.

Input:
    One image stream: the tweezer camera.

Output:
    A data message (no array) per accepted frame, published on ``<name>``,
    carrying ``atom_counts`` (summed over every site's box),
    ``background_counts`` (summed over every other pixel of the frame), the
    pixel numbers ``atom_pixels`` and ``background_pixels`` they were summed
    over, ``sites``, and the input's ``index``, ``timestamp`` and ``rid`` when
    it has them. Without usable sites both counts are NaN and ``error`` says
    why.

Properties:
    *   imagestreams: ([str]) input image streams.
    *   image_index: (int) which frame of a shot to count, by its ``index``
        header; negative counts every frame.

A pixel in two overlapping boxes is counted once. Boxes reaching past the
frame's edge are cut at it. The saved parameters are read at most every
``REFRESH_S`` seconds and again on a new ``rid``, and the last answer is kept
while the Experiment Manager cannot be reached.
"""

import time

import numpy as np

from pytweezer.analysis.analysis_base import ImageAnalysis, run_analysis
from pytweezer.analysis.trap_sites import to_roi
from pytweezer.experiment.client import (
    ManagerError,
    ManagerUnavailable,
    get_saved_parameters,
)
from pytweezer.logging_utils import get_logger
from pytweezer.servers import DataClient, PropertyAttribute

LOGGER = get_logger("tweezer_counts")

REFRESH_S = 10.0
NO_SITES = "No trap sites are saved: run FindTrapSites and accept its proposal"


def site_mask(shape, sites, window_size):
    """Boolean mask of ``shape``, true in a ``window_size`` box at each ``(row, col)`` of ``sites``."""
    mask = np.zeros(shape, dtype=bool)
    half = window_size // 2
    for row, col in np.rint(sites).astype(int):
        mask[
            max(row - half, 0) : max(row + half + 1, 0),
            max(col - half, 0) : max(col + half + 1, 0),
        ] = True
    return mask


class TweezerCounts(ImageAnalysis):
    _image_index = PropertyAttribute("image_index", 0)

    _saved = None
    _saved_asked_at = -np.inf
    _rid = None

    def __init__(self, name):
        super().__init__(name)
        # Scalars, not a frame, so they go out on the Data hub.
        self.dataq = DataClient(name.split("/")[-1])

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

    def process(self, head, data):
        image_index = int(self._image_index)
        if data is None or data.ndim != 2:
            return
        if image_index >= 0 and head.get("index", image_index) != image_index:
            return
        rid = head.get("rid")
        if rid != self._rid:
            self._rid = rid
            self._saved_asked_at = -np.inf

        out_head = {
            key: head[key] for key in ("index", "timestamp", "rid") if key in head
        }
        saved = self._saved_parameters() or {}
        if "trap_sites" not in saved or "trap_window_size" not in saved:
            self.dataq.send(
                out_head
                | {"atom_counts": np.nan, "background_counts": np.nan, "sites": 0}
                | {"error": NO_SITES},
                None,
            )
            return

        sites = to_roi(
            saved["trap_sites"], head.get("roi_y0", 0), head.get("roi_x0", 0)
        )
        mask = site_mask(data.shape, sites, int(saved["trap_window_size"]))
        frame = np.asarray(data, dtype=np.float64)
        atom_pixels = int(mask.sum())
        self.dataq.send(
            out_head
            | {
                "atom_counts": float(frame[mask].sum()),
                "background_counts": float(frame[~mask].sum()),
                "atom_pixels": atom_pixels,
                "background_pixels": int(mask.size - atom_pixels),
                "sites": len(sites),
            },
            None,
        )
        return


if __name__ == "__main__":
    run_analysis(TweezerCounts)
