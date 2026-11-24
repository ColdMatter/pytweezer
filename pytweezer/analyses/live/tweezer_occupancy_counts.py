"""Total camera counts at the occupied and at the empty trap sites of each shot's first image.

A site is occupied when its photons exceed the saved ``trap_threshold``,
scored as ``LoadTweezers`` scores it: the frame top-hat filtered with
``trap_feature_size``, a ``trap_window_size`` box summed at each of the saved
``trap_sites`` (full-sensor pixels, placed in the frame by its
``roi_x0``/``roi_y0`` header; 0 when missing) and converted to photons. The
counts summed over each group's boxes are the raw pixel values, with no
filtering, offset subtraction or conversion to photons.

Input:
    One image stream: the tweezer camera.

Output:
    A data message (no array) per first image (``index`` 0, or no ``index``),
    published on ``<name>``, carrying ``occupied_counts`` and ``empty_counts``,
    their means per site ``mean_occupied_counts`` and ``mean_empty_counts``
    (NaN when the group is empty), their difference ``mean_atom_counts`` (an
    occupied site's counts less the empty sites' background), the number of
    sites in each group ``occupied_sites`` and ``empty_sites``, the fraction
    of sites occupied
    ``occupancy``, ``sites``, and the input's ``index``, ``timestamp`` and
    ``rid`` when it has them. Without usable sites the counts and occupancy
    are NaN and ``error`` says why.

Properties:
    *   imagestreams: ([str]) input image streams.

Boxes reaching past the frame's edge are cut at it. The saved parameters are
read at most every ``REFRESH_S`` seconds and again on a new ``rid``, and the
last answer is kept while the Experiment Manager cannot be reached.
"""

import time

import numpy as np

from pytweezer.analysis import analysis as an
from pytweezer.analysis.analysis_base import ImageAnalysis, run_analysis
from pytweezer.analysis.array_loading import photons_at_sites
from pytweezer.analysis.trap_sites import to_roi
from pytweezer.experiment.client import (
    ManagerError,
    ManagerUnavailable,
    get_saved_parameters,
)
from pytweezer.logging_utils import get_logger
from pytweezer.servers import DataClient

LOGGER = get_logger("tweezer_occupancy_counts")

REFRESH_S = 10.0
FIRST_IMAGE = 0
TRAP_SITE_KEYS = (
    "trap_sites",
    "trap_window_size",
    "trap_feature_size",
    "trap_threshold",
)
NO_SITES = "No trap sites are saved: run FindTrapSites and accept its proposal"


class TweezerOccupancyCounts(ImageAnalysis):
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
        if data is None or data.ndim != 2:
            return
        if head.get("index", FIRST_IMAGE) != FIRST_IMAGE:
            return
        rid = head.get("rid")
        if rid != self._rid:
            self._rid = rid
            self._saved_asked_at = -np.inf

        out_head = {
            key: head[key] for key in ("index", "timestamp", "rid") if key in head
        }
        saved = self._saved_parameters() or {}
        if any(key not in saved for key in TRAP_SITE_KEYS):
            self.dataq.send(
                out_head
                | dict.fromkeys(
                    (
                        "occupied_counts",
                        "empty_counts",
                        "mean_occupied_counts",
                        "mean_empty_counts",
                        "mean_atom_counts",
                        "occupancy",
                    ),
                    np.nan,
                )
                | {"sites": 0}
                | {"error": NO_SITES},
                None,
            )
            return

        sites = np.rint(
            to_roi(saved["trap_sites"], head.get("roi_y0", 0), head.get("roi_x0", 0))
        ).astype(int)
        sites = {i: (int(row), int(col)) for i, (row, col) in enumerate(sites)}
        window_size = int(saved["trap_window_size"])
        filtered = an.morphological_tophat_high_pass(
            np.asarray(data, dtype=np.float64),
            feature_size=int(saved["trap_feature_size"]),
        )
        occupied = (
            photons_at_sites([filtered], sites, window_size)[0]
            > saved["trap_threshold"]
        )
        counts = an.sum_pixel_values(np.asarray(data), sites, window_size=window_size)
        occupied_counts, empty_counts = counts[occupied], counts[~occupied]
        mean_occupied, mean_empty = (
            mean_or_nan(occupied_counts),
            mean_or_nan(empty_counts),
        )
        self.dataq.send(
            out_head
            | {
                "occupied_counts": float(occupied_counts.sum()),
                "empty_counts": float(empty_counts.sum()),
                "mean_occupied_counts": mean_occupied,
                "mean_empty_counts": mean_empty,
                "mean_atom_counts": mean_occupied - mean_empty,
                "occupied_sites": len(occupied_counts),
                "empty_sites": len(empty_counts),
                "occupancy": float(occupied.mean()),
                "sites": len(sites),
            },
            None,
        )


def mean_or_nan(values):
    return float(values.mean()) if len(values) else np.nan


if __name__ == "__main__":
    run_analysis(TweezerOccupancyCounts)
