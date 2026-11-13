from __future__ import annotations

from typing import ClassVar

import numpy as np

from pytweezer.analysis import analysis as an
from pytweezer.analysis.trap_sites import default_geometry, fit_trap_sites
from pytweezer.experiment.arguments import Bool, Integer, Number
from pytweezer.experiment.optimisers import SEARCH, Fit
from pytweezer.experiments.load_tweezers import LoadTweezers


class FindTrapSites(Fit, LoadTweezers):
    """Find the trap sites by fitting the array's geometry to the atoms, and propose them.

    Takes ``calibration_shots`` with the tweezers off during the background
    drop, and fits the geometry of the array on the SLM (or, when
    the SLM does not know it, a ``default_rows`` x ``default_columns`` grid
    ``default_spacing`` apart) to their mean background-subtracted, top-hat
    filtered image, free in position, rotation and scale (see
    :mod:`pytweezer.analysis.trap_sites`). It then fits the occupancy
    threshold at those sites, and proposes both for every tweezer measurement
    to use.
    """

    proposal_units: ClassVar[dict[str, str]] = {
        "trap_sites": "px",
        "trap_threshold": "photons",
        "trap_window_size": "px",
        "trap_feature_size": "px",
        "trap_geometry": "µm",
    }
    proposal_group = "Tweezer array"

    save_raw_images = Bool(
        False,
        group="Camera",
        tooltip="record every shot's camera frames as hamcam_images",
    )
    calibration_shots = Integer(100, min=1, group=SEARCH)
    window_size = Integer(
        5, min=1, unit="px", group=SEARCH, tooltip="box summed per site"
    )
    feature_size = Integer(
        10, min=1, unit="px", group=SEARCH, tooltip="top-hat filter size"
    )
    initial_threshold = Number(
        100,
        unit="photons",
        group=SEARCH,
        tooltip="starting guess for the occupancy threshold fit",
    )
    default_rows = Integer(8, min=1, group="Default array")
    default_columns = Integer(8, min=1, group="Default array")
    default_spacing = Number(5e-6, unit="um", scale=1e-6, min=0, group="Default array")

    def saved_trap_sites(self, saved):
        return None, ""  # it finds them

    def prepare(self) -> None:
        super().prepare()
        array = self.slm.get_array()
        if array:
            self.geometry = np.column_stack([array["x_um"], array["y_um"]])
        else:
            self.geometry = default_geometry(
                self.default_rows, self.default_columns, self.default_spacing * 1e6
            )
        self.record("sites_from", "slm" if array else "default")

    def take_calibration_shots(self) -> tuple[list, list]:
        """``calibration_shots`` shots with the tweezers off in the background drop.

        Returns their first and second frames. With ``save_raw_images`` the
        frames are recorded as ``hamcam_images``, ``(shots, 2, height, width)``.
        """
        stacks = np.array(
            [
                self.take_shot(
                    BackgroundDrop=1, tTweezerSetVVA0=0.0, tTweezerSetVVA1=0.0
                )
                for _ in range(self.calibration_shots)
            ]
        )
        if self.save_raw_images:
            self.record("hamcam_images", stacks)
        return list(stacks[:, 0]), list(stacks[:, 1])

    def fit(self) -> dict:
        firsts, backgrounds = self.take_calibration_shots()
        average = an.tweezer_show_bg_subtracted(firsts, backgrounds, show=False)
        self.record("calibration_image", average)
        filtered = an.morphological_tophat_high_pass(
            np.asarray(average, dtype=float), feature_size=self.feature_size
        )
        found = fit_trap_sites(filtered, self.geometry, window_size=self.window_size)
        self.record("roi_sites", found.sites, unit="px")
        self.record("scale", found.scale, unit="px/µm")
        self.record("angle", found.angle_deg, unit="deg")
        self.record("x0", found.x0, unit="px")
        self.record("y0", found.y0, unit="px")
        whole_pixels = {
            index: (round(y), round(x)) for index, (y, x) in enumerate(found.sites)
        }
        _, _, threshold, fidelity = an.get_array_loading_statistics(
            self.top_hat(firsts, self.feature_size),
            whole_pixels,
            threshold=self.initial_threshold,
            window_size=self.window_size,
            binning=60,
            show_histogram=False,
            threshold_detection=True,
            verbose=False,
        )
        self.record("threshold", float(threshold), unit="photons")
        self.record("calibration_fidelity", float(fidelity))
        return {
            "trap_sites": found.sites + self.roi_origin[::-1],
            "trap_threshold": float(threshold),
            "trap_window_size": self.window_size,
            "trap_feature_size": self.feature_size,
            "trap_geometry": self.geometry,
        }
