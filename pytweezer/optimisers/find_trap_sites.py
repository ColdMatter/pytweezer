from __future__ import annotations

from typing import ClassVar

import numpy as np

from pytweezer.analysis.array_loading import TrapSites, top_hat_image
from pytweezer.analysis.trap_sites import default_geometry, fit_trap_sites
from pytweezer.experiment.arguments import Bool, Integer, Number
from pytweezer.experiment.optimisers import SEARCH, Fit
from pytweezer.experiments.load_tweezers import LoadTweezers


class FindTrapSites(Fit, LoadTweezers):
    """Find the trap sites by fitting the array's geometry to the atoms, and propose them.

    Takes ``calibration_shots`` with no drop, and fits the geometry of the
    array the rig has on the SLM (or, when it does not know, a
    ``default_rows`` x ``default_columns`` grid ``default_spacing`` apart) to
    the mean of their first images, top-hat filtered, free in position,
    rotation and scale (see :mod:`pytweezer.analysis.trap_sites`). It then
    fits the occupancy threshold at those sites, and proposes both for every
    tweezer measurement to use.
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

    def prepare(self) -> None:
        super().prepare()
        array = self.rig.get_array()
        if array:
            self.geometry = np.column_stack([array["x_um"], array["y_um"]])
        else:
            self.geometry = default_geometry(
                self.default_rows, self.default_columns, self.default_spacing * 1e6
            )
        self.record("sites_from", "rig" if array else "default")

    def saved_trap_sites(self, saved):
        return None, ""  # it finds them

    def take_calibration_shots(self) -> list:
        """``calibration_shots`` shots with no drop; their first frames.

        With ``save_raw_images`` both frames are recorded as
        ``hamcam_images``, ``(shots, 2, height, width)``.
        """
        stacks = []
        for _ in range(self.calibration_shots):
            mark = self.hamcam.start_shot({"rid": self.rid})
            self.run_motmaster_sequence(BackgroundDrop=0)
            stacks.append(self.hamcam.wait_frames(mark, 2))
        stacks = np.array(stacks)
        if self.save_raw_images:
            self.record("hamcam_images", stacks)
        return list(stacks[:, 0])

    def fit(self) -> dict:
        firsts = self.take_calibration_shots()
        average = np.mean(firsts, axis=0)
        self.record("calibration_image", average)
        found = fit_trap_sites(
            top_hat_image(average, self.feature_size),
            self.geometry,
            window_size=self.window_size,
        )
        self.record("roi_sites", found.sites, unit="px")
        self.record("scale", found.scale, unit="px/µm")
        self.record("angle", found.angle_deg, unit="deg")
        self.record("x0", found.x0, unit="px")
        self.record("y0", found.y0, unit="px")
        whole_pixels = {
            index: (round(y), round(x)) for index, (y, x) in enumerate(found.sites)
        }
        calibration_sites = TrapSites(
            whole_pixels, self.window_size, self.feature_size, self.initial_threshold
        )
        calibration = calibration_sites.fit_loading(firsts)
        self.record("threshold", calibration.threshold, unit="photons")
        self.record("calibration_fidelity", calibration.fidelity)
        return {
            "trap_sites": found.sites + self.roi_origin[::-1],
            "trap_threshold": calibration.threshold,
            "trap_window_size": self.window_size,
            "trap_feature_size": self.feature_size,
            "trap_geometry": self.geometry,
        }
