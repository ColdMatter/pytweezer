"""Load the Rb tweezers, then rearrange the atoms into a target array, one shot per point."""

from __future__ import annotations

import math

import numpy as np

from pytweezer.analysis import analysis as an
from pytweezer.analysis.array_loading import occupied_sites
from pytweezer.experiment.arguments import Choice, Integer, Number
from pytweezer.experiment.motmaster import MotMaster
from pytweezer.experiment.scan import Point, Scan
from pytweezer.experiments.load_tweezers import LoadTweezers
from pytweezer.helper_scripts.parallel import after, run_parallel

#: HamCam frames the script takes after the occupancy frame.
FOLLOW_UP_FRAMES = 2


def fit_affine(from_xy: np.ndarray, to_xy: np.ndarray) -> tuple[np.ndarray, float]:
    """The least-squares affine map taking ``(N, 2)`` points ``from_xy`` to ``to_xy``.

    Returns:
        The ``(3, 2)`` matrix for :func:`apply_affine`, and the RMS distance
        between ``to_xy`` and the mapped ``from_xy``.

    Raises:
        ValueError: ``from_xy`` are collinear, so they fix no map of the plane.
    """
    from_xy = np.asarray(from_xy, dtype=float)
    to_xy = np.asarray(to_xy, dtype=float)
    matrix, _, rank, _ = np.linalg.lstsq(_homogeneous(from_xy), to_xy, rcond=None)
    if rank < 3:
        raise ValueError(
            "the array's traps are collinear, so they cannot place a target "
            "array off their line on the camera"
        )
    misfit = apply_affine(matrix, from_xy) - to_xy
    return matrix, float(np.sqrt((misfit**2).sum(axis=1).mean()))


def apply_affine(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """``(N, 2)`` points ``xy`` mapped by a :func:`fit_affine` matrix."""
    return _homogeneous(np.asarray(xy, dtype=float)) @ matrix


def _homogeneous(xy: np.ndarray) -> np.ndarray:
    return np.column_stack([xy, np.ones(len(xy))])


class RearrangeTweezers(LoadTweezers):
    """Move the atoms loaded into the array on the SLM into a target array, one shot per point.

    Run after :class:`~pytweezer.experiments.slm_tweezer_array.SlmTweezerArray`
    has put the array on the SLM and
    :class:`~pytweezer.optimisers.find_trap_sites.FindTrapSites` has fitted its
    sites. :meth:`prepare` has the rig build the target array and places its
    sites on the HamCam by the affine map from the array's trap positions to the
    saved sites. Each shot, the rig reads the occupancy frame and moves the atoms
    while MOTMaster, fired ``arm_delay`` after the rig is asked, takes it and
    then two follow-up frames of the target array.

    Each point records the initial sites' ``occupied`` and ``loading``, the
    target index each atom was sent to (``destination``, -1 for none), each
    target's fill in the two follow-up frames (``target_filled_1``/``_2``,
    ``fill_fraction_1``/``_2``), the frames played on the SLM (``n_frames``)
    and the rig's timings in ms. With ``save_raw_images`` the three frames are
    ``rearrangement_images``; with ``save_tweezer_counts`` the raw window sums
    are ``initial_counts`` (frame 0) and ``target_counts`` (frames 1 and 2).
    Once per run it records the initial array's ``array_x_um``/``array_y_um``,
    the fitted ``um_to_px`` (a :func:`fit_affine` matrix taking µm to HamCam
    ``(x, y)`` ROI pixels) and its ``affine_residual_px``, and the target's
    ``target_sites``, ``target_x_um``/``target_y_um``, ``target_weights`` and
    uniformity figures.
    """

    rb = MotMaster(
        "Rb MotMaster",
        script="RbTweezerBasic2026_2_Assemble",
        saved=LoadTweezers.rb.saved,
    )
    analyses = ()

    target_kind = Choice(["kagome", "square"], group="Target array")
    target_rows = Integer(6, min=1, group="Target array", tooltip="square only")
    target_columns = Integer(6, min=1, group="Target array", tooltip="square only")
    target_sites = Integer(70, min=1, group="Target array", tooltip="kagome only")
    target_spacing = Number(
        5e-6, unit="um", scale=1e-6, min=0.1e-6, group="Target array"
    )
    target_angle = Number(
        math.radians(2.5), unit="deg", scale=math.pi / 180, group="Target array"
    )
    target_iterations = Integer(5000, min=1, group="Target array")
    target_damping = Number(0.5, min=0, max=1, group="Target array")
    target_phase_randomness = Number(
        1.0,
        min=0,
        max=1,
        group="Target array",
        tooltip="spread of the initial trap phases",
    )

    d0 = Number(
        0.5e-6,
        unit="um",
        scale=1e-6,
        min=0.01e-6,
        group="Rearrangement",
        tooltip="largest step of any trap between SLM frames",
    )
    profile = Choice(["linear", "minimum_jerk"], group="Rearrangement")
    arm_delay = Number(
        0.1,
        unit="ms",
        scale=1e-3,
        min=0,
        group="Rearrangement",
        tooltip="how long after the rig is asked to rearrange MOTMaster is fired",
    )

    def planned_points(self, scan: Scan) -> list[Point]:
        for axis in scan.axes:
            if axis.argument.startswith("target_"):
                raise ValueError(
                    f"{axis.argument} cannot be scanned: the target array is "
                    "built on the rig before the first point"
                )
        return super().planned_points(scan)

    def prepare(self) -> None:
        super().prepare()
        if self.trap_sites is None:
            raise RuntimeError(self.no_trap_sites_reason)
        self.hamcam.setup_acquisition(1 + FOLLOW_UP_FRAMES, mode="snap")

        array = self.rig.get_array()
        if array is None:
            raise RuntimeError(
                "no array on the SLM; run SlmTweezerArray, then FindTrapSites"
            )
        trap_sites = np.array(list(self.trap_sites.values()))
        array_um = np.column_stack([array["x_um"], array["y_um"]])
        um_to_px, residual_px = fit_affine(array_um, trap_sites[:, ::-1])
        if residual_px > self.trap_window_size / 2:
            raise RuntimeError(
                "the saved trap sites do not fit the rig's array (residual "
                f"{residual_px:.1f} px); run FindTrapSites again and accept its "
                "proposal"
            )

        target = self.rig.prepare_rearrangement(
            kind=self.target_kind,
            rows=self.target_rows,
            columns=self.target_columns,
            sites=self.target_sites,
            spacing_um=self.target_spacing * 1e6,
            angle_deg=math.degrees(self.target_angle),
            iterations=self.target_iterations,
            damping=self.target_damping,
            phase_randomness=self.target_phase_randomness,
            occupancy_sites=trap_sites,
            threshold_photons=self.trap_threshold,
            window_size=self.trap_window_size,
            feature_size=self.trap_feature_size,
            follow_up_frames=FOLLOW_UP_FRAMES,
        )
        target_um = np.column_stack([target["x_um"], target["y_um"]])
        target_xy = apply_affine(um_to_px, target_um)
        self.target_pixels = np.rint(target_xy[:, ::-1]).astype(int)
        outside = self.windows_leaving_roi(self.target_pixels)
        if outside:
            raise RuntimeError(f"target sites {outside} leave the camera's ROI")

        self.record("array_x_um", np.asarray(array["x_um"], dtype=float), unit="um")
        self.record("array_y_um", np.asarray(array["y_um"], dtype=float), unit="um")
        self.record("um_to_px", um_to_px)
        self.record("target_sites", self.target_pixels, unit="px")
        self.record("target_x_um", np.asarray(target["x_um"]), unit="um")
        self.record("target_y_um", np.asarray(target["y_um"]), unit="um")
        self.record("target_weights", np.asarray(target["weights"]))
        self.record("target_uniformity", float(target["uniformity"]))
        self.record("target_min_max_ratio", float(target["min_max_ratio"]))
        self.record("target_mse", float(target["mse"]))
        self.record("affine_residual_px", residual_px, unit="px")

    def run_point(self) -> None:
        self.update_gain()
        shot, _ = run_parallel(
            lambda: self.rig.rearrange(
                d0_um=self.d0 * 1e6, profile=self.profile, header={"rid": self.rid}
            ),
            after(self.arm_delay, self.run_sequences),
        )
        frames = np.asarray(shot["frames"])
        if self.save_raw_images:
            self.record("rearrangement_images", frames)
        occupied = np.asarray(shot["occupied"], dtype=bool)
        self.record("occupied", occupied)
        self.record("loading", float(occupied.mean()))
        self.record("destination", np.asarray(shot["destination"], dtype=int))
        for number, frame in enumerate(frames[1:], start=1):
            filled = occupied_sites(
                frame,
                self.target_pixels,
                window_size=self.trap_window_size,
                feature_size=self.trap_feature_size,
                threshold=self.trap_threshold,
            )
            self.record(f"target_filled_{number}", filled)
            self.record(f"fill_fraction_{number}", float(filled.mean()))
        if self.save_tweezer_counts:
            self.record(
                "initial_counts", self.site_counts(frames[:1])[0], unit="counts"
            )
            target_pixels = dict(enumerate(map(tuple, self.target_pixels)))
            target_counts = [
                an.sum_pixel_values(
                    frame, target_pixels, window_size=self.trap_window_size
                )
                for frame in frames[1:]
            ]
            self.record("target_counts", np.array(target_counts), unit="counts")
        self.record("n_frames", int(shot["n_frames"]))
        for name in (
            "occupancy_ms",
            "calculation_and_upload_ms",
            "total_rearrangement_ms",
        ):
            self.record(name, float(shot[name]), unit="ms")
