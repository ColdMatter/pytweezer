"""Light at the tweezer sites of a ``MergeCaFTweezer`` file: ``caf_tweezer_sites``.

The sites are the SLM array the file recorded (``slm_arrays``, with each
point's ``slm_array``) placed on the HamCam by the run's ``site_*`` arguments,
the ROI origin the camera set (``hamcam_roi_origin``) and its
``hamcam_binning``. Traps whose
apertures would overlap are one site, at their mean position, since their light
can't be told apart. Each shot's ``hamcam_image``, less its
``hamcam_background`` when the file has one, is scored at the sites against
empty positions of the same frame
(:mod:`pytweezer.analysis.site_photometry`). Files from before these were
recorded can't be analysed.
"""

import math
from dataclasses import dataclass

import numpy as np

from pytweezer.analyses.default.tweezer_histogram import point_groups
from pytweezer.analysis.site_photometry import merge_overlapping, score_sites
from pytweezer.analysis.trap_sites import geometry_to_pixels
from pytweezer.experiment.storage import Measurement

IMAGE = "hamcam_image"
BACKGROUND = "hamcam_background"
ARRAY_INDEX = "slm_array"
ARRAYS = "slm_arrays"
ROI_ORIGIN = "hamcam_roi_origin"
CALIBRATION = ("site_centre_x", "site_centre_y", "site_scale", "site_angle")
BINNING = "hamcam_binning"
#: Score above which a shot counts as showing light at a site.
BRIGHT_Z = 3.0


@dataclass(frozen=True)
class SiteSettings:
    """Which shots to analyse and the photometry's radii in frame (binned) pixels.

    Attributes:
        group: Index into :func:`point_groups`; ``None`` for every point.
        aperture: Radius of the disc summed at each site.
        ring_inner: Inner radius of the background ring.
        ring_outer: Outer radius of the background ring.
    """

    group: int | None = None
    aperture: float = 4.0
    ring_inner: float = 8.0
    ring_outer: float = 12.0


@dataclass(frozen=True)
class SiteAnalysis:
    """The sites, their light shot by shot, and the simulated array they come from.

    Attributes:
        shots: Shots analysed.
        background_subtracted: Whether each frame had its background frame subtracted.
        mean_image: Mean analysed frame, ``(H, W)``.
        sites: ``(sites, 2)`` positions in the frame, ``(y, x)`` pixels.
        site_traps: For each site, the indices into ``traps_um`` of the traps it holds.
        counts: ``(shots, sites)`` signal at each site, counts.
        z: ``(shots, sites)`` score at each site.
        empty_z: Scores at every empty position of every shot, flat.
        bright_by_chance: Shots per site expected above :data:`BRIGHT_Z` by chance.
        combined: Per shot, the sites' scores summed, unit variance if independent.
        combined_chance: Mean ``combined`` over the shots for random sets of empty positions.
        offset: Per shot, the frame's median, counts.
        trap_plane: Simulated intensity in the trap plane, row index along y.
        trap_plane_extent: ``(x_min, x_max, y_min, y_max)`` of ``trap_plane``, µm.
        traps_um: ``(sites, 2)`` trap positions ``(x, y)`` in the focal plane, µm.
    """

    shots: int
    background_subtracted: bool
    mean_image: np.ndarray
    sites: np.ndarray
    site_traps: tuple[tuple[int, ...], ...]
    counts: np.ndarray
    z: np.ndarray
    empty_z: np.ndarray
    bright_by_chance: float
    combined: np.ndarray
    combined_chance: np.ndarray
    offset: np.ndarray
    trap_plane: np.ndarray
    trap_plane_extent: tuple[float, float, float, float]
    traps_um: np.ndarray


def missing(measurement: Measurement) -> str | None:
    """What the file lacks for :func:`caf_tweezer_sites`, or ``None`` if nothing."""
    if IMAGE not in measurement.result_shapes:
        return "This file recorded no HamCam images."
    if (
        ARRAY_INDEX not in measurement.result_shapes
        or ARRAYS not in measurement.constants
    ):
        return "This file predates recording the SLM arrays."
    if ROI_ORIGIN not in measurement.constants:
        return "This file predates recording the HamCam ROI origin."
    if any(measurement.arguments.get(name) is None for name in CALIBRATION):
        return "This file has no CaF site calibration (site_* arguments)."
    return None


def needed_results(measurement: Measurement) -> list[str]:
    """The results :func:`caf_tweezer_sites` reads from this file."""
    names = [IMAGE, ARRAY_INDEX]
    if BACKGROUND in measurement.result_shapes:
        names.append(BACKGROUND)
    return names


def site_label(traps: tuple[int, ...]) -> str:
    """A site's name from the traps it holds: ``"2"``, or ``"0+1+2+3"`` for merged traps."""
    return "+".join(str(trap) for trap in traps)


def site_positions(measurement: Measurement, array: dict) -> np.ndarray:
    """``(sites, 2)`` ``(y, x)`` pixels of ``array``'s traps in the file's HamCam frames."""
    arguments = measurement.arguments
    geometry = np.column_stack([array["x_um"], array["y_um"]])
    sensor = geometry_to_pixels(
        geometry,
        arguments["site_scale"] * 1e-6,
        math.degrees(arguments["site_angle"]),
        arguments["site_centre_x"],
        arguments["site_centre_y"],
    )
    x0, y0 = np.asarray(measurement.constants[ROI_ORIGIN], float)
    # Files from before the argument existed were unbinned.
    binning = measurement.arguments.get(BINNING) or 1
    # Binned pixel j spans sensor pixels j*binning .. (j+1)*binning - 1 from the origin.
    return (sensor - [y0, x0] - (binning - 1) / 2) / binning


def caf_tweezer_sites(measurement: Measurement, settings: SiteSettings) -> SiteAnalysis:
    """Score the selected shots at the sites of the SLM array they were taken with.

    Raises:
        ValueError: If the selection has no shots or spans arrays with different
            trap positions, or a site's background ring leaves the frame.
    """
    n_points = len(measurement.points["index"])
    if settings.group is None:
        mask = np.ones(n_points, dtype=bool)
    else:
        mask = point_groups(measurement)[settings.group][1]
    if not mask.any():
        raise ValueError("no shots in this selection")
    arrays = measurement.constants[ARRAYS]["arrays"]
    used = sorted({int(i) for i in measurement.results[ARRAY_INDEX][mask]})
    positions = {
        (
            tuple(arrays[i]["x_um"]),
            tuple(arrays[i]["y_um"]),
            tuple(arrays[i]["blaze_um"]),
        )
        for i in used
    }
    if len(positions) > 1:
        raise ValueError(
            "these shots were taken with differently placed SLM arrays; "
            "pick one scan value in Shots from"
        )
    array = arrays[used[0]]

    frames = np.asarray(measurement.results[IMAGE][mask], float)
    subtracted = BACKGROUND in measurement.results
    if subtracted:
        frames = frames - np.asarray(measurement.results[BACKGROUND][mask], float)
    sites, site_traps = merge_overlapping(
        site_positions(measurement, array), settings.aperture
    )
    scores = score_sites(
        frames, sites, settings.aperture, settings.ring_inner, settings.ring_outer
    )

    blaze_x, blaze_y = array["blaze_um"]
    grid_x = np.asarray(array["grid_x_um"]) + blaze_x
    grid_y = np.asarray(array["grid_y_um"]) + blaze_y
    half_x, half_y = (grid_x[1] - grid_x[0]) / 2, (grid_y[1] - grid_y[0]) / 2
    mean_image = frames.mean(axis=0)
    return SiteAnalysis(
        shots=len(frames),
        background_subtracted=subtracted,
        mean_image=mean_image,
        sites=sites,
        site_traps=tuple(site_traps),
        counts=scores.counts,
        z=scores.z,
        empty_z=scores.empty_z.ravel(),
        bright_by_chance=float((scores.empty_z > BRIGHT_Z).mean() * len(frames)),
        combined=scores.combined(),
        combined_chance=scores.combined_chance(),
        offset=np.median(frames.reshape(len(frames), -1), axis=1),
        trap_plane=np.asarray(array["simulated_trap_plane"], float),
        trap_plane_extent=(
            grid_x[0] - half_x,
            grid_x[-1] + half_x,
            grid_y[0] - half_y,
            grid_y[-1] + half_y,
        ),
        traps_um=np.column_stack(
            [np.asarray(array["x_um"]) + blaze_x, np.asarray(array["y_um"]) + blaze_y]
        ),
    )
