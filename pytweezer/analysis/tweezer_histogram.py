"""The tweezer loading histogram of a measurement file.

The per-file counterpart of the live ``tweezer_loading`` analysis, giving the
same ``(header, values)`` pair from what a file recorded, in the header's
``unit``:

* ``first_counts``/``second_counts`` per point, each site's summed raw pixel
  values, with the ``trap_sites`` and ``threshold`` (photons) constants. These
  are histogrammed in counts, or converted with
  ``analysis.convert_counts_to_photons`` when ``unit`` is ``"photons"``;
* ``first_photons``/``second_photons`` per point (files from before raw
  counts), always in photons; or
* the ``hamcam_images`` stack per point, ``(2, H, W)`` (``LoadTweezers``
  without trap sites), always in photons. The sites are found in the mean of
  every top-hat filtered first frame in the file.

Whichever image is histogrammed, the header also gives the imaging survival
from the first image to the second, with both scored against the first
image's threshold.
"""

from dataclasses import dataclass

import numpy as np

from pytweezer.analysis import analysis as an
from pytweezer.analysis.array_loading import (
    find_sites,
    imaging_survival,
    loading_statistics,
    photons_at_sites,
)
from pytweezer.experiment.storage import Measurement
from pytweezer.experiment.summary import display_value

COUNTS = ("first_counts", "second_counts")
PHOTONS = ("first_photons", "second_photons")
FRAMES = "hamcam_images"
RESULTS = {"counts": COUNTS, "photons": PHOTONS, "frames": (FRAMES,)}


@dataclass(frozen=True)
class HistogramSettings:
    """What to histogram, and how to score files that recorded only frames.

    Attributes:
        image: 0 for the first image of each shot, 1 for the second.
        unit: ``"counts"`` or ``"photons"``; a file of recorded counts is
            histogrammed in this unit, any other file in photons.
        group: Index into :func:`point_groups`; ``None`` for every point.
        sites: Trap sites to find (frames only).
        feature_size: Top-hat filter size in pixels (frames only).
        window_size: Side of the box summed at each site in pixels (frames only).
        detection_step: Step in counts by which the site-finding level is
            lowered (frames only).
        fallback_threshold: Occupancy threshold in photons when the fit fails
            and the file recorded none.
    """

    image: int = 0
    unit: str = "counts"
    group: int | None = None
    sites: int = 64
    feature_size: int = 10
    window_size: int = 5
    detection_step: float = 100.0
    fallback_threshold: float = 100.0


def source(measurement: Measurement) -> str | None:
    """``"counts"``, ``"photons"`` or ``"frames"``, by what the file recorded; ``None`` if none."""
    for kind, names in RESULTS.items():
        if all(name in measurement.result_shapes for name in names):
            return kind
    return None


def needed_results(measurement: Measurement) -> list[str]:
    """The results :func:`tweezer_histogram` reads from this file."""
    return list(RESULTS.get(source(measurement), ()))


def point_groups(measurement: Measurement) -> list[tuple[str, np.ndarray]]:
    """``(label, mask over the points run)`` per distinct value of the scan axes.

    Repetitions of a value pool into one group. Without scan axes each point
    is its own group.
    """
    points = measurement.points
    axes = [axis.argument for axis in measurement.scan.axes]
    if not axes:
        indices = points["index"]
        return [(f"point {index}", indices == index) for index in indices]
    keys = list(zip(*(points[axis].tolist() for axis in axes), strict=True))
    groups = []
    for key in dict.fromkeys(keys):
        label = ", ".join(
            f"{axis} = {display_value(measurement, axis, value)}"
            for axis, value in zip(axes, key, strict=True)
        )
        groups.append((label, np.array([k == key for k in keys], dtype=bool)))
    return groups


def tweezer_histogram(
    measurement: Measurement, settings: HistogramSettings
) -> tuple[dict, np.ndarray | None]:
    """The live analysis's header and ``(shots, sites)`` values for the selected shots.

    When there are no shots or the sites cannot be found, the values are
    ``None`` and the header's ``"error"`` says why.

    Raises:
        KeyError: If the file recorded no counts, photons or frames, or
            ``measurement`` was loaded without them.
    """
    kind = source(measurement)
    if kind is None:
        raise KeyError("this file recorded no tweezer images, counts or photons")
    if settings.group is None:
        mask = np.ones(len(measurement.points["index"]), dtype=bool)
    else:
        mask = point_groups(measurement)[settings.group][1]
    unit = settings.unit if kind == "counts" else "photons"
    head = {"shots": 0, "sites": settings.sites, "source": kind, "unit": unit}

    if kind in ("counts", "photons"):
        images = [
            np.asarray(measurement.results[name], float)[mask].reshape(
                -1, len(measurement.constants["trap_sites"])
            )
            for name in RESULTS[kind]
        ]
        sites = {
            i: (int(row), int(col))
            for i, (row, col) in enumerate(measurement.constants["trap_sites"])
        }
        fallback_threshold = float(
            measurement.constants.get("threshold", settings.fallback_threshold)
        )
        if kind == "counts" and unit == "photons":
            images = [an.convert_counts_to_photons(image) for image in images]
        elif kind == "counts":
            fallback_threshold = float(an.convert_photons_to_counts(fallback_threshold))
    else:
        if not mask.any():
            return head | {"error": "No shots in this selection"}, None
        stack = measurement.results[FRAMES]
        firsts = _filtered(stack[:, 0], settings)
        sites = find_sites(
            np.mean(firsts, axis=0), settings.sites, settings.detection_step
        )
        if sites is None:
            return head | {
                "error": f"Found no {settings.sites} trap sites in the mean first image"
            }, None
        images = [
            photons_at_sites(
                [frame for frame, keep in zip(frames, mask, strict=True) if keep],
                sites,
                settings.window_size,
            )
            for frames in (firsts, _filtered(stack[:, 1], settings))
        ]
        fallback_threshold = settings.fallback_threshold

    photons = images[settings.image]
    head |= {"shots": len(photons), "sites": len(sites)}
    if not len(photons):
        return head | {"error": "No shots in this selection"}, None
    head |= loading_statistics(photons, sites, fallback_threshold)
    first_threshold = (
        head["threshold"]
        if settings.image == 0
        else loading_statistics(images[0], sites, fallback_threshold)["threshold"]
    )
    return head | imaging_survival(*images, first_threshold), photons


def _filtered(frames, settings):
    return [
        an.morphological_tophat_high_pass(
            np.asarray(frame, dtype=np.float64), feature_size=settings.feature_size
        )
        for frame in frames
    ]
