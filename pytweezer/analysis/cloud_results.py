"""The cloud fits a measurement file recorded, for the Results tab.

Reads what :meth:`~pytweezer.experiment.camera_experiment.CameraExperiment.record_cloud_fit`
records per point for each camera: ``<camera>_<field>`` for every field of
:data:`~pytweezer.analysis.cloud_fit.FIELDS`, ``<camera>_<field>_error`` for
each of :data:`~pytweezer.analysis.cloud_fit.ERROR_FIELDS`, and optionally
``<camera>_mean_image``.

The file does not record where the mean image sits in the camera frame, so
:func:`point_image` refits it in its own pixel coordinates: the same fit of the
same image as the one recorded, placed where the image is drawn.
"""

import numpy as np

from pytweezer.analysis.cloud_fit import (
    ERROR_FIELDS,
    FIELDS,
    CloudFit,
    fit_cloud,
)
from pytweezer.experiment.storage import Measurement
from pytweezer.experiment.summary import aggregate

SIZES = ("sigma_x", "sigma_y")


def cloud_cameras(measurement: Measurement) -> list[str]:
    """The cameras whose cloud fits the file recorded, in recording order."""
    return [
        name.removesuffix("_sigma_x")
        for name, shape in measurement.result_shapes.items()
        if name.endswith("_sigma_x") and shape == ()
    ]


def scanned_arguments(measurement: Measurement) -> list[str]:
    """The numeric scanned arguments, in scan order."""
    return [
        axis.argument
        for axis in measurement.scan.axes
        if np.asarray(measurement.points[axis.argument]).dtype.kind in "biuf"
    ]


def needed_results(measurement: Measurement) -> list[str]:
    """Every camera's recorded fit fields and their errors; no mean images."""
    names = []
    for camera in cloud_cameras(measurement):
        names += [f"{camera}_{field}" for field in FIELDS]
        names += [f"{camera}_{field}_error" for field in ERROR_FIELDS]
    return [name for name in names if name in measurement.result_shapes]


def cloud_sizes(
    measurement: Measurement, camera: str, axis: str
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``{"sigma_x": (axis values, mean, error), "sigma_y": ...}`` against ``axis``.

    Points sharing a value of ``axis`` are pooled: the mean and standard error
    over those whose fit succeeded. A value with one such point keeps that
    point's recorded per-frame error (NaN when it had a single frame).

    Args:
        measurement: Loaded with the camera's ``sigma_x``, ``sigma_y`` and
            their errors.
        camera: As :func:`cloud_cameras` gives.
        axis: A ``/points`` column, such as a scanned argument or ``"index"``.
    """
    x = np.asarray(measurement.points[axis])
    sizes = {}
    for field in SIZES:
        y = np.asarray(measurement.results[f"{camera}_{field}"], dtype=float)
        recorded_error = np.asarray(
            measurement.results[f"{camera}_{field}_error"], dtype=float
        )
        empty = (x[:0], np.empty(0), np.empty(0), np.empty(0, dtype=int))
        xs, means, errors, counts = next(iter(aggregate(x, y).values()), empty)
        for i in np.flatnonzero(counts == 1):
            errors[i] = recorded_error[(x == xs[i]) & np.isfinite(y)][0]
        sizes[field] = (xs, means, errors)
    return sizes


def point_image(
    measurement: Measurement, camera: str, index: int, mean_image: np.ndarray | None
) -> tuple[np.ndarray | None, CloudFit]:
    """Point ``index``'s mean image and its cloud fit.

    With an image, the fit is in the image's pixel coordinates (pixel
    ``(row, col)`` centred at ``x = col + 0.5``, ``y = row + 0.5``), so it
    overlays the image drawn row-major. The image is ``None`` when the file
    recorded none for this point; the fit is then the recorded one, in the
    file's units.

    Args:
        measurement: Loaded with the camera's fields.
        camera: As :func:`cloud_cameras` gives.
        index: Row of the point in ``measurement.points``.
        mean_image: The point's recorded ``<camera>_mean_image``, read on its
            own as it is large; ``None`` if the file has none.
    """
    image = None if mean_image is None else np.asarray(mean_image)
    if image is not None and np.isfinite(image).any():
        return image, fit_cloud(image)
    values = {
        field: float(measurement.results[f"{camera}_{field}"][index])
        for field in FIELDS
    }
    return None, CloudFit(
        **values,
        units=measurement.units.get(f"{camera}_x0", ""),
        success=bool(np.isfinite(values["sigma_x"])),
    )
