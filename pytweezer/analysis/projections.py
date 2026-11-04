"""Sum an image along x and y and publish the two 1-D profiles with Gaussian fits.

Input:
    One image stream (typically a ROI crop from ``roi_slice.py`` or the output
    of ``shot_background.py``).

Output:
    Two *data* streams (published on the Data hub, not the Image hub):

    * ``<name>_x`` -- profile summed down the columns (intensity vs x),
    * ``<name>_y`` -- profile summed across the rows (intensity vs y).

    Each is sent as a 3xN ``[coords, sums, fit_curve]`` array, so ``live_plot.py``
    and ``gaussian_fit.py`` can read ``A[0]`` as the axis and ``A[1]`` as the
    data. The header is the input header plus ``center``, ``sigma``,
    ``amplitude``, ``offset`` and ``success`` of the 1-D Gaussian fit.

Properties:
    * imagestreams: ([str]) input image streams.

Coordinates follow :mod:`pytweezer.analysis.cloud_fit`: pixel ``i`` sits at
``(origin + i + 0.5) * resolution`` from the header's ``_offset`` and
``_imgresolution``, in pixels (resolution 1) without ``_imgresolution``.

The fit is only trusted when it is a clear peak: ``success`` and the fit curve
are withheld (``success`` False, ``fit_curve`` all NaN) when the least-squares
fit failed, its centre is outside the profile, its width is under a pixel or
over the profile's length, or its amplitude is below ``MIN_FIT_SNR`` times the
rms residual. That keeps an empty
frame from publishing a fit of its noise.

This consumes images but emits 1-D data, so (like ``find_tweezer_atoms.py``) it
subscribes with the inherited :class:`ImageAnalysis` client and publishes on a
separate :class:`DataClient`.

The sums use ``dtype=float64`` so the accumulator is wide enough to avoid
integer overflow on raw camera counts, without up-casting the whole frame first.
"""

import numpy as np

from pytweezer.analysis.analysis_base import ImageAnalysis, run_analysis
from pytweezer.analysis.cloud_fit import fit_profile, frame_geometry, profile_curve
from pytweezer.logging_utils import get_logger
from pytweezer.servers import DataClient

LOGGER = get_logger(__name__)

MIN_FIT_SNR = 6.0


def _trusted_curve(coords, values):
    """Return ``(fit, curve)`` for a clear peak, else ``(None, all-NaN curve)``."""
    try:
        fit = fit_profile(coords, values)
        curve = profile_curve(coords, fit)
    except Exception:
        LOGGER.warning("profile fit failed", exc_info=True)
        return None, np.full(coords.shape, np.nan)
    residual_rms = float(np.sqrt(np.mean((values - curve) ** 2)))
    spacing = np.ptp(coords) / max(coords.size - 1, 1)
    clear_peak = (
        fit.success
        and np.isfinite(curve).all()
        and fit.amplitude > MIN_FIT_SNR * residual_rms
        and spacing <= fit.sigma <= np.ptp(coords)
        and coords[0] <= fit.center <= coords[-1]
    )
    if not clear_peak:
        return fit, np.full(coords.shape, np.nan)
    return fit, curve


class Projections(ImageAnalysis):
    def __init__(self, name):
        super().__init__(name)
        self.dataq = DataClient(name.split("/")[-1])

    def process(self, head, data):
        if data is None or data.ndim != 2:
            return

        h, w = data.shape
        resolution, origin = frame_geometry(head)
        res_x, res_y = resolution if resolution is not None else (1.0, 1.0)

        x_values = data.sum(axis=0, dtype=np.float64)
        y_values = data.sum(axis=1, dtype=np.float64)
        x_coords = (origin[0] + np.arange(w) + 0.5) * res_x
        y_coords = (origin[1] + np.arange(h) + 0.5) * res_y

        for coords, values, channel in (
            (x_coords, x_values, "_x"),
            (y_coords, y_values, "_y"),
        ):
            fit, curve = _trusted_curve(coords, values)
            out_head = dict(head)
            out_head.update(
                {
                    "center": float(fit.center) if fit else float("nan"),
                    "sigma": float(fit.sigma) if fit else float("nan"),
                    "amplitude": float(fit.amplitude) if fit else float("nan"),
                    "offset": float(fit.offset) if fit else float("nan"),
                    "success": bool(not np.isnan(curve).any()),
                }
            )
            self.dataq.send(
                out_head, np.vstack([coords, values, curve]), channel=channel
            )
        return


if __name__ == "__main__":
    run_analysis(Projections)
