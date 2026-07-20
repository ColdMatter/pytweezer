"""Fit a 1-D Gaussian to an incoming data trace and publish the parameters.

Input:
    One data stream carrying either a 1-D array (``y`` only, x is the index) or
    a 2xN ``[x, y]`` array -- e.g. a profile from ``projections.py``.

Output:
    A data message (no array) whose header carries the fit parameters:
    ``amplitude``, ``center``, ``sigma``, ``fwhm``, ``offset``, plus ``method``
    and ``success``. Published on ``<name>`` (the process's own stream).

Properties:
    *   datastreams: ([str]) input data streams.
    *   method: (str) ``'lsq'`` (default, accurate least-squares fit) or
        ``'moments'`` (a faster closed-form estimate -- centre is reliable but
        the width is biased high by baseline noise in the tails; use only when
        raw speed matters more than an accurate sigma).

The ``'lsq'`` path seeds :func:`scipy.optimize.curve_fit` with the
:func:`gaussian_moments` estimate so it converges in a few iterations, and caps
``maxfev`` so a pathological trace cannot stall the analysis loop.
"""

import numpy as np

from pytweezer.analysis.analysis_base import DataAnalysis, run_analysis
from pytweezer.servers import PropertyAttribute


def gaussian(x, amplitude, center, sigma, offset):
    """Gaussian model: ``offset + amplitude * exp(-0.5*((x-center)/sigma)**2)``."""
    return offset + amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def gaussian_moments(x, y):
    """Closed-form Gaussian estimate from the data's moments (fast, no fit).

    Returns a params dict; ``success`` is False when the trace carries no
    positive signal above baseline (a flat or empty profile).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Robust baseline so noise/background does not bias the width estimate.
    offset = float(np.percentile(y, 10)) if y.size else 0.0
    weights = np.clip(y - offset, 0.0, None)
    total = float(weights.sum())

    if total <= 0.0 or x.size == 0:
        peak = float(x[int(np.argmax(y))]) if x.size else 0.0
        return {
            "amplitude": 0.0,
            "center": peak,
            "sigma": 0.0,
            "fwhm": 0.0,
            "offset": offset,
            "success": False,
        }

    center = float((weights * x).sum() / total)
    variance = float((weights * (x - center) ** 2).sum() / total)
    sigma = float(np.sqrt(max(variance, 0.0)))
    amplitude = float(y.max() - offset)
    return {
        "amplitude": amplitude,
        "center": center,
        "sigma": sigma,
        "fwhm": 2.3548200450309493 * sigma,  # 2*sqrt(2*ln2) * sigma
        "offset": offset,
        "success": True,
    }


class GaussianFit(DataAnalysis):
    _method = PropertyAttribute("method", "lsq")

    @staticmethod
    def _split_xy(data):
        """Return ``(x, y)`` from a 1-D (y-only) or 2xN ([x, y]) array."""
        arr = np.asarray(data)
        if arr.ndim == 2 and arr.shape[0] >= 2:
            return arr[0], arr[1]
        arr = arr.ravel()
        return np.arange(arr.size), arr

    def _fit_lsq(self, x, y, seed):
        """Refine the moment estimate with least squares; fall back on failure."""
        if not seed["success"] or seed["sigma"] <= 0:
            return seed
        # Imported lazily so the fast 'moments' path never pays the SciPy import.
        from scipy.optimize import curve_fit

        p0 = [seed["amplitude"], seed["center"], seed["sigma"], seed["offset"]]
        try:
            popt, _ = curve_fit(gaussian, x, y, p0=p0, maxfev=2000)
        except Exception as error:
            print(self.name, ": lsq fit failed, using moments:", error)
            return seed

        amplitude, center, sigma = float(popt[0]), float(popt[1]), abs(float(popt[2]))
        return {
            "amplitude": amplitude,
            "center": center,
            "sigma": sigma,
            "fwhm": 2.3548200450309493 * sigma,
            "offset": float(popt[3]),
            "success": True,
        }

    def process(self, head, data):
        if data is None:
            return None

        x, y = self._split_xy(data)
        params = gaussian_moments(x, y)
        if self._method == "lsq":
            params = self._fit_lsq(x, y, params)
        params["method"] = self._method

        out_head = dict(head)
        out_head.update(params)
        # Send params only (no array): DataClient publishes the dict alone.
        return out_head, None


if __name__ == "__main__":
    run_analysis(GaussianFit)
