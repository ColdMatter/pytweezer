"""Standalone compiled white-tophat high-pass filter for tweezer-array images.

This package wraps the compiled C++ extension ``fast_tophat_filter`` (built from
``bindings.cpp`` / ``tophat_filter.hpp`` in this folder) and exposes two
functions that are drop-in replacements for
``scipy.ndimage.white_tophat(image, size=N, mode='reflect')`` and the project's
``morphological_tophat_high_pass`` -- bit-for-bit identical, ~2x faster.

    from cpp_implementations.filter import white_tophat, batch_white_tophat

    filtered = white_tophat(image, 10)        # one (H, W) frame
    filtered = batch_white_tophat(stack, 10)  # whole (n, H, W) stack, one call

If the extension has not been built yet, importing raises a clear error telling
you to run ``make`` in this directory.
"""

from __future__ import annotations

try:
    from . import fast_tophat_filter as _ext
except ImportError as exc:  # pragma: no cover - build-time guidance
    raise ImportError(
        "The compiled extension 'fast_tophat_filter' is not built. "
        "Run `make` inside cpp_implementations/filter/ "
        "(needs a C++17 compiler and pybind11 in the project venv)."
    ) from exc

white_tophat = _ext.white_tophat
batch_white_tophat = _ext.batch_white_tophat

__all__ = ["white_tophat", "batch_white_tophat"]
