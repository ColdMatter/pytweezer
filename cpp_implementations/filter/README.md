# filter — standalone compiled white-tophat filter

A self-contained, compiled C++ version of the **white-tophat high-pass filter** —
the background-removal step that runs on every atom-array frame before trap-site
readout. This folder is just the fast filter, ready to drop into the
rearrangement hot path.

It is a drop-in replacement for `scipy.ndimage.white_tophat(image, size=N,
mode='reflect')` and the project's `morphological_tophat_high_pass`:
**bit-for-bit identical** output (max diff `0.0`, verified), **~2× faster** than
scipy's Cython because it uses an O(n) van Herk / Gil-Werman sliding min/max that
does not slow down as the feature size grows.

## Use

```python
from cpp_implementations.filter import white_tophat, batch_white_tophat

filtered = white_tophat(image, 10)        # one (H, W) frame
filtered = batch_white_tophat(stack, 10)  # whole (n, H, W) stack, one call
```

Accepts float32/float64; returns the same dtype. `batch_white_tophat` filters the
whole stack in a single C call (no Python loop), reusing scratch across frames.

The prebuilt extension (`fast_tophat_filter.cpython-312-darwin.so`) ships in this
folder, so the import above works with no build step on this machine.

## Build / rebuild

Needs a C++17 compiler (clang/gcc) and `pybind11` in the project venv.

```bash
cd cpp_implementations/filter
make            # builds fast_tophat_filter*.so
make clean
```

Override the interpreter with `make PY=/path/to/python`. On Windows build the
module with pybind11's standard recipe against `bindings.cpp`.

## Files

| File | Role |
|---|---|
| `tophat_filter.hpp` | Header-only algorithm: van Herk separable min/max + white tophat (single + batch). |
| `bindings.cpp` | pybind11 wrapper → `fast_tophat_filter` (`white_tophat`, `batch_white_tophat`). |
| `__init__.py` | Python package re-exporting the two functions with a build-missing error message. |
| `Makefile` | Builds the extension. |
| `fast_tophat_filter.cpython-312-darwin.so` | Prebuilt extension (CPython 3.12, macOS arm64). |

