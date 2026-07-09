## sum_pixel_values_cpp

Pybind11 drop-in replacement for `pytweezer.analysis.analysis.sum_pixel_values`.

**Requirements:** C++17 compiler (MSVC on Windows), `pybind11`, `numpy`.

**Build (from this directory, in the project venv):**
```
pip install pybind11 numpy

# pick one - both compile the .pyd, pip just also installs it for you:
pip install .                              # builds + installs into site-packages
python setup.py build_ext --inplace        # builds .pyd here only, needs cwd/sys.path
```

**Usage:**
```python
import sum_pixel_values_cpp as ps
pixel_sums = ps.sum_pixel_values(image, grid_positions, grid_shape, window_size=3)
```

`image` must be 2-D `uint16`. `grid_positions` is a `{(i,j): (y,x)}` dict.
