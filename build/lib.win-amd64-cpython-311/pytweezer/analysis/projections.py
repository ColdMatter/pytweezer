"""Sum an image along x and y and publish the two 1-D profiles.

Input:
    One image stream (typically a ROI crop from ``roi_slice.py``).

Output:
    Two *data* streams (published on the Data hub, not the Image hub):
      * ``<name>_x`` -- profile summed down the columns (intensity vs x),
      * ``<name>_y`` -- profile summed across the rows (intensity vs y).
    Each is sent as a 2xN ``[coords, values]`` array so ``live_plot.py`` and
    ``gaussian_fit.py`` can read ``A[0]`` as the axis and ``A[1]`` as the data.

Properties:
    *   imagestreams: ([str]) input image streams.

This consumes images but emits 1-D data, so (like ``find_tweezer_atoms.py``) it
subscribes with the inherited :class:`ImageAnalysis` client and publishes on a
separate :class:`DataClient`.

The sums use ``dtype=float64`` so the accumulator is wide enough to avoid
integer overflow on raw camera counts, without up-casting the whole frame first.
"""

import numpy as np

from pytweezer.analysis.analysis_base import ImageAnalysis, run_analysis
from pytweezer.servers import DataClient


class Projections(ImageAnalysis):
    def __init__(self, name):
        super().__init__(name)
        # Separate publisher on the Data hub for the 1-D profiles.
        self.dataq = DataClient(name.split("/")[-1])

    def process(self, head, data):
        if data.ndim != 2:
            return None

        h, w = data.shape
        offset = head.get("_offset", [0, 0])
        resolution = head.get("_imgresolution", [1, 1])

        # x maps to columns (axis 1), y to rows (axis 0); wide accumulator.
        x_values = data.sum(axis=0, dtype=np.float64)
        y_values = data.sum(axis=1, dtype=np.float64)

        x_coords = (np.arange(w) + offset[0]) * resolution[0]
        y_coords = (np.arange(h) + offset[1]) * resolution[1]

        self.dataq.send(head, np.vstack([x_coords, x_values]), channel="_x")
        self.dataq.send(head, np.vstack([y_coords, y_values]), channel="_y")
        return None


if __name__ == "__main__":
    run_analysis(Projections)
