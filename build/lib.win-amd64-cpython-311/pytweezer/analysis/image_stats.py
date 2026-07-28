"""Compute summary statistics of each image and publish them.

Input:
    One image stream.

Output:
    A data message (no array) carrying ``mean``, ``std``, ``stderr``, ``min``,
    ``max``, ``sum`` and the pixel count ``n``, published on ``<name>`` (the
    Data hub). Intended for the image viewer's data sidebar.

Properties:
    *   imagestreams: ([str]) input image streams.

``stderr`` is the standard error of the mean, ``std / sqrt(n)``. Statistics are
computed in float64 so integer camera frames do not overflow or lose precision.
"""

import numpy as np

from pytweezer.analysis.analysis_base import ImageAnalysis, run_analysis
from pytweezer.servers import DataClient


class ImageStats(ImageAnalysis):
    def __init__(self, name):
        super().__init__(name)
        # Stats are 1-D scalars, so publish them on the Data hub.
        self.dataq = DataClient(name.split("/")[-1])

    def process(self, head, data):
        values = np.asarray(data, dtype=np.float64)
        n = int(values.size)
        if n == 0:
            return None

        mean = float(values.mean())
        std = float(values.std())
        stats = {
            "n": n,
            "mean": mean,
            "std": std,
            "stderr": float(std / np.sqrt(n)),
            "min": float(values.min()),
            "max": float(values.max()),
            "sum": float(values.sum()),
        }
        if "_imgindex" in head:
            stats["_imgindex"] = head["_imgindex"]
        if "timestamp" in head:
            stats["timestamp"] = head["timestamp"]

        self.dataq.send(stats, None)
        return None


if __name__ == "__main__":
    run_analysis(ImageStats)
