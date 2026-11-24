"""Publish the cloud fit of every frame of an image stream.

Input:
    One image stream, raw or background-subtracted.

Output:
    A data message (no array) on ``<name>``: the fields of :class:`CloudFit`
    plus ``rid``, ``index``, ``_imgresolution`` and ``_offset`` from the input
    header. A frame that cannot be fitted publishes ``success`` False and NaN
    for every numeric fit field, so a plot of history shows a gap.

Properties:
    *   imagestreams: ([str]) input image streams.
"""

import math

from pytweezer.analysis.analysis_base import ImageAnalysis, run_analysis
from pytweezer.analysis.cloud_fit import FIELDS, fit_cloud, frame_geometry
from pytweezer.logging_utils import get_logger
from pytweezer.servers import DataClient

LOGGER = get_logger(__name__)

_PASS_THROUGH = ("rid", "index", "_imgresolution", "_offset")


class CloudFitAnalysis(ImageAnalysis):
    """Publish :class:`CloudFit` of every frame on the Data hub."""

    def __init__(self, name):
        super().__init__(name)
        self.dataq = DataClient(name.split("/")[-1])

    def process(self, head, data):
        if data is None or data.ndim != 2:
            return
        out = {key: head[key] for key in _PASS_THROUGH if key in head}
        resolution, origin = frame_geometry(head)
        try:
            fit = fit_cloud(data, resolution, origin).as_dict()
        except Exception:
            LOGGER.warning("cloud fit failed on a frame", exc_info=True)
            fit = {"units": "px" if resolution is None else "m", "success": False}
        if not fit["success"]:
            fit.update(dict.fromkeys(FIELDS, math.nan))
        out.update(fit)
        self.dataq.send(out, None)
        return


if __name__ == "__main__":
    run_analysis(CloudFitAnalysis)
