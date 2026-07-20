"""Crop sub-images from ROIs drawn in the image viewers and republish them.

Input:
    One image stream.

Output:
    One image stream per configured ROI, published on channel ``_<roi>`` (so a
    ROI whose property path ends in ``.../slice`` is republished on
    ``<name>_slice``). Downstream viewers/analyses can subscribe to each crop.

Properties:
    *   imagestreams: ([str]) input image streams.
    *   roipaths: ([str]) absolute Property paths of the ROIs to apply, e.g.
        ``/Image Monitor/ROI/slice``. These are exactly the entries the image
        viewers register under ``/Servers/roilist`` and persist to
        ``properties.json`` (each with a ``pos`` and ``size`` sub-key). Pick the
        ones you want here; ``--None--`` entries are ignored.

ROI geometry lives in the *physical* plot coordinates the viewer draws in
(``pos``/``size`` in metres, matching :mod:`pytweezer.GUI.viewers.image_monitor`),
so they are mapped back to pixel indices using the image header's ``_offset``
(pixel origin) and ``_imgresolution`` (metres/pixel). When those are absent the
maths collapses to treating the ROI numbers as raw pixels.

The crop is a NumPy view; it is passed through :func:`numpy.ascontiguousarray`
before sending so ZMQ transmits a packed buffer rather than a stride into the
parent frame.
"""

import numpy as np

from pytweezer.analysis.analysis_base import ImageAnalysis, run_analysis
from pytweezer.servers import PropertyAttribute


class RoiSlice(ImageAnalysis):
    _roipaths = PropertyAttribute("roipaths", ["--None--"])

    @staticmethod
    def _pixel_bounds(pos, size, offset, resolution, shape):
        """Convert one ROI's physical (pos, size) to clipped pixel bounds.

        Returns ``(i0, i1, j0, j1)`` row/column slice bounds, or ``None`` if the
        ROI does not overlap the image. ``x`` maps to columns (axis 1) and ``y``
        to rows (axis 0), matching the viewer's ``setRect`` convention.
        """
        h, w = shape[0], shape[1]
        res_x, res_y = float(resolution[0]) or 1.0, float(resolution[1]) or 1.0
        off_x, off_y = float(offset[0]), float(offset[1])

        # physical = (pixel + offset) * resolution  ->  pixel = physical/res - offset
        x0 = pos[0] / res_x - off_x
        x1 = (pos[0] + size[0]) / res_x - off_x
        y0 = pos[1] / res_y - off_y
        y1 = (pos[1] + size[1]) / res_y - off_y

        # ROIs can be dragged inverted (negative size); normalise.
        j0, j1 = sorted((x0, x1))
        i0, i1 = sorted((y0, y1))

        j0 = int(np.clip(np.floor(j0), 0, w))
        j1 = int(np.clip(np.ceil(j1), 0, w))
        i0 = int(np.clip(np.floor(i0), 0, h))
        i1 = int(np.clip(np.ceil(i1), 0, h))

        if j1 <= j0 or i1 <= i0:
            return None
        return i0, i1, j0, j1

    def process(self, head, data):
        if data.ndim != 2:
            return None

        offset = head.get("_offset", [0, 0])
        resolution = head.get("_imgresolution", [1, 1])

        for path in self._roipaths:
            if not path or path == "--None--":
                continue
            base = path.rstrip("/")
            pos = self._props.get(base + "/pos", [0, 0])
            size = self._props.get(base + "/size", [0, 0])

            bounds = self._pixel_bounds(pos, size, offset, resolution, data.shape)
            if bounds is None:
                continue
            i0, i1, j0, j1 = bounds

            # View into the frame; pack it so ZMQ sends only the sub-image bytes.
            sub = np.ascontiguousarray(data[i0:i1, j0:j1])

            # Shift the pixel origin so downstream viewers place the crop where
            # the ROI sits in the parent frame.
            sub_head = dict(head)
            sub_head["_offset"] = [offset[0] + j0, offset[1] + i0]

            region = base.split("/")[-1]
            self.client.send(sub_head, sub, channel="_" + region)

        # We publish per-ROI above; nothing for the base to send.
        return None


if __name__ == "__main__":
    run_analysis(RoiSlice)
