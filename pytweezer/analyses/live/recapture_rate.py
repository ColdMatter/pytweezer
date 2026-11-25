"""Recapture rate of each three-frame MOTMaster shot.

Each shot images the cloud (frame 0), the recaptured cloud (frame 1) and the
background (frame 2), as for the CaF notebooks' ``get_recapture_rate_three_image``.
A shot's recapture rate is ``sum(second - background) / sum(first - background)``
with that shot's own background frame.

Input:
    One image stream from a camera that ``PycafCamera`` watches (or that an
    experiment reads), so that each frame's ``index`` is its place in the run.

Output:
    For each complete shot, a data message (no array) on ``<name>`` carrying
    ``recapture_rate``, the background-subtracted ``first_counts`` and
    ``second_counts``, ``shot`` (its number in the run, from 0), and the input's
    ``pycaf_run``, ``rid`` and ``timestamp`` when it has them. The same header
    goes out on the image hub with the shot's background-subtracted first and
    second frames side by side.

Properties:
    *   imagestreams: ([str]) input image streams.
    *   skip: (int) shots at the start of each run left out.
    *   crop: ([int]) ``[row_start, row_stop, col_start, col_stop]`` of the
        region summed, as the notebook's ``crop``; empty for the whole frame.

Frames with neither a ``pycaf_run`` nor a ``rid`` header are ignored: their
``index`` counts from the camera's acquisition start and does not say which
frame of a shot they are. A shot missing a frame is left out.
"""

from types import MappingProxyType

import numpy as np

from pytweezer.analysis.analysis_base import ImageAnalysis, run_analysis
from pytweezer.servers import DataClient, PropertyAttribute

FRAMES_PER_SHOT = 3
FIRST, SECOND, BACKGROUND = range(FRAMES_PER_SHOT)
PASSED_KEYS = ("pycaf_run", "rid", "timestamp")


class RecaptureRate(ImageAnalysis):
    _skip = PropertyAttribute("skip", 0)
    _crop = PropertyAttribute("crop", [])

    # (run tag, shot number) of the shot being collected, and its frames by role.
    _shot = None
    _frames = MappingProxyType({})

    def __init__(self, name):
        super().__init__(name)
        self.dataq = DataClient(name.split("/")[-1])

    def process(self, head, data):
        tag = (head.get("pycaf_run"), head.get("rid"))
        index = head.get("index")
        if data is None or data.ndim != 2 or index is None or tag == (None, None):
            return None
        frame = self._cropped(np.asarray(data, dtype=np.float64))
        if frame is None or frame.size == 0:
            return None

        shot_number, role = divmod(int(index), FRAMES_PER_SHOT)
        if shot_number < self._skip:
            return None
        if self._shot != (tag, shot_number):
            self._shot, self._frames = (tag, shot_number), {}
        self._frames[role] = frame
        if role != BACKGROUND or len(self._frames) != FRAMES_PER_SHOT:
            return None
        if len({f.shape for f in self._frames.values()}) != 1:
            return None
        return self._publish(head, shot_number)

    def _cropped(self, frame):
        crop = self._crop
        if not crop:
            return frame
        if len(crop) != 4:
            return None
        row_start, row_stop, col_start, col_stop = (int(edge) for edge in crop)
        return frame[row_start:row_stop, col_start:col_stop]

    def _publish(self, head, shot_number):
        background = self._frames[BACKGROUND]
        first = self._frames[FIRST] - background
        second = self._frames[SECOND] - background
        first_counts, second_counts = first.sum(), second.sum()
        out_head = {
            "recapture_rate": float(second_counts / first_counts)
            if first_counts
            else float("nan"),
            "first_counts": float(first_counts),
            "second_counts": float(second_counts),
            "shot": shot_number,
        }
        out_head.update({key: head[key] for key in PASSED_KEYS if key in head})
        self.dataq.send(out_head, None)
        return dict(out_head), np.hstack([first, second])


if __name__ == "__main__":
    run_analysis(RecaptureRate)
