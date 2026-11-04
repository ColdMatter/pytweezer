"""Subtract the background shot of each task from its frames.

Input:
    One image stream. Run one instance per camera: backgrounds are matched by
    ``rid`` and frame shape only, not by stream.

Output:
    An image stream ``<name>`` carrying every frame that is not a background
    frame as ``float64(frame) - background``, with the input header.

Properties:
    *   imagestreams: ([str]) input image streams.

A frame whose header has ``background`` set is held, not published. Frames of
one background shot (same ``rid``) are averaged; a background frame with a
different ``rid`` starts a new average. The difference is signed, so noise
around a zero background stays unbiased. A frame whose ``rid`` or shape does not
match the held background is published unchanged, as ``float64``.
"""

import numpy as np

from pytweezer.analysis.analysis_base import ImageAnalysis, run_analysis


class ShotBackground(ImageAnalysis):
    _held_rid = None
    _held_sum = None
    _held_count = 0

    def process(self, head, data):
        if data is None or data.ndim != 2:
            return None
        frame = np.asarray(data, dtype=np.float64)

        if head.get("background"):
            self._hold(head.get("rid"), frame)
            return None

        out_head = dict(head)
        if (
            self._held_sum is not None
            and head.get("rid") == self._held_rid
            and frame.shape == self._held_sum.shape
        ):
            frame = frame - self._held_sum / self._held_count
        return out_head, frame

    def _hold(self, rid, frame):
        if (
            self._held_sum is None
            or rid != self._held_rid
            or frame.shape != self._held_sum.shape
        ):
            self._held_rid, self._held_sum, self._held_count = rid, frame.copy(), 1
        else:
            self._held_sum = self._held_sum + frame
            self._held_count += 1


if __name__ == "__main__":
    run_analysis(ShotBackground)
