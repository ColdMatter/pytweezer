"""Subtract the background shot of each task from its frames.

Input:
    One or more image streams; each keeps its own background.

Output:
    An image stream ``<name>`` carrying every frame that is not a background
    frame as ``float64(frame) - background``, with the input header.

Properties:
    *   imagestreams: ([str]) input image streams.

A frame whose header has ``background`` set is held, not published. Frames of
one background shot (same ``rid``) are averaged; a background frame with a
different ``rid`` starts a new average for that stream. The difference is signed, so noise
around a zero background stays unbiased. A frame whose ``rid`` or shape does not
match its stream's held background is published unchanged, as ``float64``.
"""

from types import MappingProxyType

import numpy as np

from pytweezer.analysis.analysis_base import ImageAnalysis, run_analysis


class ShotBackground(ImageAnalysis):
    # stream -> (rid, sum of frames, count); replaced on each change.
    _held = MappingProxyType({})

    def process(self, head, data):
        if data is None or data.ndim != 2:
            return None
        frame = np.asarray(data, dtype=np.float64)
        stream = self.current_stream

        if head.get("background"):
            self._hold(stream, head.get("rid"), frame)
            return None

        out_head = dict(head)
        held = self._held.get(stream)
        if held is not None:
            rid, total, count = held
            if head.get("rid") == rid and frame.shape == total.shape:
                frame = frame - total / count
        return out_head, frame

    def _hold(self, stream, rid, frame):
        held = self._held.get(stream)
        if held is not None and held[0] == rid and held[1].shape == frame.shape:
            entry = (rid, held[1] + frame, held[2] + 1)
        else:
            entry = (rid, frame.copy(), 1)
        self._held = {**self._held, stream: entry}


if __name__ == "__main__":
    run_analysis(ShotBackground)
