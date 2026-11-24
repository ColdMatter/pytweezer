"""Subtract one frame of a run from another of the same run.

Input:
    One or more image streams.

Output:
    For each input stream ``<stream>``, an image stream ``bg_subtract_<stream>``
    carrying ``float64(signal) - float64(background)`` with the signal frame's
    header, once per run.

Properties:
    *   imagestreams: ([str]) input image streams.
    *   signal_index: (int) position of the signal frame within a run.
    *   background_index: (int) position of the background frame within a run.

A frame's position is the number of frames of its stream received before it with
the same ``rid``, counting from 0; a frame with a new ``rid`` restarts the count
and discards whatever was held from the previous run. The header's own ``index``
is not used, since a live-broadcasting camera counts it from the start of
acquisition. Frames at any other position are ignored. The output name is
prefixed rather than suffixed because subscriptions match topic prefixes:
``<stream>_bg_subtract`` would reach every subscriber of ``<stream>``, this
process included.
"""

from types import MappingProxyType

import numpy as np

from pytweezer.analysis.analysis_base import ImageAnalysis, run_analysis
from pytweezer.servers import ImageClient, PropertyAttribute

OUTPUT_PREFIX = "bg_subtract_"


def output_stream(input_stream):
    return OUTPUT_PREFIX + input_stream


class BgSubtract(ImageAnalysis):
    _signal_index = PropertyAttribute("signal_index", 0)
    _background_index = PropertyAttribute("background_index", 1)

    # stream -> (rid, frames received in this rid, {position: (head, frame)});
    # replaced on each change.
    _held = MappingProxyType({})

    def __init__(self, name):
        super().__init__(name)
        self._output_clients = {}

    def process(self, head, data):
        if data is None or data.ndim != 2:
            return None
        signal_index = int(self._signal_index)
        background_index = int(self._background_index)
        stream = self.current_stream
        rid = head.get("rid")

        held = self._held.get(stream)
        if held is None or held[0] != rid:
            held = (rid, 0, {})
        _, index, frames = held
        if index in (signal_index, background_index):
            frames = {**frames, index: (head, np.asarray(data, dtype=np.float64))}
        self._held = {**self._held, stream: (rid, index + 1, frames)}

        if index in (signal_index, background_index) and len(frames) == len(
            {signal_index, background_index}
        ):
            signal_head, signal = frames[signal_index]
            _, background = frames[background_index]
            if signal.shape != background.shape:
                return None
            return dict(signal_head), signal - background
        return None

    def send_result(self, head, data):
        stream = self.current_stream
        if stream not in self._output_clients:
            self._output_clients[stream] = ImageClient(output_stream(stream))
        self._output_clients[stream].send(head, data)


if __name__ == "__main__":
    run_analysis(BgSubtract)
