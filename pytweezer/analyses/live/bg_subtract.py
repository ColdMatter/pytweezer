"""Subtract one frame of a run from another of the same run.

Input:
    One or more image streams.

Output:
    For each input stream ``<stream>``, an image stream ``bg_subtract_<stream>``
    carrying ``float64(signal) - float64(background)`` with the signal frame's
    header, once per run.

Properties:
    *   imagestreams: ([str]) input image streams.
    *   signal_index: (int) ``index`` of the signal frame within a shot.
    *   background_index: (int) ``index`` of the background frame within a shot.

Only frames an experiment's shot tagged (``Camera.start_shot``, which adds the
``rid`` and numbers ``index`` from 0 within the shot) are used; untagged live
frames number ``index`` from the start of acquisition and are ignored. A frame
whose ``index`` does not exceed the previous one's, or whose ``rid`` differs,
starts a new shot and discards whatever was held from the last. Frames at any
other ``index`` are ignored. The output name is
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

    # stream -> (rid, index of the last frame, {index: (head, frame)});
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
        if rid is None:
            return None
        index = int(head["index"])

        held = self._held.get(stream)
        if held is None or held[0] != rid or index <= held[1]:
            frames = {}
        else:
            frames = held[2]
        if index in (signal_index, background_index):
            frames = {**frames, index: (head, np.asarray(data, dtype=np.float64))}
        self._held = {**self._held, stream: (rid, index, frames)}

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
