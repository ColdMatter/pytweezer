"""Recapture rate over a MOTMaster run of three-frame shots.

The live counterpart of the CaF notebooks' ``get_recapture_rate_three_image``.
Each shot images the cloud (frame 0), the recaptured cloud (frame 1) and the
background (frame 2). With ``B`` the run's mean background frame, a shot's rate
is ``sum(second - B) / sum(first - B)``; the run's recapture rate is the mean of
its shots' rates, with their standard error.

Input:
    One image stream from a camera that ``PycafCamera`` watches (or that an
    experiment reads), so that each frame's ``index`` is its place in the run.

Output:
    After each complete shot, a data message (no array) on ``<name>`` carrying
    ``recapture_rate`` and ``recapture_rate_err`` over the run's shots so far,
    ``shots``, and the latest shot's ``shot_rate`` and background-subtracted
    ``first_counts`` and ``second_counts``, plus the input's ``pycaf_run``,
    ``rid`` and ``timestamp`` when it has them. The same header goes out on the
    image hub with the run's mean background-subtracted first and second frames
    side by side.

Properties:
    *   imagestreams: ([str]) input image streams.
    *   skip: (int) shots at the start of each run left out.
    *   crop: ([int]) ``[row_start, row_stop, col_start, col_stop]`` of the
        region summed, as the notebook's ``crop``; empty for the whole frame.

Every message recomputes the rates over all the run's shots so far, with the
background averaged over them, so a run's last message gives the notebook's
numbers for that run's zip. A frame whose ``pycaf_run`` or ``rid`` differs from
the last one's, or whose ``index`` goes back, starts a new run. Frames with
neither tag are ignored: their ``index`` counts from the camera's acquisition
start and does not say which frame of a shot they are. A shot missing a frame
is left out.
"""

import numpy as np

from pytweezer.analysis.analysis_base import ImageAnalysis, run_analysis
from pytweezer.servers import DataClient, PropertyAttribute

FRAMES_PER_SHOT = 3
FIRST, SECOND, BACKGROUND = range(FRAMES_PER_SHOT)
PASSED_KEYS = ("pycaf_run", "rid", "timestamp")


def recapture_statistics(first_sums, second_sums, background_sums):
    """``(rate, standard error, each shot's rate)`` from each shot's frame sums."""
    background = np.mean(background_sums)
    with np.errstate(divide="ignore", invalid="ignore"):
        shot_rates = (np.asarray(second_sums) - background) / (
            np.asarray(first_sums) - background
        )
    return shot_rates.mean(), shot_rates.std() / np.sqrt(len(shot_rates)), shot_rates


class _Run:
    def __init__(self, tag, shape):
        self.tag = tag
        self.shape = shape
        self.last_index = -1
        self.shot = None
        self.frames = {}
        self.sums = []  # [first, second, background] of each complete shot
        self.frame_totals = np.zeros((FRAMES_PER_SHOT, *shape))

    def add_shot(self):
        stack = np.stack([self.frames[role] for role in range(FRAMES_PER_SHOT)])
        self.frame_totals += stack
        self.sums.append(stack.sum(axis=(1, 2)))


class RecaptureRate(ImageAnalysis):
    _skip = PropertyAttribute("skip", 0)
    _crop = PropertyAttribute("crop", [])

    _run = None

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

        run = self._run
        if (
            run is None
            or run.tag != tag
            or index <= run.last_index
            or run.shape != frame.shape
        ):
            run = self._run = _Run(tag, frame.shape)
        run.last_index = index

        shot, role = divmod(int(index), FRAMES_PER_SHOT)
        if shot < self._skip:
            return None
        if run.shot != shot:
            run.shot, run.frames = shot, {}
        run.frames[role] = frame
        if role != BACKGROUND or len(run.frames) != FRAMES_PER_SHOT:
            return None
        run.add_shot()
        return self._publish(head, run)

    def _cropped(self, frame):
        crop = self._crop
        if not crop:
            return frame
        if len(crop) != 4:
            return None
        row_start, row_stop, col_start, col_stop = (int(edge) for edge in crop)
        return frame[row_start:row_stop, col_start:col_stop]

    def _publish(self, head, run):
        first, second, background = np.transpose(run.sums)
        rate, rate_err, shot_rates = recapture_statistics(first, second, background)
        mean_background = background.mean()
        out_head = {
            "recapture_rate": float(rate),
            "recapture_rate_err": float(rate_err),
            "shots": len(shot_rates),
            "shot_rate": float(shot_rates[-1]),
            "first_counts": float(first[-1] - mean_background),
            "second_counts": float(second[-1] - mean_background),
        }
        out_head.update({key: head[key] for key in PASSED_KEYS if key in head})
        self.dataq.send(out_head, None)

        mean_frames = run.frame_totals / len(run.sums)
        side_by_side = np.hstack(
            [
                mean_frames[FIRST] - mean_frames[BACKGROUND],
                mean_frames[SECOND] - mean_frames[BACKGROUND],
            ]
        )
        return dict(out_head), side_by_side


if __name__ == "__main__":
    run_analysis(RecaptureRate)
