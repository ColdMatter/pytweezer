"""Helpers for collecting the frames a shot produced on free-running cameras."""

import time
from collections.abc import Mapping
from typing import Any


def settle_frame_counts(
    cameras: Mapping[str, Any], settle_s: float, timeout_s: float, poll_s: float
) -> None:
    """Block until every camera's frame count has stopped changing.

    Args:
        cameras: Cameras by key, each with ``frames_acquired()``.
        settle_s: How long all the counts must stay unchanged.
        timeout_s: Longest total wait, after which this returns regardless.
        poll_s: Interval between frame-count polls.
    """

    def counts():
        return {key: camera.frames_acquired() for key, camera in cameras.items()}

    last = counts()
    now = time.monotonic()
    stable_since, deadline = now, now + timeout_s
    while now - stable_since < settle_s and now < deadline:
        time.sleep(poll_s)
        current = counts()
        now = time.monotonic()
        if current != last:
            last, stable_since = current, now
