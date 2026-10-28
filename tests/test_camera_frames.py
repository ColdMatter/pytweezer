import time

import pytest

from pytweezer.drivers import camera_base
from pytweezer.drivers.camera_base import SimulatedCamera


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(camera_base.time, "monotonic", clock)
    return clock


@pytest.fixture
def camera(clock):
    camera = SimulatedCamera()
    camera.sim_frame_rate = 5.0
    yield camera
    camera.close()


class Recorder:
    def __init__(self):
        self.indices = []

    def send(self, info, image):
        self.indices.append(info["index"])


def test_arm_starts_a_stopped_camera_once(camera, monkeypatch):
    setups = []
    monkeypatch.setattr(camera, "setup_acquisition", lambda *a: setups.append(a))
    camera.arm()
    camera.arm()
    assert camera._acquisition_in_progress()
    assert len(setups) == 1


def test_frames_since_a_mark_are_exactly_the_new_ones(camera, clock):
    camera.arm()
    mark = camera.frames_acquired()
    clock.now += 0.6
    assert camera.read_frames_from(mark).shape[0] == 3


def test_reading_with_no_new_frames_gives_an_empty_stack(camera):
    camera.arm()
    assert camera.read_frames_from(camera.frames_acquired()).shape == (0, 0, 0)


def test_live_broadcast_sends_each_new_frame_once(camera, clock):
    recorder = Recorder()
    camera.image_client = recorder
    camera.arm()
    camera.start_live(poll_s=0.01)
    clock.now += 0.6
    deadline = time.perf_counter() + 5
    while len(recorder.indices) < 3 and time.perf_counter() < deadline:
        time.sleep(0.01)
    time.sleep(0.05)
    camera.stop_live()
    assert recorder.indices == [0, 1, 2]
