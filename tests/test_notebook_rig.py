from datetime import date

import numpy as np
import pytest

from pytweezer.notebook import Rig, zips
from tests.test_notebook_zips import fake_motmaster_zip, read_zip_like_analysis3

MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


class FakeCamera:
    """Gains ``frames_per_iteration`` frames per MOTMaster iteration when triggered.

    With ``lag_polls``, the last frame of a shot only shows up after that many more
    ``frames_acquired()`` calls, as a frame still in transit would.
    """

    def __init__(self, frames_per_iteration, lag_polls=0, shape=(4, 5)):
        self.frames_per_iteration = frames_per_iteration
        self.lag_polls = lag_polls
        self.shape = shape
        self.count = 0
        self.late = 0
        self.polls_until_late = 0
        self.armed_with = None
        self.value = lambda index: index

    def arm(self, buffer_frames=1000):
        self.armed_with = buffer_frames

    def trigger(self, iterations):
        n = self.frames_per_iteration * iterations
        if self.lag_polls and n:
            self.count += n - 1
            self.late, self.polls_until_late = 1, self.lag_polls
        else:
            self.count += n

    def frames_acquired(self):
        if self.late:
            if self.polls_until_late == 0:
                self.count += self.late
                self.late = 0
            else:
                self.polls_until_late -= 1
        return self.count

    def read_frames_from(self, start):
        if start >= self.count:
            return np.empty((0, 0, 0), dtype=np.uint16)
        return np.stack(
            [
                np.full(self.shape, self.value(index), dtype=np.uint16)
                for index in range(start, self.count)
            ]
        )

    def settings(self):
        return {"fake": True}


class FakeMotMaster:
    """Triggers the cameras on every Go and, when saving, writes a run zip."""

    def __init__(self, root, cameras):
        self.root = root
        self.cameras = cameras
        self.calls = []
        self.linked_calls = []
        self.iterations = 1
        self.save = False
        self.gos = 0
        self.fail_on_go = None
        self.set_points = {"BX": 564.58, "v0": 494.43}

    def set_motmaster_experiment(self, script):
        self.script = script

    def set_run_until_stopped(self, value):
        self.calls.append(("run_until_stopped", value))

    def set_trigger_mode(self, value):
        self.calls.append(("trigger_mode", value))

    def set_iterations(self, iterations):
        self.iterations = iterations

    def set_save_toggle(self, save):
        self.save = save
        self.calls.append(("save", save))

    def set_linked(self, value):
        self.linked_calls.append(value)

    def get_params(self):
        return {"dummy": 0.0, "tLoad": 100}

    def laser_set_points_wml(self):
        return dict(self.set_points)

    def ramp_laser_set_point_wml(self, laser, target):
        self.calls.append(("ramp", laser, target))
        self.set_points[laser] = target

    def set_laser_set_point_wml(self, laser, value):
        self.calls.append(("set", laser, value))
        self.set_points[laser] = value

    def start_motmaster_experiment(self, parameters=None):
        self.calls.append(("go",))
        self.gos += 1
        if self.fail_on_go == self.gos:
            raise KeyboardInterrupt
        self.go_parameters = parameters
        for camera in self.cameras:
            camera.trigger(self.iterations)
        if self.save:
            today = date.today()
            folder = zips.day_folder(self.root, today)
            number = len(zips.snapshot([folder], "CaF"))
            tag = f"CaF{today.day:02d}{MONTHS[today.month - 1]}{today:%y}00"
            fake_motmaster_zip(folder, f"{tag}_{number:03d}")


@pytest.fixture
def rig_parts(tmp_path):
    def make(caf=3, rb=2, lag_polls=0):
        cameras = {"CaF": FakeCamera(caf, lag_polls), "Rb": FakeCamera(rb)}
        motmaster = FakeMotMaster(tmp_path, cameras.values())
        rig = Rig(
            motmaster,
            cameras,
            element="CaF",
            data_root=tmp_path,
            frame_settle_s=0.06,
            frame_poll_s=0.01,
        )
        return rig, motmaster, cameras

    return make


def test_run_returns_each_cameras_frames_and_parameters(rig_parts):
    rig, _, _ = rig_parts()
    shot = rig.run("S", dummy=1.5)
    assert shot.images["CaF"].shape == (3, 4, 5)
    assert shot.images["Rb"].shape == (2, 4, 5)
    assert shot.images["CaF"].dtype == float
    assert shot.params["dummy"] == 1.5
    assert shot.params["tLoad"] == 100


def test_cameras_are_armed_once_the_rig_is_made(rig_parts):
    _, _, cameras = rig_parts()
    assert all(camera.armed_with == 1000 for camera in cameras.values())


def test_saved_run_puts_caf_frames_in_motmasters_zip_and_rb_in_a_sibling(rig_parts):
    rig, _, _ = rig_parts()
    shot = rig.run("S")
    caf, caf_params = read_zip_like_analysis3(shot.zips["CaF"])
    rb, rb_params = read_zip_like_analysis3(shot.zips["Rb"])
    assert caf.shape[0] == 3 and rb.shape[0] == 2
    assert caf_params == rb_params == {"dummy": 0.0}
    assert shot.zips["Rb"].name == "Rb" + shot.zips["CaF"].name[3:]
    assert shot.zip_number == 0


def test_iterations_land_in_one_zip(rig_parts):
    rig, _, _ = rig_parts()
    shot = rig.run("S", iterations=3)
    images, _ = read_zip_like_analysis3(shot.zips["CaF"])
    assert images.shape[0] == 9
    assert shot.images["CaF"].shape[0] == 9


def test_unsaved_run_writes_no_zip(rig_parts, tmp_path):
    rig, motmaster, _ = rig_parts()
    shot = rig.run("S", save=False)
    assert shot.zips == {}
    assert shot.zip_number is None
    assert ("save", False) in motmaster.calls
    assert not list(tmp_path.rglob("*.zip"))


def test_frames_arriving_after_go_are_collected(rig_parts):
    rig, _, _ = rig_parts(caf=3, lag_polls=1)
    assert rig.run("S").images["CaF"].shape[0] == 3


def test_changed_frame_count_warns(rig_parts):
    rig, _, cameras = rig_parts()
    rig.run("S", save=False)
    cameras["CaF"].frames_per_iteration = 2
    with pytest.warns(UserWarning, match="CaF: 2 frames from 'S', 3 last time"):
        rig.run("S", save=False)


def test_a_different_script_does_not_warn(rig_parts, recwarn):
    rig, _, cameras = rig_parts()
    rig.run("S", save=False)
    cameras["CaF"].frames_per_iteration = 2
    rig.run("T", save=False)
    assert not recwarn.list


def test_linked_mode_is_set_and_released(rig_parts):
    rig, motmaster, _ = rig_parts()
    rig.linked = True
    rig.run("S", save=False)
    rig.run("S", save=False)
    assert motmaster.linked_calls == [True]
    rig.close()
    assert motmaster.linked_calls == [True, False]


def test_an_unlinked_rig_clears_a_stale_link_and_leaves_it_on_close(rig_parts):
    rig, motmaster, _ = rig_parts()
    rig.run("S", save=False)
    rig.close()
    assert motmaster.linked_calls == [False]


def test_runs_are_never_triggered_or_left_running(rig_parts):
    rig, motmaster, _ = rig_parts()
    rig.run("S", save=False)
    assert ("trigger_mode", False) in motmaster.calls
    assert ("run_until_stopped", False) in motmaster.calls


def test_a_motmaster_object_needs_an_element_and_data_root(tmp_path):
    with pytest.raises(ValueError, match="element"):
        Rig(FakeMotMaster(tmp_path, []), {})


def test_scan_runs_once_per_value_with_that_value(rig_parts):
    rig, motmaster, _ = rig_parts()
    shots = rig.scan("S", "dummy", [1, 2, 3])
    assert [shot.params["dummy"] for shot in shots] == [1, 2, 3]
    assert [shot.zip_number for shot in shots] == [0, 1, 2]
    assert [motmaster.go_parameters] == [{"dummy": 3}]


def test_scan_laser_ramps_then_steps_through_the_values(rig_parts):
    rig, motmaster, _ = rig_parts()
    values = [564.57, 564.58, 564.59]
    shots = rig.scan_laser("S", "BX", values, save=False)
    laser_calls = [call for call in motmaster.calls if call[0] in ("ramp", "set", "go")]
    assert laser_calls[0] == ("ramp", "BX", values[0])
    assert laser_calls[1:] == [
        call for value in values for call in (("set", "BX", value), ("go",))
    ]
    assert [shot.params["BX"] for shot in shots] == values


def test_interrupted_scan_keeps_completed_shots(rig_parts):
    rig, motmaster, _ = rig_parts()
    motmaster.fail_on_go = 3
    with pytest.raises(KeyboardInterrupt):
        rig.scan("S", "dummy", [1, 2, 3, 4], save=False)
    assert [shot.params["dummy"] for shot in rig.last_shots] == [1, 2]


def test_laser_set_points_are_read_from_the_motmaster(rig_parts):
    rig, _, _ = rig_parts()
    assert rig.laser_set_points() == {"BX": 564.58, "v0": 494.43}


def test_numpy_values_reach_motmaster_as_python_numbers(rig_parts):
    rig, motmaster, _ = rig_parts()
    rig.scan("S", "dummy", np.arange(2), save=False)
    assert type(motmaster.go_parameters["dummy"]) is int
