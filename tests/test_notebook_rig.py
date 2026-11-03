import warnings
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from pytweezer.drivers.shot_frames import settle_frame_counts
from pytweezer.experiment import run_local
from pytweezer.experiment.task import TaskStatus
from pytweezer.experiments.caf_fluorescence import LASERS
from pytweezer.notebook import Rig, plot_auto_mot_results, zips
from pytweezer.optimisers.optimise_mot_lasers import OptimiseMOTLasers
from tests.test_caf_fluorescence import START
from tests.test_notebook_zips import fake_motmaster_zip, read_zip_like_analysis3
from tests.test_optimise_mot_lasers import caf as caf_experiment  # noqa: F401
from tests.test_optimise_mot_lasers import saved_set_points  # noqa: F401

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
        self.arms = 0
        self.value = lambda index: index

    def arm(self, buffer_frames=1000):
        self.armed_with = buffer_frames
        self.arms += 1

    def close_rpc(self):
        self.closed = True

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

    def close_rpc(self):
        self.closed = True

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


def test_settle_frame_counts_waits_for_a_late_frame():
    camera = FakeCamera(1, lag_polls=3)
    camera.trigger(1)
    settle_frame_counts({"CaF": camera}, 0.05, 2.0, 0.01)
    assert camera.frames_acquired() == 1


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
    assert motmaster.linked_calls == [True]
    rig.close()
    assert motmaster.linked_calls == [True, False]


def test_a_dropped_link_is_claimed_again_before_the_next_run(rig_parts):
    rig, motmaster, _ = rig_parts()
    rig.linked = True
    rig.run("S", save=False)
    rig.run("S", save=False)
    assert motmaster.linked_calls == [True, True]


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


@pytest.fixture
def optimiser_run(monkeypatch):
    """Patch the manager out of ``Rig.auto_mot``; returns the submissions and the task it waits for."""
    run = SimpleNamespace(
        submissions=[], task=SimpleNamespace(status=TaskStatus.COMPLETED, error=None)
    )

    def submit(experiment, **arguments):
        run.submissions.append((experiment, arguments))
        return 42

    monkeypatch.setattr("pytweezer.notebook.rig.submit", submit)
    monkeypatch.setattr("pytweezer.notebook.rig.wait", lambda rid: run.task)
    monkeypatch.setattr(
        "pytweezer.notebook.rig.fetch_measurement", lambda rid: run.measurement
    )
    return run


def canned_measurement(*lasers):
    """The columns and constants of an OptimiseMOTLasers file that scanned ``lasers``."""
    stages = np.repeat([LASERS.index(laser) for laser in lasers], 3)
    return SimpleNamespace(
        results={
            "stage": stages,
            "setting": np.tile([1.0, 2.0, 3.0], len(lasers)),
            "counts": np.tile([5.0, 9.0, 4.0], len(lasers)),
            "counts_error": np.full(len(stages), 0.5),
        },
        constants={
            "start_set_points": START,
            **{f"proposed.caf_laser_{laser}": 2.0 for laser in lasers},
        },
    )


def test_auto_mot_submits_the_chosen_stages(rig_parts, optimiser_run):
    rig, _, _ = rig_parts()
    optimiser_run.measurement = canned_measurement("BX", "v0")
    rig.auto_mot(
        ["BX", "v0"],
        spans={"BX": 20e-6},
        points={"v0": 12},
        centres={"v0": 494.4},
        mot_field=2.0,
    )
    ((experiment, arguments),) = optimiser_run.submissions
    assert experiment is OptimiseMOTLasers
    assert {laser: arguments[f"{laser}_run"] for laser in LASERS} == {
        "v0": True,
        "BX": True,
        "v1": False,
        "v2": False,
        "v3": False,
        "TCool": False,
    }
    assert arguments["BX_span"] == 20e-6
    assert arguments["v0_points"] == 12
    assert arguments["v0_centre"] == 494.4
    assert arguments["mot_field"] == 2.0
    assert "v1_span" not in arguments


@pytest.mark.usefixtures("caf_experiment", "saved_set_points")
def test_auto_mot_returns_each_stage_in_the_shape_the_plot_takes(
    rig_parts, optimiser_run
):
    optimiser_run.measurement = run_local(
        OptimiseMOTLasers,
        frame_settle_s=0.0,
        v0_run=True,
        BX_run=True,
        v1_run=False,
        TCool_run=False,
        v0_points=5,
        BX_points=4,
    )
    assert optimiser_run.measurement.status == "completed"
    rig, _, _ = rig_parts()
    results = rig.auto_mot(["BX", "v0"])
    assert list(results) == ["v0", "BX"]
    measurement = optimiser_run.measurement
    for laser, n_points in (("BX", 4), ("v0", 5)):
        stage = results[laser]
        assert set(stage) == {
            "scan_range",
            "numbers",
            "errors",
            "set_point",
            "start_set_point",
        }
        assert len(stage["scan_range"]) == len(stage["numbers"]) == n_points
        assert len(stage["errors"]) == n_points
        assert stage["start_set_point"] == START[laser]
        assert (
            stage["set_point"] == measurement.constants[f"proposed.caf_laser_{laser}"]
        )
    on_v0 = measurement.results["stage"] == LASERS.index("v0")
    assert results["v0"]["scan_range"] == measurement.results["setting"][on_v0].tolist()
    assert results["v0"]["numbers"] == measurement.results["counts"][on_v0].tolist()
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    fig, axes = plot_auto_mot_results(results)
    assert axes.size >= len(results)
    plt.close(fig)


def test_auto_mot_refuses_an_unknown_laser(rig_parts, optimiser_run):
    rig, _, _ = rig_parts()
    with pytest.raises(KeyError, match="TCool"):
        rig.auto_mot(["v00"])
    with pytest.raises(KeyError, match="v00"):
        rig.auto_mot(["v0"], spans={"v00": 1e-6})
    assert optimiser_run.submissions == []


@pytest.mark.parametrize("status", [TaskStatus.FAILED, TaskStatus.TERMINATED])
def test_auto_mot_raises_for_a_run_that_did_not_complete(
    rig_parts, optimiser_run, status
):
    optimiser_run.task = SimpleNamespace(status=status, error="no frames")
    rig, _, _ = rig_parts()
    with pytest.raises(RuntimeError, match=f"{status.value}.*no frames"):
        rig.auto_mot(["v0"])


def test_auto_mot_does_not_accept(rig_parts, optimiser_run, monkeypatch):
    optimiser_run.measurement = canned_measurement("v0")
    accepted = []
    monkeypatch.setattr(
        "pytweezer.experiment.client.set_saved_parameters",
        lambda *args, **kwargs: accepted.append(args),
    )
    rig, motmaster, _ = rig_parts()
    rig.auto_mot(["v0"])
    ((_, arguments),) = optimiser_run.submissions
    assert not any("accept" in name for name in arguments)
    assert accepted == []
    assert motmaster.calls == []


def test_every_changed_frame_count_warns_from_the_callers_line(rig_parts):
    rig, motmaster, cameras = rig_parts()
    counts = iter([3, 2, 3, 2])
    camera = cameras["CaF"]
    original_trigger = camera.trigger

    def alternating(iterations):
        camera.frames_per_iteration = next(counts)
        original_trigger(iterations)

    camera.trigger = alternating
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        rig.scan("S", "dummy", [1, 2, 3, 4], save=False)
    count_warnings = [w for w in caught if "frames from" in str(w.message)]
    assert len(count_warnings) == 3
    assert {w.filename for w in count_warnings} == {__file__}


def test_a_camera_is_armed_again_before_every_shot(rig_parts):
    rig, _, cameras = rig_parts()
    rig.run("S", save=False)
    rig.run("S", save=False)
    assert cameras["Rb"].arms == 3


def connect_by_name(monkeypatch, tmp_path, simulate=False):
    """Make the rig's get_device hand out fakes, keeping every one it made."""
    made = []

    def get_device(name):
        if name == "MM":
            device = FakeMotMaster(
                tmp_path,
                [
                    made_camera
                    for made_camera in made
                    if isinstance(made_camera, FakeCamera)
                ],
            )
        else:
            device = FakeCamera(1)
        made.append(device)
        return device

    monkeypatch.setattr("pytweezer.notebook.rig.get_device", get_device)
    monkeypatch.setattr(
        "pytweezer.notebook.rig._motmaster_file_config",
        lambda name: {"data_prefix": "CaF", "data_root_path": str(tmp_path)},
    )
    monkeypatch.setattr(
        "pytweezer.notebook.rig.get_device_config",
        lambda name: {"simulate": simulate},
    )
    return made


def test_an_interrupted_call_reconnects_the_rigs_own_clients(monkeypatch, tmp_path):
    made = connect_by_name(monkeypatch, tmp_path)
    rig = Rig("MM", {"CaF": "CaF cam"}, frame_settle_s=0.0)
    first_motmaster = rig.motmaster
    first_motmaster.fail_on_go = 1
    with pytest.raises(KeyboardInterrupt):
        rig.run("S", save=False)
    assert first_motmaster.closed
    assert rig.motmaster is not first_motmaster
    assert rig.cameras["CaF"] is not made[1]
    assert rig.run("S", save=False).params["tLoad"] == 100


def test_a_simulated_motmaster_runs_without_saving(monkeypatch, tmp_path):
    connect_by_name(monkeypatch, tmp_path, simulate=True)
    rig = Rig("MM", {}, frame_settle_s=0.0)
    with pytest.warns(UserWarning, match="simulated"):
        shot = rig.run("S")
    assert shot.zips == {}
    assert ("save", False) in rig.motmaster.calls
