import warnings
from datetime import date

import numpy as np
import pytest

from pytweezer.notebook import PycafCamera, zips
from tests.test_notebook_rig import MONTHS, FakeCamera
from tests.test_notebook_zips import fake_motmaster_zip, read_zip_like_analysis3


class FakeHamCam(FakeCamera):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def stop_live(self):
        self.calls.append("stop_live")

    def set_external_exposure_mode(self):
        self.calls.append("external")

    def start_live(self, buffer_frames=1000):
        self.calls.append("start_live")
        super().start_live(buffer_frames)


class FakeExperiment:
    """pycaf.Experiment's scan loop: one ``self._go`` per value, errors printed."""

    def __init__(self, root, cameras, save=True):
        self.root = root
        self.cameras = cameras
        self.save = save
        self.frames_per_run = None  # list overriding the camera's count per run
        self.gos = 0

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def _go(self, dictionary=None):
        n = self.frames_per_run[self.gos] if self.frames_per_run else None
        self.gos += 1
        for camera in self.cameras:
            if n is None:
                camera.trigger(1)
            else:
                camera.count += n
        if self.save:
            today = date.today()
            folder = zips.day_folder(self.root, today)
            number = len(zips.snapshot([folder], "CaF"))
            tag = f"CaF{today.day:02d}{MONTHS[today.month - 1]}{today:%y}50"
            fake_motmaster_zip(folder, f"{tag}_{number:03d}")

    def scan_motmaster_parameter(self, script, parameter, values):
        try:
            for value in values:
                self._go({parameter: value})
        except Exception as e:
            print(f"Error: {e} encountered")
        return []

    def motmaster_single_run(self, script):
        self._go({})


@pytest.fixture
def parts(tmp_path):
    def make(frames=3, save=True, **kwargs):
        camera = FakeHamCam(frames, **kwargs)
        expt = FakeExperiment(tmp_path, [camera], save=save)
        hamcam = PycafCamera(
            expt,
            camera,
            element="CaF",
            data_root=tmp_path,
            frame_settle_s=0.03,
            frame_poll_s=0.005,
        )
        return hamcam, expt, camera

    return make


def test_setup_puts_camera_in_external_exposure_then_live(parts):
    _, _, camera = parts()
    assert camera.calls == ["stop_live", "external", "start_live"]


def test_scan_frames_are_split_per_run_and_saved_beside_each_zip(parts):
    hamcam, expt, camera = parts(frames=3)
    camera.value = lambda index: index
    expt.scan_motmaster_parameter("s", "tBlueMOTField", [1, 2, 3, 4])

    images = hamcam.store_images()

    assert [stack.shape for stack in images] == [(3, 4, 5)] * 4
    assert [stack[:, 0, 0].tolist() for stack in images] == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
    ]
    assert [path.name[:2] for path in hamcam.last_zips] == ["Rb"] * 4
    for stack, path in zip(images, hamcam.last_zips):
        from_zip, params = read_zip_like_analysis3(path)
        assert np.array_equal(from_zip, stack)
        assert params == {"dummy": 0.0}
    # Each Rb zip has the same run number as the CaF zip it sits beside.
    assert [p.name.split("_")[-1] for p in hamcam.last_zips] == [
        "000.zip",
        "001.zip",
        "002.zip",
        "003.zip",
    ]


def test_store_takes_only_runs_since_the_last_store(parts):
    hamcam, expt, camera = parts(frames=2)
    expt.scan_motmaster_parameter("s", "p", [1, 2])
    hamcam.store_images()
    camera.count += 5  # stray frames between scans are not any run's
    expt.motmaster_single_run("s")

    images = hamcam.store_images()

    assert len(images) == 1
    assert images[0][:, 0, 0].tolist() == [9, 10]
    assert hamcam.last_zips[0].name.endswith("_002.zip")


def test_uneven_frame_counts_stay_with_their_run_and_warn(parts):
    hamcam, expt, _ = parts()
    expt.frames_per_run = [3, 2, 3]
    expt.scan_motmaster_parameter("s", "p", [1, 2, 3])

    with pytest.warns(UserWarning, match=r"different frame counts \[2, 3\]"):
        images = hamcam.store_images()

    assert [len(stack) for stack in images] == [3, 2, 3]
    assert images[2][:, 0, 0].tolist() == [5, 6, 7]


def test_late_frame_is_waited_for(parts):
    hamcam, expt, _ = parts(frames=3, lag_polls=2)
    expt.scan_motmaster_parameter("s", "p", [1, 2])
    images = hamcam.store_images()
    assert [stack[:, 0, 0].tolist() for stack in images] == [[0, 1, 2], [3, 4, 5]]


def test_unsaved_runs_return_frames_and_warn(parts):
    hamcam, expt, _ = parts(save=False)
    expt.motmaster_single_run("s")
    with pytest.warns(UserWarning, match="no new MOTMaster zip"):
        images = hamcam.store_images()
    assert len(images) == 1
    assert hamcam.last_zips == [None]


def test_store_with_no_runs_warns(parts):
    hamcam, _, _ = parts()
    with pytest.warns(UserWarning, match="No MOTMaster runs"):
        assert hamcam.store_images() == []


def test_rerunning_the_cell_replaces_the_old_watcher(parts, tmp_path):
    first, expt, camera = parts()
    second = PycafCamera(
        expt,
        camera,
        element="CaF",
        data_root=tmp_path,
        frame_settle_s=0.03,
        frame_poll_s=0.005,
    )
    expt.motmaster_single_run("s")
    assert len(second.store_images()) == 1
    with pytest.warns(UserWarning, match="No MOTMaster runs"):
        first.store_images()


def test_close_restores_go(parts):
    hamcam, expt, _ = parts()
    hamcam.close()
    assert "_go" not in expt.__dict__
    expt.motmaster_single_run("s")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hamcam._runs == []


def test_closing_the_inner_of_two_cameras_keeps_the_outer(parts, tmp_path):
    inner, expt, camera = parts()
    other = FakeHamCam(1)
    expt.cameras.append(other)
    outer = PycafCamera(
        expt,
        other,
        key="Tweezer",
        element="CaF",
        data_root=tmp_path,
        frame_settle_s=0.03,
        frame_poll_s=0.005,
    )
    inner.close()
    expt.motmaster_single_run("s")
    assert inner._runs == []
    assert [len(stack) for stack in outer.store_images()] == [1]


def test_key_equal_to_motmaster_element_is_refused(tmp_path):
    expt = FakeExperiment(tmp_path, [])
    with pytest.raises(ValueError, match="MOTMaster's own prefix"):
        PycafCamera(expt, FakeHamCam(1), key="CaF", element="CaF", data_root=tmp_path)


def test_run_started_outside_pycaf_between_scans_is_ignored(parts, tmp_path):
    hamcam, expt, camera = parts(frames=2)
    expt.scan_motmaster_parameter("s", "p", [1])
    # A run from the MOTMaster window: the camera fires and MOTMaster saves a zip,
    # but not through expt._go.
    type(expt)._go(expt, {})
    expt.scan_motmaster_parameter("s", "p", [2])

    images = hamcam.store_images()

    assert [stack[:, 0, 0].tolist() for stack in images] == [[0, 1], [4, 5]]
    assert [p.name.split("_")[-1] for p in hamcam.last_zips] == ["000.zip", "002.zip"]
    folder = hamcam.last_zips[0].parent
    assert not any(
        p.name.startswith("Rb") and p.name.endswith("_001.zip")
        for p in folder.iterdir()
    )
