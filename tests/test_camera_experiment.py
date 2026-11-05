"""CameraExperiment: marked background shots and recorded cloud fits."""

import numpy as np
import pytest

from pytweezer.experiment import Bool, run_local
from pytweezer.experiment.camera_experiment import CameraExperiment, CameraShots
from pytweezer.experiment.motmaster import MotMaster
from pytweezer.servers import device_server

ROI_ORIGIN = (110, 220)
BINNING = 2
PIXEL_M = 1e-6
CLOUD = {"x0": 40.3, "y0": 50.6, "sigma_x": 6.0, "sigma_y": 4.0, "amplitude": 800.0}


def gaussian_frame(shape=(100, 120), offset=100.0, seed=0, cloud=True):
    """Pixel ``c`` of the frame sits at ``x = c``; Poisson noise."""
    rows, cols = np.indices(shape, dtype=np.float64)
    model = offset + cloud * CLOUD["amplitude"] * np.exp(
        -0.5
        * (
            ((cols - CLOUD["x0"]) / CLOUD["sigma_x"]) ** 2
            + ((rows - CLOUD["y0"]) / CLOUD["sigma_y"]) ** 2
        )
    )
    return np.random.default_rng(seed).poisson(model).astype(np.uint16)


class FakeSequencer:
    def close_rpc(self):
        pass

    def set_linked(self, value): ...
    def set_motmaster_experiment(self, script): ...
    def set_run_until_stopped(self, value): ...
    def set_iterations(self, value): ...
    def set_save_toggle(self, value): ...
    def set_trigger_mode(self, value): ...

    def get_params(self):
        return {}

    def start_motmaster_experiment(self, parameters): ...


class FakeCamera:
    """Hands out ``frame`` ``nframes`` times, or a cloudless frame to a background shot."""

    def __init__(self):
        self.frame = gaussian_frame()
        self.background_frame = gaussian_frame(cloud=False, seed=1)
        self.headers = []
        self.binning = BINNING
        self.roi_origin = ROI_ORIGIN

    def close_rpc(self):
        pass

    def set_binning(self, binning): ...
    def set_trigger_source(self, source): ...
    def setup_acquisition(self, nframes, mode="sequence"): ...
    def start_acquisition(self): ...
    def stop_acquisition(self): ...
    def stop_live(self): ...

    def is_live(self):
        return False

    def set_roi(self, x0, width, y0, height):
        return ROI_ORIGIN

    def frame_geometry(self):
        return self.binning, self.roi_origin

    def acquire_n_frames(self, nframes, broadcast=False, header=None):
        self.headers.append(header)
        shot = self.background_frame if header.get("background") else self.frame
        return np.stack([shot] * nframes)


@pytest.fixture
def camera(monkeypatch):
    camera = FakeCamera()
    fakes = {"Rb MotMaster": FakeSequencer(), "Rb ThorCam": camera}
    monkeypatch.setattr(
        "pytweezer.experiment.experiment._get_device",
        lambda name, timeout: fakes[name],
    )
    config = device_server.get_config()
    devices = {
        **config["Devices"],
        "Rb ThorCam": {**config["Devices"]["Rb ThorCam"], "object_pixel_m": PIXEL_M},
    }
    monkeypatch.setattr(
        device_server, "get_config", lambda: {**config, "Devices": devices}
    )
    return camera


class Shots(CameraExperiment):
    rb = MotMaster("Rb MotMaster", script="RbTweezerBasic")
    thorcam = CameraShots(
        "Rb ThorCam",
        frames=2,
        binning=BINNING,
        roi=(*ROI_ORIGIN[:1], 240, ROI_ORIGIN[1], 200),
    )
    save_mean_image = Bool(True, group="Camera")
    cloud_species = "Rb"

    def run_point(self):
        background = self.take_shot(background=True)["thorcam"]
        frames = self.take_shot()["thorcam"]
        self.record_cloud_fit("thorcam", frames, background.mean(axis=0))


def run(experiment_cls=Shots, **args):
    measurement = run_local(experiment_cls, **args)
    assert measurement.status == "completed", measurement.attrs["error"]
    return measurement


def test_background_shot_marks_broadcast_headers(camera):
    run()

    assert [header.get("background") for header in camera.headers] == [True, None]
    assert all("rid" in header for header in camera.headers)


def test_point_records_cloud_fit_fields(camera):
    camera.frame = gaussian_frame()
    measurement = run()
    results = measurement.results
    resolution = PIXEL_M * BINNING
    origin_x, origin_y = ROI_ORIGIN[0] / BINNING, ROI_ORIGIN[1] / BINNING

    assert results["thorcam_sigma_x"][0] == pytest.approx(6 * resolution, rel=0.03)
    assert results["thorcam_sigma_y"][0] == pytest.approx(4 * resolution, rel=0.03)
    assert results["thorcam_x0"][0] == pytest.approx(
        (origin_x + CLOUD["x0"] + 0.5) * resolution, rel=1e-3
    )
    assert results["thorcam_y0"][0] == pytest.approx(
        (origin_y + CLOUD["y0"] + 0.5) * resolution, rel=1e-3
    )
    assert results["thorcam_sigma_x_error"][0] >= 0


def test_the_background_frame_is_subtracted_from_the_fit(camera):
    # The cloud sits on an offset of 100, which the background shot also has.
    measurement = run()

    assert abs(measurement.results["thorcam_offset"][0]) < 5


@pytest.mark.parametrize(
    "frame",
    [
        np.random.default_rng(2).poisson(100.0, (100, 120)).astype(np.uint16),
        np.full((100, 120), np.nan),
    ],
    ids=["blank", "nan"],
)
def test_point_without_a_cloud_records_nan(camera, frame):
    camera.frame = frame
    measurement = run()

    assert np.isnan(measurement.results["thorcam_sigma_x"][0])
    assert np.isnan(measurement.results["thorcam_x0_error"][0])


def test_cloud_geometry_comes_from_the_camera_not_the_declared_shots(camera):
    class Undeclared(Shots):
        thorcam = CameraShots("Rb ThorCam", frames=2)

    camera.binning, camera.roi_origin = 4, (60, 100)
    measurement = run(Undeclared)
    resolution = PIXEL_M * 4

    assert measurement.results["thorcam_sigma_x"][0] == pytest.approx(
        6 * resolution, rel=0.03
    )
    assert measurement.results["thorcam_x0"][0] == pytest.approx(
        (60 / 4 + CLOUD["x0"] + 0.5) * resolution, rel=1e-3
    )
    assert measurement.results["thorcam_y0"][0] == pytest.approx(
        (100 / 4 + CLOUD["y0"] + 0.5) * resolution, rel=1e-3
    )


@pytest.mark.parametrize("species, mass", [("CaF", 59.0), ("Rb", 86.909), (None, None)])
def test_cloud_species_records_the_mass(camera, species, mass):
    class Species(Shots):
        cloud_species = species

    measurement = run(Species)

    assert measurement.constants.get("cloud_mass_amu") == mass


@pytest.mark.parametrize("save", [True, False])
def test_mean_image_is_recorded_only_when_asked(camera, save):
    class Saving(Shots):
        save_mean_image = Bool(save, group="Camera")

    measurement = run(Saving)

    assert ("thorcam_mean_image" in measurement.results) is save
