"""Standalone viewer for a Thorlabs Zelux (or other Thorlabs TSI) camera.

Puts the camera in software-trigger mode, one frame per trigger, with a settable
exposure, and shows each frame in a pyqtgraph window.

    uv run python tools/zelux_viewer.py --exposure-ms 10
    uv run python tools/zelux_viewer.py --serial 12345 --auto 200

Needs Thorlabs' Scientific Camera SDK (the "Scientific Camera Interfaces" download).
On Windows the default install location is used; elsewhere pass ``--sdk-dir`` to the
folder holding the SDK's native libraries.
"""

import argparse
import sys

import pylablib as pll
import pylablib.devices.Thorlabs as thorlabs
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

WINDOWS_SDK_DIR = (
    r"C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support"
    r"\Scientific Camera Interfaces\SDK\Python Toolkit\dlls\64_lib"
)

pg.setConfigOptions(imageAxisOrder="row-major")


def open_camera(serial: str | None, sdk_dir: str | None):
    if sdk_dir is None and sys.platform == "win32":
        sdk_dir = WINDOWS_SDK_DIR
    if sdk_dir is not None:
        pll.par["devices/dlls/thorlabs_tlcam"] = sdk_dir
    cameras = thorlabs.list_cameras_tlcam()
    if not cameras:
        raise SystemExit(
            "No Thorlabs TSI cameras found (is the SDK installed and the camera visible?)"
        )
    if serial is None:
        serial = cameras[0]
    elif serial not in cameras:
        raise SystemExit(f"Camera {serial!r} not found; available: {cameras}")
    camera = thorlabs.ThorlabsTLCamera(serial=serial)
    camera.open()
    return camera


class Viewer(QtWidgets.QWidget):
    POLL_MS = 10

    def __init__(self, camera, exposure_ms: float, auto_interval_ms: int | None):
        super().__init__()
        self.camera = camera
        self.waiting_since_ms: int | None = None
        self.frame_count = 0

        self.exposure = QtWidgets.QDoubleSpinBox()
        self.exposure.setSuffix(" ms")
        self.exposure.setDecimals(3)
        self.exposure.setRange(0.001, 60_000)
        self.exposure.setValue(exposure_ms)
        self.exposure.editingFinished.connect(self.apply_exposure)

        self.trigger_button = QtWidgets.QPushButton("Trigger")
        self.trigger_button.clicked.connect(self.trigger)

        self.auto = QtWidgets.QCheckBox("Auto-trigger every")
        self.auto_interval = QtWidgets.QSpinBox()
        self.auto_interval.setSuffix(" ms")
        self.auto_interval.setRange(1, 600_000)
        self.auto_interval.setValue(auto_interval_ms or 200)
        self.auto.setChecked(auto_interval_ms is not None)

        self.status = QtWidgets.QLabel()
        self.view = pg.ImageView()

        controls = QtWidgets.QHBoxLayout()
        for widget in (
            QtWidgets.QLabel("Exposure"),
            self.exposure,
            self.trigger_button,
            self.auto,
            self.auto_interval,
        ):
            controls.addWidget(widget)
        controls.addStretch()
        controls.addWidget(self.status)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.view)

        self.poll_timer = QtCore.QTimer(self, interval=self.POLL_MS)
        self.poll_timer.timeout.connect(self.poll)
        self.auto_timer = QtCore.QTimer(self)
        self.auto_timer.timeout.connect(self.trigger)
        self.auto.toggled.connect(self.update_auto)
        self.auto_interval.valueChanged.connect(self.update_auto)

        self.clock = QtCore.QElapsedTimer()
        self.clock.start()
        self.arm()
        self.update_auto()

    def arm(self):
        """(Re)start acquisition in software-trigger mode: one frame per trigger."""
        camera = self.camera
        camera.stop_acquisition()
        camera.set_trigger_mode("int")
        actual_s = camera.set_exposure(self.exposure.value() * 1e-3)
        self.exposure.blockSignals(True)
        self.exposure.setValue(actual_s * 1e3)
        self.exposure.blockSignals(False)
        camera.setup_acquisition(nframes=10)
        camera.start_acquisition(frames_per_trigger=1, auto_start=False)
        self.waiting_since_ms = None
        self.status.setText(f"armed, exposure {actual_s * 1e3:.3f} ms")

    def apply_exposure(self):
        self.poll_timer.stop()
        self.arm()
        self.trigger_button.setEnabled(True)

    def update_auto(self, *_):
        if self.auto.isChecked():
            self.auto_timer.start(self.auto_interval.value())
        else:
            self.auto_timer.stop()

    def trigger(self):
        if self.waiting_since_ms is not None:
            return
        self.camera.send_software_trigger()
        self.waiting_since_ms = self.clock.elapsed()
        self.trigger_button.setEnabled(False)
        self.poll_timer.start()

    def poll(self):
        frames = self.camera.read_multiple_images()
        waited_ms = self.clock.elapsed() - self.waiting_since_ms
        if not frames:
            timeout_ms = self.exposure.value() + 2000
            if waited_ms > timeout_ms:
                self.finish_wait()
                self.status.setText("timed out waiting for a frame")
            return
        image = frames[-1]
        self.frame_count += 1
        self.view.setImage(
            image,
            autoRange=self.frame_count == 1,
            autoLevels=True,
            autoHistogramRange=False,
        )
        self.finish_wait()
        self.status.setText(
            f"frame {self.frame_count}, {waited_ms} ms from trigger, "
            f"max {int(image.max())}"
        )

    def finish_wait(self):
        self.poll_timer.stop()
        self.waiting_since_ms = None
        self.trigger_button.setEnabled(True)

    def closeEvent(self, event):
        self.exposure.editingFinished.disconnect(self.apply_exposure)
        self.poll_timer.stop()
        self.auto_timer.stop()
        self.camera.close()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exposure-ms", type=float, default=10.0)
    parser.add_argument("--serial", help="camera serial number (default: first found)")
    parser.add_argument(
        "--sdk-dir", help="folder with the Thorlabs SDK native libraries"
    )
    parser.add_argument(
        "--auto",
        type=int,
        metavar="INTERVAL_MS",
        help="start with auto-trigger on, at this interval",
    )
    args = parser.parse_args()

    camera = open_camera(args.serial, args.sdk_dir)
    app = QtWidgets.QApplication(sys.argv[:1])
    viewer = Viewer(camera, args.exposure_ms, args.auto)
    viewer.setWindowTitle(f"Zelux {camera.get_device_info().serial_number}")
    viewer.resize(900, 700)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
