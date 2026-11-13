from pytweezer.drivers import thorcam
from pytweezer.drivers.camera_base import simulated_camera_for


class ThorCam(thorcam.ThorCam):
    """The ThorLabs camera that watches the tweezer array, serial number 38570 unless
    another is given. Arguments are those of :class:`pytweezer.drivers.thorcam.ThorCam`."""

    def __init__(
        self,
        image_dir: str | None = None,
        timeout: float = 5.0,
        stream_name: str | None = None,
        serial_number: str | None = "38570",
        trigger_mode: str | None = None,
        trigger_polarity: str = "rise",
        binning: int | None = None,
        gain_db: float | None = None,
        object_pixel_m: float | None = None,
    ):
        super().__init__(
            image_dir=image_dir,
            timeout=timeout,
            stream_name=stream_name,
            serial_number=serial_number,
            trigger_mode=trigger_mode,
            trigger_polarity=trigger_polarity,
            binning=binning,
            gain_db=gain_db,
            object_pixel_m=object_pixel_m,
        )


SimulatedThorLabsCamera = simulated_camera_for(ThorCam)
