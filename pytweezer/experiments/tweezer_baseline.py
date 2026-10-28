"""Stochastic loading of the tweezer array, imaged twice per shot on the ImagEM X2."""

from typing import Any

from pytweezer.experiment import Device, Integer, Text
from pytweezer.experiment.experiment_parameter_manager import ExpParameterManager
from pytweezer.experiment.motmaster import (
    MotMaster,
    MotMasterExperiment,
    MotMasterInteger,
    MotMasterNumber,
)

#: MOTMaster script parameter -> its saved value in parameters.json. Both
#: imaging pulses use the one saved imaging setting.
SAVED_MOTMASTER_PARAMETERS = {
    "tMOTLoadRepVCO": "tMOTLoadRepVCO",
    "tMOTLoadCoolVCO": "tMOTLoadCoolVCO",
    "tMOTLoadRepVVA": "tMOTLoadRepVVA",
    "tMOTLoadCoolVVA": "tMOTLoadCoolVVA",
    "tMolCoolVCO1": "tMolCoolVCO1",
    "tMolRepVCO1": "tMolRepVCO1",
    "tMolCoolVVA1": "tMolCoolVVA1",
    "tMolRepVVA1": "tMolRepVVA1",
    "tMolCoolVCO2": "tMolCoolVCO2",
    "tMolRepVCO2": "tMolRepVCO2",
    "tMolCoolVVA2": "tMolCoolVVA2",
    "tMolRepVVA2": "tMolRepVVA2",
    "tImgCoolVCO1": "tImgCoolVCO",
    "tImgCoolVVA1": "tImgCoolVVA",
    "tImgCoolVCO2": "tImgCoolVCO",
    "tImgCoolVVA2": "tImgCoolVVA",
    "tMOTShimXccValue": "tMOTShimXccValue",
    "tMOTShimYccValue": "tMOTShimYccValue",
    "tMOTShimZccValue": "tMOTShimZccValue",
    "tMolassesShimXccValue": "tMolassesShimXccValue",
    "tMolassesShimYccValue": "tMolassesShimYccValue",
    "tMolassesShimZccValue": "tMolassesShimZccValue",
}


class TweezerBaseline(MotMasterExperiment):
    """Baseline loading of the tweezer array: one shot per point, two images each.

    The cooling and shim settings are read from parameters.json when the task
    starts (recorded as the ``motmaster_parameters`` constant); any of them can
    still be overridden by its dotted name, e.g. ``rb.tMolCoolVCO1``. With
    ``background_drop`` set the second image is taken after the atoms are
    dropped. Run the shots as ``Scan(repetitions=...)``.
    """

    rb = MotMaster("Rb MotMaster", script="RbTweezerBasic2026_2")
    camera = Device("Rb HamCam")

    tweezer_vva_0 = MotMasterNumber(0.0, parameter="tTweezerSetVVA0")
    tweezer_vva_1 = MotMasterNumber(0.0, parameter="tTweezerSetVVA1")
    background_drop = MotMasterInteger(1, parameter="BackgroundDrop", min=0, max=1)

    sensitivity = Integer(600, min=0, group="Camera", tooltip="EM gain sensitivity")
    roi_x0 = Integer(110, min=0, group="Camera")
    roi_y0 = Integer(220, min=0, group="Camera")
    roi_width = Integer(256, min=1, group="Camera")
    roi_height = Integer(256, min=1, group="Camera")

    parameters_file = Text(
        "",
        tooltip="parameters.json holding the saved MOTMaster settings; "
        "empty for this checkout's own",
    )

    def motmaster_parameters(self) -> dict[str, dict[str, Any]]:
        parameters = super().motmaster_parameters()
        parameters["rb"] = self._saved_parameters | parameters["rb"]
        return parameters

    def prepare(self) -> None:
        saved = (
            ExpParameterManager(self.parameters_file)
            if self.parameters_file
            else ExpParameterManager()
        )
        self._saved_parameters = {
            parameter: saved.get_parameter(name)
            for parameter, name in SAVED_MOTMASTER_PARAMETERS.items()
        }
        super().prepare()
        self.record("motmaster_parameters", self.motmaster_parameters()["rb"])

        self.camera.set_roi(self.roi_x0, self.roi_width, self.roi_y0, self.roi_height)
        self.camera.set_external_exposure_mode()
        self.camera.enable_em_gain(True)
        self.camera.enable_direct_em_gain(True)
        self.camera.set_sensitivity(self.sensitivity)
        self.camera.setup_acquisition("snap", 2)
        conversion_factor, conversion_offset = self.camera.get_conversion_factor()
        self.record("conversion_factor", conversion_factor)
        self.record("conversion_offset", conversion_offset)

    def run_point(self) -> None:
        self.camera.start_acquisition()
        self.run_sequences()
        first_image, second_image = self.camera.acquire_n_frames(2)
        self.camera.stop_acquisition()
        self.record("first_image", first_image)
        self.record("second_image", second_image)
        self.record(
            "signal_counts",
            float(first_image.sum(dtype=float) - second_image.sum(dtype=float)),
            unit="counts",
        )

    def finish(self) -> None:
        self.camera.stop_acquisition()
