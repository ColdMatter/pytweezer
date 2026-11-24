"""Measure the tweezers' trap frequency by parametric heating."""

from __future__ import annotations

from pytweezer.experiment.arguments import Number
from pytweezer.experiment.motmaster import MotMasterInteger
from pytweezer.experiments.load_tweezers import LoadTweezers

#: ``tTweezerModulationVCF`` spans this range, linearly onto the VCO's frequency range.
VCF_RANGE_V = (-10.0, 10.0)


def vcf_voltage(frequency: float, vco_min: float, vco_max: float) -> float:
    """The ``tTweezerModulationVCF`` voltage that sets the VCO to ``frequency``."""
    low, high = VCF_RANGE_V
    return low + (frequency - vco_min) / (vco_max - vco_min) * (high - low)


class TrapFrequencyScan(LoadTweezers):
    """Image the tweezers before and after modulating their intensity at ``modulation_frequency``.

    The atoms are heated out when the modulation is at twice a trap frequency,
    so scanning ``modulation_frequency`` and plotting ``survival`` shows dips at
    twice the radial and axial frequencies. The modulation comes from a VCO set
    by the analogue ``tTweezerModulationVCF``, which maps -10 to 10 V linearly
    onto ``vco_min_frequency`` to ``vco_max_frequency``: these must match the
    VCO's range as set on the generator. The voltage sent is recorded per point
    as ``modulation_voltage``.
    """

    modulation_frequency = Number(
        100e3,
        unit="kHz",
        scale=1e3,
        min=0,
        group="Modulation",
        tooltip="frequency of the tweezer intensity modulation",
    )
    vco_min_frequency = Number(
        10e3,
        unit="kHz",
        scale=1e3,
        min=0,
        group="Modulation",
        tooltip="VCO frequency at -10 V on tTweezerModulationVCF",
        scannable=False,
    )
    vco_max_frequency = Number(
        300e3,
        unit="kHz",
        scale=1e3,
        min=0,
        group="Modulation",
        tooltip="VCO frequency at +10 V on tTweezerModulationVCF",
        scannable=False,
    )
    hold_time = MotMasterInteger(
        2000,
        parameter="tDelay2",
        min=0,
        group="Modulation",
        tooltip="MOTMaster ticks between the two images, while the tweezers "
        "are modulated",
    )

    def prepare(self) -> None:
        if self.vco_max_frequency <= self.vco_min_frequency:
            raise ValueError(
                "vco_max_frequency must be above vco_min_frequency, not "
                f"{self.vco_max_frequency} <= {self.vco_min_frequency} Hz"
            )
        super().prepare()

    def run_point(self) -> None:
        if (
            not self.vco_min_frequency
            <= self.modulation_frequency
            <= self.vco_max_frequency
        ):
            raise ValueError(
                f"modulation_frequency {self.modulation_frequency} Hz is outside "
                f"the VCO's range, {self.vco_min_frequency} to "
                f"{self.vco_max_frequency} Hz"
            )
        voltage = vcf_voltage(
            self.modulation_frequency, self.vco_min_frequency, self.vco_max_frequency
        )
        self.record("modulation_voltage", voltage, unit="V")
        self.record_tweezers(
            self.take_shot(
                BackgroundDrop=0, TweezerModulation=1, tTweezerModulationVCF=voltage
            )
        )
