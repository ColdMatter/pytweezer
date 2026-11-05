"""Tune the CaF MOT lasers, one at a time, for the brightest fluorescence."""

from __future__ import annotations

from typing import ClassVar

from pytweezer.experiment.arguments import Argument, Number
from pytweezer.experiment.client import get_saved_parameters, set_saved_parameters
from pytweezer.experiment.optimisers import SEARCH, Stage, StagedScan
from pytweezer.experiments.caf_fluorescence import CaFFluorescence
from pytweezer.experiments.caf_mot import LASERS
from pytweezer.logging_utils import get_logger
from pytweezer.servers.device_client import get_device

logger = get_logger(__name__)

#: The saved parameter holding each laser's set point (THz).
LASER_KEYS = {laser: f"caf_laser_{laser}" for laser in LASERS}

LASER_GROUP = "CaF lasers"

#: ``(span in THz, points, scanned by default)`` of each laser's stage.
_SCANS = {
    "v0": (10e-6, 30, True),
    "BX": (30e-6, 30, False),
    "v1": (40e-6, 20, True),
    "v2": (150e-6, 30, False),
    "v3": (150e-6, 30, False),
    "TCool": (40e-6, 30, True),
}


class OptimiseMOTLasers(StagedScan, CaFFluorescence):
    """Scan each CaF laser about its current set point for the brightest MOT.

    One stage per laser, in :data:`~pytweezer.experiments.caf_mot.LASERS`
    order, each scanning that laser within its span of the set point the lock
    held when the run started plus ``<laser>_offset``, and proposing the set
    point with the most ``counts``; later lasers are scanned with the earlier
    ones at their best. v0, v1 and TCool are scanned by default. The saved
    ``caf_laser_*`` are only where a proposal is saved on acceptance; a scan
    never reads them.

    Accepting a proposal also ramps that laser to it (:meth:`apply`).
    """

    objective = "counts"
    goal = "max"
    stages: ClassVar[list[Stage]] = [
        Stage(laser, span=span, points=points, save_as=LASER_KEYS[laser], run=run)
        for laser, (span, points, run) in _SCANS.items()
    ]
    proposal_units: ClassVar[dict[str, str]] = dict.fromkeys(LASER_KEYS.values(), "THz")
    proposal_group = LASER_GROUP

    @classmethod
    def centre_argument(cls, stage: Stage) -> tuple[str, Argument]:
        return f"{stage.setting}_offset", Number(
            0.0,
            group=SEARCH,
            tooltip="added to the set point the lock held when the run started",
        )

    def centre_of(self, stage: Stage) -> float:
        return self.start_set_points[stage.setting] + getattr(
            self, f"{stage.setting}_offset"
        )

    @classmethod
    def apply(cls, proposals: dict[str, float]) -> None:
        """Ramp each laser whose saved set point was accepted to its new value.

        Every laser is tried even if one fails.

        Args:
            proposals: ``{saved key: THz}``; only these lasers are moved.

        Raises:
            KeyError: Before any laser moves, if a key is not a ``caf_laser_*``.
            RuntimeError: Naming the lasers that could not be ramped.
        """
        laser_of = {key: laser for laser, key in LASER_KEYS.items()}
        if unknown := sorted(set(proposals) - set(laser_of)):
            raise KeyError(f"not laser set points: {unknown}")
        targets = {laser_of[key]: float(value) for key, value in proposals.items()}
        errors: dict[str, Exception] = {}
        motmaster = get_device("CaF MotMaster")
        try:
            for laser, set_point in targets.items():
                try:
                    motmaster.ramp_laser_set_point_wml(laser, set_point)
                except Exception as error:
                    logger.exception("Could not ramp %s to %s THz", laser, set_point)
                    errors[laser] = error
        finally:
            motmaster.close_rpc()
        if errors:
            missed = {laser: targets[laser] for laser in errors}
            raise RuntimeError(
                f"could not ramp lasers to their accepted set points (THz) {missed}"
            ) from next(iter(errors.values()))


def create_saved_set_points(set_points: dict[str, float] | None = None) -> list[str]:
    """Create the ``caf_laser_*`` saved parameters that do not exist yet.

    Run once before the first :class:`OptimiseMOTLasers` run. A key that exists
    is left as it is.

    Args:
        set_points: ``{laser: THz}`` to create the missing keys from; the
            lock's current set points, read from ``CaF MotMaster``, if ``None``.

    Returns:
        The saved keys created.

    Raises:
        KeyError: If ``set_points`` names a laser not in ``LASERS``.
    """
    if set_points is None:
        motmaster = get_device("CaF MotMaster")
        try:
            set_points = motmaster.laser_set_points_wml()
        finally:
            motmaster.close_rpc()
    if unknown := sorted(set(set_points) - set(LASER_KEYS)):
        raise KeyError(f"unknown lasers {unknown}; the lasers are {list(LASERS)}")
    existing = get_saved_parameters()
    values = {
        LASER_KEYS[laser]: float(set_point)
        for laser, set_point in set_points.items()
        if LASER_KEYS[laser] not in existing
    }
    if values:
        set_saved_parameters(
            values,
            "initial laser set points",
            create=True,
            group=LASER_GROUP,
            unit="THz",
        )
        logger.info("Created saved set points %s", values)
    return list(values)
