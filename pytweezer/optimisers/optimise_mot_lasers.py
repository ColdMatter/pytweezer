"""Tune the CaF MOT lasers, one at a time, for the brightest fluorescence."""

from __future__ import annotations

from typing import ClassVar

from pytweezer.experiment.client import get_saved_parameters, set_saved_parameters
from pytweezer.experiment.optimisers import Stage, StagedScan
from pytweezer.experiments.caf_fluorescence import LASERS, CaFFluorescence
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
    """Scan each CaF laser about its saved set point for the brightest MOT.

    One stage per laser, in :data:`~pytweezer.experiments.caf_fluorescence.LASERS`
    order, each scanning that laser within its span of ``<laser>_centre``
    (the saved ``caf_laser_<laser>`` unless set) and proposing the set point
    with the most ``counts``; later lasers are scanned with the earlier ones at
    their best. v0, v1 and TCool are scanned by default. Every saved
    ``caf_laser_*`` must exist, enabled stage or not
    (:func:`create_saved_set_points` makes them).

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
    def apply(cls, proposals: dict[str, float]) -> None:
        """Ramp each laser whose saved set point was accepted to its new value.

        Args:
            proposals: ``{saved key: THz}``; only these lasers are moved.
        """
        laser_of = {key: laser for laser, key in LASER_KEYS.items()}
        motmaster = get_device("CaF MotMaster")
        try:
            for key, set_point in proposals.items():
                motmaster.ramp_laser_set_point_wml(laser_of[key], float(set_point))
        finally:
            motmaster.close_rpc()


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
