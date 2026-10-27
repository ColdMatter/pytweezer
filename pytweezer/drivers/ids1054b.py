"""RS PRO IDS-1054B four-channel oscilloscope driver.

The IDS-1054B is a rebadged GW Instek GDS-1054B and speaks that family's SCPI
dialect over a raw LAN socket on port 3000. The scope never answers a query it
does not recognise, so a mistyped command shows up as a VISA timeout rather than
an error reply; ``query(":SYST:ERR?")`` says what went wrong.
"""

from __future__ import annotations

import numpy as np
import pyvisa

from pytweezer.logging_utils import get_logger
from pytweezer.servers.simulated_device import simulate

LOGGER = get_logger("ids1054b")

CHANNELS = (1, 2, 3, 4)

#: Waveform samples are int16 at 25 counts per vertical division, with the
#: on-screen position offset already removed (checked against ``:MEAS:MAX?``).
COUNTS_PER_DIV = 25


def _check_channel(channel: int) -> int:
    channel = int(channel)
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS}, got {channel}")
    return channel


def _parse_header(text: str) -> dict[str, str]:
    """Split the ``key,value;key,value;...`` preamble that precedes waveform data."""
    return dict(
        field.split(",", 1) for field in text.strip().split(";") if "," in field
    )


class IDS1054B:
    """IDS-1054B oscilloscope over pyvisa-py.

    Args:
        address: VISA resource string, e.g. ``"TCPIP::10.59.3.50::3000::SOCKET"``.
        timeout_ms: VISA I/O timeout. Long records (10 M points) take several
            seconds to transfer, so raise this if reading those.
    """

    def __init__(self, address: str, timeout_ms: int = 5000):
        self.address = address
        self.timeout_ms = int(timeout_ms)
        self._inst = None
        self._connect()

    def _connect(self) -> None:
        rm = pyvisa.ResourceManager("@py")
        inst = rm.open_resource(
            self.address,
            read_termination="\n",
            write_termination="\n",
            timeout=self.timeout_ms,
        )
        try:
            idn = inst.query("*IDN?")
        except Exception:
            inst.close()
            raise
        self._inst = inst
        LOGGER.info("Connected to %s at %s", idn, self.address)

    def close(self) -> None:
        if self._inst is not None:
            self._inst.close()
            self._inst = None

    def write(self, command: str) -> None:
        self._inst.write(command)

    def query(self, command: str) -> str:
        return self._inst.query(command)

    def identify(self) -> str:
        return self.query("*IDN?")

    # ------------------------------------------------------------------ #
    # Acquisition control
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        self.write(":RUN")

    def stop(self) -> None:
        self.write(":STOP")

    def single(self) -> None:
        """Arm a single-shot acquisition; the scope stops once it triggers."""
        self.write(":SINGle")

    def force_trigger(self) -> None:
        self.write(":FORCe")

    def trigger_state(self) -> str:
        """One of ``ARMED``, ``READY``, ``AUTO``, ``TRIGGERED`` or ``STOP``."""
        return self.query(":TRIGger:STATe?")

    # ------------------------------------------------------------------ #
    # Settings
    # ------------------------------------------------------------------ #

    def get_channel(self, channel: int) -> dict:
        ch = f":CHANnel{_check_channel(channel)}"
        return {
            "display": self.query(f"{ch}:DISPlay?") == "ON",
            "scale": float(self.query(f"{ch}:SCALe?")),
            "position": float(self.query(f"{ch}:POSition?")),
            "coupling": self.query(f"{ch}:COUPling?"),
            "probe_ratio": float(self.query(f"{ch}:PROBe:RATio?")),
            "bandwidth_limit": self.query(f"{ch}:BWLimit?"),
        }

    def set_channel(
        self,
        channel: int,
        display: bool | None = None,
        scale: float | None = None,
        position: float | None = None,
        coupling: str | None = None,
        probe_ratio: float | None = None,
    ) -> dict:
        """Change the given vertical settings and return them all as the scope now has them.

        Args:
            channel: 1-4.
            display: Show the trace. A hidden channel cannot be read out.
            scale: Volts per division; the scope rounds to its 1-2-5 steps.
            position: Trace offset in volts.
            coupling: ``"DC"``, ``"AC"`` or ``"GND"``.
            probe_ratio: Probe attenuation, e.g. 10 for a 10x probe.
        """
        ch = f":CHANnel{_check_channel(channel)}"
        if display is not None:
            self.write(f"{ch}:DISPlay {'ON' if display else 'OFF'}")
        if scale is not None:
            self.write(f"{ch}:SCALe {float(scale):.3E}")
        if position is not None:
            self.write(f"{ch}:POSition {float(position):.3E}")
        if coupling is not None:
            self.write(f"{ch}:COUPling {coupling.upper()}")
        if probe_ratio is not None:
            self.write(f"{ch}:PROBe:RATio {float(probe_ratio):.3E}")
        return self.get_channel(channel)

    def get_timebase(self) -> dict:
        return {
            "scale": float(self.query(":TIMebase:SCALe?")),
            "position": float(self.query(":TIMebase:POSition?")),
            "record_length": int(float(self.query(":ACQuire:RECOrdlength?"))),
        }

    def set_timebase(
        self,
        scale: float | None = None,
        position: float | None = None,
        record_length: int | None = None,
    ) -> dict:
        """Change the given horizontal settings and return them all as the scope now has them.

        Args:
            scale: Seconds per division.
            position: Trigger offset in seconds.
            record_length: Points per record: 1e3, 1e4, 1e5, 1e6 or 1e7.
        """
        if scale is not None:
            self.write(f":TIMebase:SCALe {float(scale):.3E}")
        if position is not None:
            self.write(f":TIMebase:POSition {float(position):.3E}")
        if record_length is not None:
            self.write(f":ACQuire:RECOrdlength {float(record_length):.0E}")
        return self.get_timebase()

    def get_trigger(self) -> dict:
        return {
            "source": self.query(":TRIGger:SOURce?"),
            "type": self.query(":TRIGger:TYPe?"),
            "mode": self.query(":TRIGger:MODe?"),
            "slope": self.query(":TRIGger:EDGe:SLOP?"),
            "level": float(self.query(":TRIGger:LEVel?")),
        }

    def set_trigger(
        self,
        source: str | None = None,
        level: float | None = None,
        slope: str | None = None,
        mode: str | None = None,
    ) -> dict:
        """Change the given edge-trigger settings and return them all as the scope now has them.

        Args:
            source: ``"CH1"``-``"CH4"``, ``"EXT"`` or ``"LINE"``.
            level: Trigger level in volts, applied to the source as it is after
                ``source`` is set.
            slope: ``"RISE"``, ``"FALL"`` or ``"EITher"``.
            mode: ``"AUTO"`` or ``"NORMal"``.
        """
        if source is not None:
            self.write(f":TRIGger:SOURce {source.upper()}")
        if level is not None:
            self.write(f":TRIGger:LEVel {float(level):.3E}")
        if slope is not None:
            self.write(f":TRIGger:EDGe:SLOP {slope.upper()}")
        if mode is not None:
            self.write(f":TRIGger:MODe {mode.upper()}")
        return self.get_trigger()

    # ------------------------------------------------------------------ #
    # Readout
    # ------------------------------------------------------------------ #

    def read_waveform(self, channel: int) -> dict:
        """Read the displayed record of one channel.

        Returns:
            ``{"time": s, "voltage": V, "header": {...}}``, where ``time`` is
            relative to the trigger point and ``header`` is the scope's own
            preamble (all values as strings).

        Raises:
            RuntimeError: If the channel is hidden; the scope would otherwise
                return a stale record.
        """
        channel = _check_channel(channel)
        if self.query(f":CHANnel{channel}:DISPlay?") != "ON":
            raise RuntimeError(f"CH{channel} is not displayed")

        self.write(f":ACQuire{channel}:MEMory?")
        header = _parse_header(self._inst.read())
        if self._inst.read_bytes(1) != b"#":
            raise RuntimeError("waveform block did not start with '#'")
        n_digits = int(self._inst.read_bytes(1))
        n_bytes = int(self._inst.read_bytes(n_digits))
        block = self._inst.read_bytes(n_bytes + 1)  # + the trailing newline

        counts = np.frombuffer(block[:n_bytes], dtype=">i2")
        volts_per_div = float(header["Vertical Scale"])
        voltage = counts * (volts_per_div / COUNTS_PER_DIV)
        dt = float(header["Sampling Period"])
        trigger_index = int(header["Trigger Address"])
        time = (np.arange(counts.size) - trigger_index) * dt
        return {"time": time, "voltage": voltage, "header": header}


@simulate(IDS1054B)
class SimulatedIDS1054B:
    """Fake scope: remembers its settings and reads out a noisy 1 kHz sine."""

    def __init__(self, **_ignored):
        self._channels = {
            ch: {
                "display": True,
                "scale": 0.1,
                "position": 0.0,
                "coupling": "DC",
                "probe_ratio": 1.0,
                "bandwidth_limit": "FULL",
            }
            for ch in CHANNELS
        }
        self._timebase = {"scale": 5e-5, "position": 0.0, "record_length": 10000}
        self._trigger = {
            "source": "CH1",
            "type": "EDGE",
            "mode": "AUTO",
            "slope": "RISE",
            "level": 0.0,
        }
        self._rng = np.random.default_rng()

    def identify(self) -> str:
        return "RS,IDS-1054B,SIMULATED,V0"

    def trigger_state(self) -> str:
        return "AUTO"

    def get_channel(self, channel: int) -> dict:
        return dict(self._channels[_check_channel(channel)])

    def set_channel(self, channel: int, **settings) -> dict:
        self._channels[_check_channel(channel)].update(
            {k: v for k, v in settings.items() if v is not None}
        )
        return self.get_channel(channel)

    def get_timebase(self) -> dict:
        return dict(self._timebase)

    def set_timebase(self, **settings) -> dict:
        self._timebase.update({k: v for k, v in settings.items() if v is not None})
        return self.get_timebase()

    def get_trigger(self) -> dict:
        return dict(self._trigger)

    def set_trigger(self, **settings) -> dict:
        self._trigger.update({k: v for k, v in settings.items() if v is not None})
        return self.get_trigger()

    def read_waveform(self, channel: int) -> dict:
        channel = _check_channel(channel)
        n = int(self._timebase["record_length"])
        dt = 10 * self._timebase["scale"] / n
        time = (np.arange(n) - n // 2) * dt
        scale = self._channels[channel]["scale"]
        voltage = 2 * scale * np.sin(2 * np.pi * 1e3 * time)
        voltage += self._rng.normal(0, scale / COUNTS_PER_DIV, n)
        return {
            "time": time,
            "voltage": voltage,
            "header": {"Source": f"CH{channel}", "Sampling Period": f"{dt:.3e}"},
        }
