"""Example logger that owns its own device: an NI ADC read with ``nidaqmx``.

This is the template for the intended Logger model — a Logger opens and owns its
data source directly (real or virtual), rather than polling an existing device
RPC server. Here the source is a National Instruments analog-input DAQ: the
logger creates its own ``nidaqmx.Task``, reads the configured channels on each
interval, and pushes the voltages to InfluxDB.

Config entry shape (``CONFIG["Loggers"][name]``)::

    {
        "active": True,
        "script": "../pytweezer/servers/logger_server.py",
        "logger": "ni_adc",
        "host": SERVER_HOST,
        "interval": 1.0,
        "simulate": SIMULATING,             # True -> no hardware, fake readings
        "channels": ["Dev1/ai0", "Dev1/ai1"],
        "measurement": "ni_adc",            # optional; defaults to "ni_adc"
        "tags": {"system": "Rb"},           # optional static tags
    }

To add another owns-its-device logger, copy this class, override ``setup``/
``read``/``close``, and register it in ``logger_server.LOGGER_REGISTRY``.
"""

import math
import random
import time

from pytweezer.loggers.base import Logger
from pytweezer.logging_utils import get_logger

logger = get_logger("Logger")


def _short_name(channel):
    """`"Dev1/ai0"` -> `"ai0"` for a tidy field name; unchanged if no slash."""
    return channel.rsplit("/", 1)[-1]


class NIADCLogger(Logger):
    """Read analog-input voltages off an NI DAQ and log them as one measurement."""

    def setup(self):
        self.channels = list(self.conf.get("channels", []))
        self.measurement = self.conf.get("measurement", "ni_adc")
        self.tags = self.conf.get("tags")
        self.simulate = bool(self.conf.get("simulate", False))
        self._task = None
        self._t0 = time.time()

        if not self.channels:
            logger.warning("Logger %r has no 'channels' configured", self.name)
            return

        if self.simulate:
            logger.warning("NI ADC logger %r running in SIMULATION MODE", self.name)
            return

        # Lazy import so a machine without NI-DAQmx can still import this module.
        import nidaqmx

        self._task = nidaqmx.Task()
        for channel in self.channels:
            self._task.ai_channels.add_ai_voltage_chan(channel)

    def read(self):
        if not self.channels:
            return None

        if self.simulate:
            values = self._simulated_values()
        else:
            raw = self._task.read()
            # nidaqmx returns a bare float for a single channel, else a list.
            values = raw if isinstance(raw, (list, tuple)) else [raw]

        fields = {_short_name(ch): v for ch, v in zip(self.channels, values)}
        return [(self.measurement, fields, self.tags)]

    def _simulated_values(self):
        """Plausible fake readings: a slow per-channel sine plus small noise."""
        t = time.time() - self._t0
        out = []
        for i, _channel in enumerate(self.channels):
            sine = math.sin(2 * math.pi * (0.05 + 0.01 * i) * t)
            out.append(sine + random.gauss(0.0, 0.01))
        return out

    def close(self):
        try:
            if self._task is not None:
                self._task.close()
        except Exception:
            pass
        finally:
            self._task = None
            super().close()
