"""NI analog-output (DAC) driver.

Only the simulated backend exists so far. :class:`SimulatedNIDAC` remembers the
last voltage written to each channel and logs every write, which is enough to
exercise a feedback coordinator end to end without hardware.

The real driver belongs here too, alongside it. It mirrors
``pytweezer/loggers/ni_adc_logger.py``'s task ownership — lazily ``import nidaqmx``,
hold one ``nidaqmx.Task``, add a channel per entry with
``task.ao_channels.add_ao_voltage_chan(channel)``, and write with ``task.write(...)``.
Its config key is already wired up as ``"driver": "nidac"``
(``pytweezer/servers/device_server.py``), which raises until the real class lands.

Values returned by these methods cross a sipyco RPC boundary, so they must stay
PYON-encodable: plain floats, dicts and lists only.
"""

from pytweezer.logging_utils import get_logger

LOGGER = get_logger("ni dac")


class SimulatedNIDAC:
    """Synthetic analog-output device: stores the last value written per channel.

    Writing to a channel outside ``channels`` raises, matching the real device's
    refusal to drive an unconfigured line.
    """

    def __init__(self, channels=None):
        self.channels = list(channels or [])
        self._values = {channel: 0.0 for channel in self.channels}
        self._closed = False

    def _require_channel(self, channel):
        if self._closed:
            raise RuntimeError("SimulatedNIDAC has been closed")
        if channel not in self._values:
            raise KeyError(
                f"Channel {channel!r} is not configured; known channels: "
                f"{sorted(self._values) or '(none)'}"
            )

    def set_voltage(self, channel, value):
        """Drive one channel and return the value actually written."""
        self._require_channel(channel)
        value = float(value)
        self._values[channel] = value
        LOGGER.debug("SimulatedNIDAC %s <- %.6f V", channel, value)
        return value

    def set_voltages(self, values):
        """Drive several channels at once from a ``{channel: value}`` mapping."""
        return {
            channel: self.set_voltage(channel, value)
            for channel, value in values.items()
        }

    def get_last_values(self):
        """Return ``{channel: last_value_written}``."""
        return dict(self._values)

    def close(self):
        self._closed = True
