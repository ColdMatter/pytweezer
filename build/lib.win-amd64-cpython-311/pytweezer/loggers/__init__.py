"""Background InfluxDB loggers.

``Logger`` is the generic base; concrete subclasses (e.g.
:class:`~pytweezer.loggers.ni_adc_logger.NIADCLogger`) each own their data source
and are launched by ``pytweezer/servers/logger_server.py``.
"""

from pytweezer.loggers.base import Logger

__all__ = ["Logger"]
