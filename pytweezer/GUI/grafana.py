"""Links from the GUI into the Grafana dashboards in ``deploy/grafana/``."""

from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode

from pytweezer.configuration.config import GRAFANA

RUN_DASHBOARD_UID = "pytweezer-run"
_MIN_MARGIN = timedelta(seconds=30)


def run_url(rid, t_start, t_end=None, *, base=None, now=None):
    """URL of the run dashboard for ``rid``, showing its time window plus margins.

    Times are datetimes or ISO strings with an offset. A missing or empty
    ``t_end`` means the run is still going, so the window runs up to ``now``.
    """
    start = _datetime(t_start)
    end = _datetime(t_end) or now or datetime.now(UTC)
    margin = max((end - start) * 0.1, _MIN_MARGIN)
    query = urlencode(
        {
            "var-rid": rid,
            "from": _epoch_ms(start - margin),
            "to": _epoch_ms(end + margin),
        }
    )
    base = (base or GRAFANA["url"]).rstrip("/")
    return f"{base}/d/{RUN_DASHBOARD_UID}?{query}"


def open_in_browser(url):
    from PyQt6.QtCore import QUrl
    from PyQt6.QtGui import QDesktopServices

    QDesktopServices.openUrl(QUrl(url))


def _datetime(value):
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _epoch_ms(moment):
    return int(moment.timestamp() * 1000)
