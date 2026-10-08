"""Links from the GUI into the Grafana run dashboard."""

from datetime import UTC, datetime
from urllib.parse import parse_qs, urlsplit

from pytweezer.GUI.grafana import run_url

BASE = "http://grafana.lab:3000"


def _query(url):
    parts = urlsplit(url)
    return parts.path, {k: v[0] for k, v in parse_qs(parts.query).items()}


def _ms(iso):
    return int(datetime.fromisoformat(iso).timestamp() * 1000)


def test_finished_run_window_has_ten_percent_margins():
    url = run_url(
        12, "2026-10-08T10:00:00+01:00", "2026-10-08T11:00:00+01:00", base=BASE
    )

    path, query = _query(url)
    assert url.startswith(BASE + "/")
    assert path == "/d/pytweezer-run"
    assert query["var-rid"] == "12"
    assert int(query["from"]) == _ms("2026-10-08T09:54:00+01:00")
    assert int(query["to"]) == _ms("2026-10-08T11:06:00+01:00")


def test_short_run_gets_at_least_thirty_seconds_either_side():
    _, query = _query(
        run_url(1, "2026-10-08T10:00:00+00:00", "2026-10-08T10:00:05+00:00", base=BASE)
    )
    assert int(query["from"]) == _ms("2026-10-08T09:59:30+00:00")
    assert int(query["to"]) == _ms("2026-10-08T10:00:35+00:00")


def test_running_task_runs_up_to_now():
    now = datetime(2026, 10, 8, 10, 10, tzinfo=UTC)
    for unfinished in (None, ""):
        _, query = _query(
            run_url(3, "2026-10-08T10:00:00+00:00", unfinished, base=BASE, now=now)
        )
        assert int(query["to"]) == _ms("2026-10-08T10:11:00+00:00")


def test_accepts_datetimes():
    start = datetime(2026, 10, 8, 10, tzinfo=UTC)
    end = datetime(2026, 10, 8, 10, 10, tzinfo=UTC)
    _, query = _query(run_url(4, start, end, base=BASE))
    assert int(query["from"]) == int(start.timestamp() * 1000) - 60_000


def test_base_url_defaults_to_config(monkeypatch):
    from pytweezer.GUI import grafana

    monkeypatch.setitem(grafana.GRAFANA, "url", "http://pc:3000/")
    url = run_url(5, "2026-10-08T10:00:00+00:00", "2026-10-08T10:01:00+00:00")
    assert url.startswith("http://pc:3000/d/pytweezer-run?")
