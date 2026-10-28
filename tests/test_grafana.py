"""GUI links into Grafana, and the checked-in dashboards and provisioning."""

import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from pytweezer.GUI import grafana
from pytweezer.GUI.grafana import RUN_DASHBOARD_UID, run_url

BASE = "http://grafana.lab:3000"
DEPLOY = Path(__file__).parents[1] / "deploy" / "grafana"
DAY = "2026-10-08T"
NOW = datetime(2026, 10, 8, 10, 10, tzinfo=UTC)


def _ms(time_of_day):
    return int(datetime.fromisoformat(DAY + time_of_day).timestamp() * 1000)


def _provisioned(kind):
    yaml = pytest.importorskip("yaml")
    path = DEPLOY / "provisioning" / kind / "pytweezer.yaml"
    return yaml.safe_load(path.read_text())


def _dashboards():
    return {
        path.name: json.loads(path.read_text())
        for path in sorted((DEPLOY / "dashboards").glob("*.json"))
    }


def _datasource_refs(node):
    if isinstance(node, dict):
        if set(node) == {"type", "uid"}:
            yield node["type"], node["uid"]
        for value in node.values():
            yield from _datasource_refs(value)
    elif isinstance(node, list):
        for value in node:
            yield from _datasource_refs(value)


@pytest.mark.parametrize(
    "t_start, t_end, shown",
    [
        ("10:00:00+01:00", "11:00:00+01:00", ("09:54:00+01:00", "11:06:00+01:00")),
        ("10:00:00+00:00", "10:00:05+00:00", ("09:59:30+00:00", "10:00:35+00:00")),
        ("10:00:00+00:00", None, ("09:59:00+00:00", "10:11:00+00:00")),
        ("10:00:00+00:00", "", ("09:59:00+00:00", "10:11:00+00:00")),
    ],
    ids=[
        "ten-percent-margins",
        "thirty-second-minimum",
        "running",
        "running-empty-end",
    ],
)
def test_run_url_shows_the_run_window_with_margins(t_start, t_end, shown):
    url = run_url(12, DAY + t_start, t_end and DAY + t_end, base=BASE, now=NOW)

    parts = urlsplit(url)
    query = {k: v[0] for k, v in parse_qs(parts.query).items()}
    assert url.startswith(BASE + "/")
    assert parts.path == f"/d/{RUN_DASHBOARD_UID}"
    assert query["var-rid"] == "12"
    assert (int(query["from"]), int(query["to"])) == tuple(map(_ms, shown))


def test_run_url_accepts_datetimes():
    times = (DAY + "10:00:00+00:00", DAY + "10:10:00+00:00")
    as_datetimes = map(datetime.fromisoformat, times)
    assert run_url(4, *as_datetimes, base=BASE) == run_url(4, *times, base=BASE)


def test_run_url_base_defaults_to_config(monkeypatch):
    monkeypatch.setitem(grafana.GRAFANA, "url", "http://pc:3000/")
    url = run_url(5, DAY + "10:00:00+00:00", DAY + "10:01:00+00:00")
    assert url.startswith(f"http://pc:3000/d/{RUN_DASHBOARD_UID}?")


def test_dashboards_have_unique_uids_and_use_the_provisioned_datasource():
    known = {
        (source["type"], source["uid"])
        for source in _provisioned("datasources")["datasources"]
    }
    dashboards = _dashboards()
    assert dashboards
    for name, dashboard in dashboards.items():
        refs = set(_datasource_refs(dashboard))
        assert refs and refs <= known, name

    uids = [dashboard["uid"] for dashboard in dashboards.values()]
    assert len(uids) == len(set(uids))


def test_run_dashboard_takes_the_rid_the_gui_links_with():
    (run,) = [d for d in _dashboards().values() if d["uid"] == RUN_DASHBOARD_UID]
    assert "rid" in {var["name"] for var in run["templating"]["list"]}


def test_alert_rules_query_the_provisioned_datasource_or_expressions():
    groups = _provisioned("alerting")["groups"]
    rules = [rule for group in groups for rule in group["rules"]]
    assert rules
    for rule in rules:
        sources = {query["datasourceUid"] for query in rule["data"]}
        assert sources <= {"pytweezer", "__expr__"}, rule["title"]
        assert rule["condition"] in {query["refId"] for query in rule["data"]}


def test_health_dashboard_shows_lab_setup_per_machine():
    health = _dashboards()["pytweezer-health.json"]
    (panel,) = [p for p in health["panels"] if p.get("title") == "Lab setup"]
    sql = panel["targets"][0]["rawSql"]
    assert "FROM machine_changes" in sql
