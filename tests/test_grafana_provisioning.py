"""The checked-in Grafana files parse and point at the provisioned data source."""

import json
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

GRAFANA = Path(__file__).parents[1] / "deploy" / "grafana"


def _datasource_refs(node):
    if isinstance(node, dict):
        if set(node) == {"type", "uid"}:
            yield node["type"], node["uid"]
        for value in node.values():
            yield from _datasource_refs(value)
    elif isinstance(node, list):
        for value in node:
            yield from _datasource_refs(value)


def test_dashboards_use_the_provisioned_datasource():
    provisioned = yaml.safe_load(
        (GRAFANA / "provisioning" / "datasources" / "pytweezer.yaml").read_text()
    )["datasources"]
    known = {(source["type"], source["uid"]) for source in provisioned}

    dashboards = sorted((GRAFANA / "dashboards").glob("*.json"))
    assert dashboards
    for path in dashboards:
        refs = set(_datasource_refs(json.loads(path.read_text())))
        assert refs and refs <= known, path.name


def _dashboards():
    return [
        json.loads(path.read_text())
        for path in sorted((GRAFANA / "dashboards").glob("*.json"))
    ]


def test_dashboard_uids_are_unique():
    uids = [dashboard["uid"] for dashboard in _dashboards()]
    assert len(uids) == len(set(uids))


def test_run_dashboard_takes_the_rid_the_gui_links_with():
    from pytweezer.GUI.grafana import RUN_DASHBOARD_UID

    (run,) = [d for d in _dashboards() if d["uid"] == RUN_DASHBOARD_UID]
    assert "rid" in {var["name"] for var in run["templating"]["list"]}


def test_alert_rules_query_the_provisioned_datasource_or_expressions():
    groups = yaml.safe_load(
        (GRAFANA / "provisioning" / "alerting" / "pytweezer.yaml").read_text()
    )["groups"]
    rules = [rule for group in groups for rule in group["rules"]]
    assert rules
    for rule in rules:
        sources = {query["datasourceUid"] for query in rule["data"]}
        assert sources <= {"pytweezer", "__expr__"}, rule["title"]
        assert rule["condition"] in {query["refId"] for query in rule["data"]}
