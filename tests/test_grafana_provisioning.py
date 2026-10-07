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
