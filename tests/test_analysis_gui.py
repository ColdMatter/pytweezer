from pytweezer.GUI.analysismanager import FilterTable, filter_state

FILTERS = {
    "Image/a": dict(
        name="a", category="Image", script="a.py", active=True, streams=["s"]
    ),
    "Data/b": dict(name="b", category="Data", script="b.py", active=False, streams=[]),
}


def test_filter_state_distinguishes_a_died_process_from_a_stopped_one():
    assert filter_state({"active": True}, running=True) == "running"
    assert filter_state({"active": True}, running=False) == "crashed"
    assert filter_state({"active": False}, running=False) == "stopped"


def test_table_updates_in_place_and_keeps_the_selection(qapp):
    table = FilterTable()
    table.update_filters(FILTERS, {"Image/a": True})
    table.selectRow(1)
    key = table.selected_key()
    item = table.item(1, 0)
    table.update_filters(FILTERS, {})
    assert table.selected_key() == key
    assert table.item(1, 0) is item  # not rebuilt
    assert [table.item(r, 1).text() for r in range(2)] == ["Stopped", "Crashed"]
