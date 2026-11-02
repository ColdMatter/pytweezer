import time

import pytest

from pytweezer.GUI import streammonitor as sm


class FakeStream:
    def __init__(self, name):
        self.queue = []

    def subscribe(self, topic):
        pass

    def has_new_data(self, timeout_ms=1):
        return bool(self.queue)

    def recv(self):
        return self.queue.pop(0)


def _log(level, message):
    return ("Logs", {"level": level, "message": message, "timestamp": ""})


def test_stream_monitor_lists_newest_first_and_filters(qapp, monkeypatch):
    monkeypatch.setattr(sm, "DataClient", FakeStream)
    monitor = sm.StreamMonitor("x", "Data")
    monitor.show()
    monitor.stream.queue += [("a/one", 1), ("b/two", 2)]
    monitor._update_list()
    assert monitor.table.item(0, 1).text() == "b/two"
    monitor.filter.setText("a/")
    assert monitor.table.isRowHidden(0) and not monitor.table.isRowHidden(1)
    monitor.pause.setChecked(True)
    monitor.stream.queue.append(("a/three", 3))
    monitor._update_list()
    assert monitor.table.rowCount() == 2


def test_log_monitor_filters_by_level_and_text(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(sm, "MessageClient", FakeStream)
    monkeypatch.setattr(sm, "get_daily_log_path", lambda: tmp_path / "none.jsonl")
    monitor = sm.LogMonitor("x")
    monitor.stream.queue += [_log("INFO", "fine"), _log("ERROR", "camera lost")]
    monitor._update_list()
    hidden = lambda: [monitor.table.isRowHidden(r) for r in range(2)]
    assert hidden() == [False, False]
    monitor.min_level.setCurrentIndex(2)  # errors only
    assert hidden() == [False, True]  # newest (the error) first
    monitor.min_level.setCurrentIndex(0)
    monitor.filter.setText("fine")
    assert hidden() == [True, False]


@pytest.mark.parametrize(
    "build",
    [
        lambda: sm.StreamMonitor("stall probe", "Data"),
        lambda: sm.LogMonitor("stall probe"),
    ],
    ids=["stream", "log"],
)
def test_a_tick_with_no_messages_does_not_wait_for_them(
    qapp, monkeypatch, tmp_path, build
):
    # A 1 ms ZMQ poll blocks for a whole Windows timer tick (~16 ms); with five
    # monitors on 100 ms timers that stalled the GUI thread most of the time.
    monkeypatch.setattr(sm, "get_daily_log_path", lambda: tmp_path / "none.jsonl")
    monitor = build()
    monitor.timer.stop()

    start = time.perf_counter()
    for _ in range(10):
        monitor._update_list()
    elapsed = time.perf_counter() - start

    assert elapsed < 0.005
