import threading
import time

import pytest

from pytweezer.parallel import after, run_parallel


def test_calls_run_concurrently_and_return_in_call_order():
    def slow():
        time.sleep(0.3)
        return "slow"

    start = time.perf_counter()
    assert run_parallel(slow, lambda: "fast", slow) == ["slow", "fast", "slow"]
    assert time.perf_counter() - start < 0.5  # sequential would be ~0.6 s


def test_after_staggers_start():
    events = {}
    lock = threading.Lock()

    def record(label):
        with lock:
            events[label] = time.perf_counter()

    run_parallel(
        lambda: record("a"),
        lambda: record("b"),
        after(0.15, lambda: record("c")),
    )
    assert events["c"] >= max(events["a"], events["b"]) + 0.1


def test_single_exception_propagates():
    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        run_parallel(boom, lambda: 1)


def test_multiple_exceptions_grouped():
    def boom(msg):
        raise ValueError(msg)

    with pytest.raises(ExceptionGroup) as excinfo:
        run_parallel(lambda: boom("a"), lambda: 1, lambda: boom("b"))
    assert sorted(str(exc) for exc in excinfo.value.exceptions) == ["a", "b"]


def test_timeout_raises():
    start = time.perf_counter()
    with pytest.raises(TimeoutError):
        run_parallel(lambda: time.sleep(1.0), timeout=0.1)
    assert time.perf_counter() - start < 0.8  # gave up early, didn't wait out the sleep
