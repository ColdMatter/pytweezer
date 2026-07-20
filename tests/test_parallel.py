"""Tests for pytweezer.parallel (run_parallel / after).

Pure Python — no hardware, no Qt. Timing assertions use generous margins so they
stay reliable under CI load.
"""

import threading
import time

import pytest

from pytweezer.parallel import after, run_parallel


def test_calls_run_concurrently():
    """Three 0.2 s sleeps finish in ~0.2 s wall time, not ~0.6 s."""
    def sleeper():
        time.sleep(0.2)

    start = time.perf_counter()
    run_parallel(sleeper, sleeper, sleeper)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5  # sequential would be ~0.6 s


def test_results_returned_in_call_order():
    """Result order matches call order regardless of finish order."""
    def slow():
        time.sleep(0.15)
        return "slow"

    def fast():
        return "fast"

    assert run_parallel(slow, fast, slow) == ["slow", "fast", "slow"]


def test_args_via_lambda_and_partial():
    from functools import partial

    assert run_parallel(lambda: 1 + 1, partial(pow, 2, 3)) == [2, 8]


def test_empty_returns_empty_list():
    assert run_parallel() == []


def test_after_staggers_start():
    """A call wrapped in after() records its timestamp last, at least the delay later."""
    events = []
    lock = threading.Lock()

    def record(label):
        with lock:
            events.append((label, time.perf_counter()))

    run_parallel(
        lambda: record("a"),
        lambda: record("b"),
        after(0.15, lambda: record("c")),
    )

    times = {label: t for label, t in events}
    assert set(times) == {"a", "b", "c"}
    assert times["c"] >= times["a"] + 0.1
    assert times["c"] >= times["b"] + 0.1
    assert max(times, key=times.get) == "c"


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

    messages = sorted(str(exc) for exc in excinfo.value.exceptions)
    assert messages == ["a", "b"]


def test_timeout_raises():
    def slow():
        time.sleep(1.0)

    start = time.perf_counter()
    with pytest.raises(TimeoutError):
        run_parallel(slow, timeout=0.1)
    assert time.perf_counter() - start < 0.8  # gave up early, didn't wait out the sleep
