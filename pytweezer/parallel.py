"""Run several blocking device calls concurrently, without ``async``/``await``.

Some experiment steps need blocking device calls issued *at the same time* rather
than back-to-back. The canonical case: arm a camera, start one MotMaster in
trigger mode (its ``Go()`` blocks, waiting on a hardware trigger), and start a
second MotMaster whose sequence emits that trigger. The armed MotMaster's call
will not return until the second one fires, so the two must be launched
concurrently::

    from pytweezer.parallel import run_parallel, after
    from pytweezer.servers.device_client import get_device

    cam = get_device("Rb HamCam")
    mm1 = get_device("Rb MotMaster Server")
    mm2 = get_device("CaF MotMaster Server")

    cam.start_acquisition()          # arm the camera (returns immediately)
    mm1.set_trigger_mode(True)       # mm1's Go() will wait for a hardware trigger

    frame, _, _ = run_parallel(
        lambda: cam.acquire_n_frames(1),               # blocks reading the frame
        mm1.start_motmaster_experiment,                # armed; waits for trigger
        after(0.05, mm2.start_motmaster_experiment),   # fires 50 ms later
    )

Each call runs in its own thread. Every device is a separate server process, so
a :func:`~pytweezer.servers.device_client.get_device` call is just a blocking
socket round-trip whose ``recv`` releases the GIL — the threads overlap for real,
and this needs no ``AsyncioClient``/``asyncio.gather`` and works from the PyQt5
GUI (which has no ``qasync``). :func:`~pytweezer.servers.device_client.get_device_async`
+ ``asyncio.gather`` remains available as the lower-level async alternative.

**Each parallel call must use a different client.** A single sipyco ``Client`` is
not safe to share across threads; give each concurrent call its own device
handle (the camera + two-MotMaster pattern above already does).
"""

import threading
import time

__all__ = ["run_parallel", "after"]


def after(delay, call):
    """Wrap ``call`` so it waits ``delay`` seconds (inside its own thread) first.

    Use this to stagger a call within :func:`run_parallel` — e.g. hold the
    triggering MotMaster briefly so the armed one is waiting before the trigger
    fires. The sleep happens on the call's own thread, so it does not delay the
    other parallel calls. Returns a zero-argument callable that forwards
    ``call``'s return value.
    """
    def _delayed():
        time.sleep(delay)
        return call()

    return _delayed


def run_parallel(*calls, timeout=None):
    """Run zero-argument callables concurrently, one thread each; return results in order.

    Parameters
    ----------
    *calls:
        Callables taking no arguments. Supply arguments with a ``lambda`` or
        :func:`functools.partial`, and use :func:`after` to stagger a call's
        start.
    timeout:
        Optional overall deadline in seconds for *all* calls to finish. On expiry
        a :class:`TimeoutError` is raised; the still-running threads are daemon
        threads and are abandoned (Python threads cannot be force-killed).
        ``None`` (default) waits indefinitely. Each client's own socket timeout
        still applies independently.

    Returns
    -------
    list
        Results in the same order as ``calls`` (regardless of finish order).

    Raises
    ------
    The call's own exception if exactly one call fails; an
    :class:`ExceptionGroup` of them if several fail. Either way, every call is
    waited on before raising.
    """
    if not calls:
        return []

    results = [None] * len(calls)
    errors = [None] * len(calls)

    def worker(index, call):
        try:
            results[index] = call()
        except Exception as exc:  # noqa: BLE001 - surfaced below, per-call
            errors[index] = exc

    threads = [
        threading.Thread(
            target=worker, args=(i, call), name=f"run_parallel-{i}", daemon=True
        )
        for i, call in enumerate(calls)
    ]
    for thread in threads:
        thread.start()

    deadline = None if timeout is None else time.monotonic() + timeout
    for thread in threads:
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        thread.join(remaining)
        if thread.is_alive():
            still_running = sum(t.is_alive() for t in threads)
            raise TimeoutError(
                f"run_parallel timed out after {timeout}s "
                f"with {still_running} call(s) still running"
            )

    raised = [exc for exc in errors if exc is not None]
    if len(raised) == 1:
        raise raised[0]
    if raised:
        raise ExceptionGroup("run_parallel: multiple calls failed", raised)

    return results
