"""Shared state over sipyco ``sync_struct``, usable from threaded code.

One process owns a structure in a :class:`~sipyco.sync_struct.Notifier` and
serves it with :class:`SyncServer`; any number of processes, on any PC, hold a
live copy with :class:`SyncMirror`. A mirror receives the whole structure when
it connects and then every modification, in order, on one TCP stream, so a
copy can never silently drift. After a disconnect it reconnects and starts over
from a fresh snapshot.

sipyco is asyncio-based but pytweezer's processes are not (the GUI has no
``qasync``), so both classes run their own event loop on a daemon thread.
Notifier mutations are only safe on that loop: from any other thread, make
them through :meth:`SyncServer.call`.
"""

import asyncio
import contextlib
import copy
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

from sipyco.pc_rpc import Server as RPCServer
from sipyco.sync_struct import Notifier, Publisher, Subscriber, process_mod

from pytweezer.logging_utils import get_logger

logger = get_logger("pytweezer.servers.sync")

RECONNECT_MIN_S = 0.2
RECONNECT_MAX_S = 5.0


class _LoopThread:
    """An asyncio event loop running forever on a daemon thread."""

    def __init__(self, name: str):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coroutine) -> Future:
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    def call(self, fn: Callable, *args) -> Future:
        async def invoke():
            return fn(*args)

        return self.submit(invoke())

    def stop(self, timeout: float = 5.0) -> None:
        if self.loop.is_closed():
            return
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout)
        if not self._thread.is_alive():
            self.loop.close()


class SyncServer:
    """Publishes notifiers, and optionally serves RPC targets, from one loop.

    RPC methods run on the loop thread, so they may mutate the notifiers
    directly. Port 0 binds a free port; :attr:`port` and :attr:`rpc_port` say
    which.

    Args:
        notifiers: Notifier name -> notifier, as :class:`SyncMirror` asks for it.
        host: Address to bind.
        port: Port for the notifiers.
        rpc_targets: Optional target name -> object for a ``pc_rpc`` server.
        rpc_port: Port for the RPC server; required with ``rpc_targets``.
    """

    def __init__(
        self,
        notifiers: dict[str, Notifier],
        host: str,
        port: int,
        rpc_targets: dict[str, Any] | None = None,
        rpc_port: int | None = None,
    ):
        self.notifiers = notifiers
        self._loop = _LoopThread(f"sync-server:{','.join(notifiers)}")
        self._publisher = Publisher(notifiers)
        self._rpc = RPCServer(rpc_targets) if rpc_targets else None
        try:
            self._loop.submit(self._start(host, port, rpc_port)).result(timeout=10)
        except BaseException:
            self._loop.stop()
            raise
        self._closed = False
        self.port = _bound_port(self._publisher)
        self.rpc_port = _bound_port(self._rpc) if self._rpc else None

    async def _start(self, host: str, port: int, rpc_port: int | None) -> None:
        await self._publisher.start(host, port)
        if self._rpc is not None:
            try:
                await self._rpc.start(host, rpc_port)
            except BaseException:
                await self._publisher.stop()
                raise

    def call(self, fn: Callable, *args) -> Future:
        """Run ``fn(*args)`` on the loop thread; the future holds its result."""
        return self._loop.call(fn, *args)

    def close(self) -> None:
        """Stop serving; calling it again does nothing."""
        if self._closed:
            return
        self._closed = True

        async def stop():
            await self._publisher.stop()
            if self._rpc is not None:
                await self._rpc.stop()

        try:
            self._loop.submit(stop()).result(timeout=5)
        finally:
            self._loop.stop()


def _bound_port(server) -> int:
    return server.server.sockets[0].getsockname()[1]


class SyncMirror:
    """A live, thread-safe copy of a notifier served by a :class:`SyncServer`.

    Reconnects for as long as it is open, backing off from 0.2 s to 5 s. Until
    the first snapshot arrives the data is ``None``.

    Args:
        host: Server address.
        port: Server notifier port.
        notifier_name: Which notifier to mirror.
        on_mod: Called as ``on_mod(mod, data)`` on the loop thread after each
            mod is applied (including the ``"init"`` that a (re)connect
            delivers), while the data lock is held.
        keep_mods: Queue every mod for :meth:`drain`, for consumers that poll.
    """

    def __init__(
        self,
        host: str,
        port: int,
        notifier_name: str,
        on_mod: Callable[[dict, Any], None] | None = None,
        keep_mods: bool = False,
    ):
        self.host = host
        self.port = port
        self.notifier_name = notifier_name
        self._on_mod = on_mod
        self._keep_mods = keep_mods
        self._lock = threading.RLock()
        self._data = None
        self._mods: list[dict] = []
        self._connected = False
        self._initialised = threading.Event()
        self._loop = _LoopThread(f"sync-mirror:{notifier_name}")
        self._task = self._loop.call(
            lambda: self._loop.loop.create_task(self._run())
        ).result()

    @property
    def connected(self) -> bool:
        """Whether the mirror is connected and holds the server's current state."""
        return self._connected

    def wait_initialised(self, timeout: float | None = None) -> bool:
        """Block until the first snapshot has arrived, or ``timeout`` passes."""
        return self._initialised.wait(timeout)

    def read(self, fn: Callable[[Any], Any]) -> Any:
        """``fn(data)`` under the data lock.

        ``fn`` may edit the data, but only locally: the next snapshot replaces
        it. It must not keep references to it.
        """
        with self._lock:
            return fn(self._data)

    def snapshot(self) -> Any:
        """A deep copy of the data."""
        with self._lock:
            return copy.deepcopy(self._data)

    def drain(self) -> list[dict]:
        """Mods received since the last call (only with ``keep_mods``)."""
        with self._lock:
            mods, self._mods = self._mods, []
        return mods

    def close(self) -> None:
        async def stop():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

        try:
            self._loop.submit(stop()).result(timeout=5)
        finally:
            self._loop.stop()
            self._connected = False

    async def _run(self) -> None:
        delay = RECONNECT_MIN_S
        while True:
            subscriber = Subscriber(
                self.notifier_name,
                target_builder=lambda struct: struct,
                notify_cb=self._apply,
            )
            try:
                await subscriber.connect(self.host, self.port)
            except OSError:
                await asyncio.sleep(delay)
                delay = min(2 * delay, RECONNECT_MAX_S)
                continue
            try:
                await subscriber.receive_task
            except asyncio.CancelledError:
                await subscriber.close()
                raise
            except Exception:
                logger.warning(
                    "Lost %s from %s:%s",
                    self.notifier_name,
                    self.host,
                    self.port,
                    exc_info=True,
                )
            finally:
                self._connected = False
            try:
                await subscriber.close()
            except Exception:
                pass
            delay = RECONNECT_MIN_S
            await asyncio.sleep(delay)

    def _apply(self, mod: dict) -> None:
        # The Subscriber has already applied ``mod`` to its own copy, and the
        # mod's values may be objects inside that copy: deep-copy so ours never
        # shares (and double-applies) anything with it.
        mod = copy.deepcopy(mod)
        with self._lock:
            if mod["action"] == "init":
                self._data = mod["struct"]
            else:
                process_mod(self._data, mod)
            if self._keep_mods:
                self._mods.append(mod)
            if self._on_mod is not None:
                try:
                    self._on_mod(mod, self._data)
                except Exception:
                    logger.exception("on_mod callback failed")
        if mod["action"] == "init":
            self._connected = True
            self._initialised.set()
