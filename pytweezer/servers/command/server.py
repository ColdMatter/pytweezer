"""
server.py
---------
Listens for commands from any number of remote clients and dispatches
them to registered handler callables on the Qt main thread.

Uses ZMQ ROUTER so multiple DEALER clients can connect simultaneously
without blocking each other. Each message is handled independently —
there is no per-client state.

Handler registration
--------------------
    server = CommandServer()
    server.register('pause',        exp_manager.pause)
    server.register('restart',      exp_manager.start_queue)
    server.register('terminate_all',exp_manager.terminate_all)
    server.start()

Handlers are called on the Qt main thread via QMetaObject.invokeMethod.
They receive the 'args' dict from the message as a keyword argument,
so handlers can optionally accept **kwargs if they need parameters:

    def my_handler(**kwargs):
        priority = kwargs.get('priority', 0)...
"""

import threading
import logging
import zmq
from PyQt5.QtCore import QObject, QMetaObject, Qt, Q_ARG, pyqtSlot
from . import protocol as proto


class CommandServer(QObject):

    def __init__(self,
                 endpoint: str = 'tcp://*:5562',
                 parent=None):
        """
        Args:
            endpoint: ZMQ ROUTER bind address.
                      Use 'tcp://*:5562' to accept from any interface,
                      or 'tcp://192.168.1.100:5562' to restrict to one NIC.
        """
        super().__init__(parent)
        self._endpoint = endpoint
        self._handlers: dict[str, callable] = {}

        self._ctx = zmq.Context()
        self._router = self._ctx.socket(zmq.ROUTER)
        self._router.setsockopt(zmq.LINGER, 0)
        self._router.setsockopt(zmq.RCVTIMEO, 500)   # allows clean shutdown

        self._stop = threading.Event()

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, cmd: str, handler: callable):
        """
        Register a callable to handle a named command.

        Args:
            cmd     : Command name string, e.g. 'pause'.
            handler : Callable invoked on the Qt main thread.
                      Receives args dict as **kwargs.
                      Return value is sent back to the client as payload.
        """
        self._handlers[cmd] = handler
        logging.info(f'[CommandServer] Registered handler for "{cmd}"')

    def unregister(self, cmd: str):
        self._handlers.pop(cmd, None)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        self._router.bind(self._endpoint)
        logging.info(f'[CommandServer] Listening on {self._endpoint}')
        t = threading.Thread(
            target=self._loop, daemon=True, name='CommandServer-ZMQ'
        )
        t.start()

    def stop(self):
        self._stop.set()
        self._router.close()
        self._ctx.term()

    # ── ZMQ loop (background thread) ──────────────────────────────────────────

    def _loop(self):
        """
        ROUTER socket receives multipart frames:
          [identity, empty_delimiter, payload]

        identity         — ZMQ-assigned client ID, needed to route the reply.
        empty_delimiter  — Required by ROUTER/DEALER framing convention.
        payload          — Our JSON command bytes.
        """
        while not self._stop.is_set():
            try:
                frames = self._router.recv_multipart()
            except zmq.Again:
                continue
            except zmq.ZMQError as e:
                if not self._stop.is_set():
                    logging.error(f'[CommandServer] ZMQ error: {e}')
                break

            if len(frames) != 3:
                logging.warning(f'[CommandServer] Unexpected frame count: {len(frames)}')
                continue

            identity, _delimiter, raw = frames

            try:
                msg = proto.decode(raw)
            except Exception as e:
                logging.error(f'[CommandServer] Failed to decode message: {e}')
                continue

            # Dispatch to Qt thread; pass identity so we can reply after
            QMetaObject.invokeMethod(
                self,
                '_qt_dispatch',
                Qt.QueuedConnection,          # non-blocking — ZMQ loop continues
                Q_ARG(object, identity),
                Q_ARG(object, msg),
            )

    # ── Qt-thread dispatch ────────────────────────────────────────────────────

    @pyqtSlot(object, object)
    def _qt_dispatch(self, identity: bytes, msg: dict):
        """
        Runs on the Qt main thread.
        Looks up the handler, calls it, then sends the reply back via
        a separate sender thread to avoid blocking the Qt event loop
        on the ZMQ send.
        """
        cmd    = msg.get('cmd', '')
        args   = msg.get('args', {})
        msg_id = msg.get('msg_id', '')
        handler = self._handlers.get(cmd)

        if handler is None:
            reply = proto.err(msg_id, f'Unknown command: "{cmd}"')
            logging.warning(f'[CommandServer] Unknown command: "{cmd}"')
        else:
            try:
                result = handler(**args)
                reply = proto.ok(msg_id, payload=result)
                logging.debug(f'[CommandServer] Handled "{cmd}" OK')
            except Exception as e:
                reply = proto.err(msg_id, str(e))
                logging.exception(f'[CommandServer] Handler for "{cmd}" raised')

        # Send reply on a thread — never block the Qt event loop on I/O
        threading.Thread(
            target=self._send_reply,
            args=(identity, reply),
            daemon=True,
        ).start()

    def _send_reply(self, identity: bytes, reply: dict):
        """Send a ROUTER reply. Must mirror the incoming frame structure."""
        try:
            self._router.send_multipart(
                [identity, b'', proto.encode(reply)]
            )
        except zmq.ZMQError as e:
            logging.error(f'[CommandServer] Failed to send reply: {e}')