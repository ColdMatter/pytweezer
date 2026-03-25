"""
server.py
---------
Owns the authoritative copy of one or more DictSyncModels.
Exposes two ZMQ sockets:

  REP (default tcp://*:5560)
      Handles INIT / SET / DEL commands from clients.
      Responds with ok() or err().

  PUB (default tcp://*:5561)
      Broadcasts every committed mutation to all subscribers.
      Topic = model_name, so clients only receive models they care about.

Thread safety
-------------
  All model mutations happen on the Qt main thread via QMetaObject.invokeMethod,
  so Qt's internal state is never touched from a background thread.
  The ZMQ loop runs in a daemon thread.
"""

import threading
import logging
import zmq
from PyQt5.QtCore import QObject, pyqtSignal, QMetaObject, Qt, Q_ARG
from . import protocol as proto


class ModelServer(QObject):
    """
    Attach one ModelServer to your application.
    Register each DictSyncModel you want to expose, then call start().

    Example
    -------
        server = ModelServer()
        server.register('schedule', schedule_model, key_type=int)
        server.start()
    """

    # Emitted on the Qt thread whenever a remote client mutates a model.
    # Useful for logging or triggering side-effects.
    remote_change = pyqtSignal(str, str, object)   # model_name, op, key

    def __init__(self,
                 rep_endpoint: str = 'tcp://*:5560',
                 pub_endpoint: str = 'tcp://*:5561',
                 parent=None):
        super().__init__(parent)
        self._rep_ep = rep_endpoint
        self._pub_ep = pub_endpoint

        # model_name -> {'model': DictSyncModel, 'key_type': callable}
        self._models: dict[str, dict] = {}

        self._ctx = zmq.Context()
        self._rep = self._ctx.socket(zmq.REP)
        self._rep.setsockopt(zmq.LINGER, 0)
        self._rep.setsockopt(zmq.RCVTIMEO, 500)   # allows clean shutdown

        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.LINGER, 0)

        self._stop = threading.Event()
        self._lock = threading.Lock()              # guards _models dict

    # ── Registration ─────────────────────────────────────────────────────────

    def register(self, name: str, model, key_type: type = str):
        """
        Register a DictSyncModel under a string name.

        Args:
            name     : Unique identifier used as the PUB topic and in messages.
            model    : A DictSyncModel instance (or subclass).
            key_type : Callable to restore key type from string, e.g. int.
                       JSON keys are always strings; this converts them back.
        """
        with self._lock:
            self._models[name] = {'model': model, 'key_type': key_type}
        logging.info(f'[ModelServer] Registered model "{name}"')

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Bind sockets and start the background ZMQ thread."""
        self._rep.bind(self._rep_ep)
        self._pub.bind(self._pub_ep)
        logging.info(f'[ModelServer] REP bound to {self._rep_ep}')
        logging.info(f'[ModelServer] PUB bound to {self._pub_ep}')
        t = threading.Thread(target=self._loop, daemon=True, name='ModelServer-ZMQ')
        t.start()

    def stop(self):
        """Signal the ZMQ thread to exit and clean up."""
        self._stop.set()
        self._rep.close()
        self._pub.close()
        self._ctx.term()

    # ── ZMQ loop (background thread) ─────────────────────────────────────────

    def _loop(self):
        while not self._stop.is_set():
            try:
                raw = self._rep.recv()
            except zmq.Again:
                continue                           # timeout — check stop flag
            except zmq.ZMQError as e:
                if not self._stop.is_set():
                    logging.error(f'[ModelServer] ZMQ error: {e}')
                break

            try:
                msg = proto.decode(raw)
                response = self._dispatch(msg)
            except Exception as e:
                logging.exception('[ModelServer] Error handling message')
                response = proto.err(str(e))

            self._rep.send(proto.encode(response))

    def _dispatch(self, msg: dict) -> dict:
        """Route an incoming message to the correct handler."""
        op = msg.get('op')
        name = msg.get('model_name')

        with self._lock:
            entry = self._models.get(name)

        if entry is None:
            return proto.err(f'Unknown model: {name}')

        model = entry['model']
        key_type = entry['key_type']

        if op == 'INIT':
            return self._handle_init(model)

        elif op == 'SET':
            key = key_type(msg['key'])
            value = msg['value']
            # ── Qt thread safety ──────────────────────────────────────────
            # DictSyncModel.__setitem__ emits Qt signals (beginInsertRows etc.)
            # These MUST run on the Qt main thread. We use invokeMethod with
            # BlockingQueued connection to safely cross the thread boundary
            # and wait for completion before broadcasting.
            QMetaObject.invokeMethod(
                self,
                '_qt_set',
                Qt.BlockingQueuedConnection,
                Q_ARG(str, name),
                Q_ARG(object, key),
                Q_ARG(object, value),
            )
            self._broadcast(proto.make_set(name, key, value))
            self.remote_change.emit(name, 'SET', key)
            return proto.ok()

        elif op == 'DEL':
            key = key_type(msg['key'])
            QMetaObject.invokeMethod(
                self,
                '_qt_del',
                Qt.BlockingQueuedConnection,
                Q_ARG(str, name),
                Q_ARG(object, key),
            )
            self._broadcast(proto.make_del(name, key))
            self.remote_change.emit(name, 'DEL', key)
            return proto.ok()

        else:
            return proto.err(f'Unknown op: {op}')

    # ── Qt-thread slots (called via invokeMethod) ─────────────────────────────

    def _qt_set(self, name: str, key, value):
        with self._lock:
            model = self._models[name]['model']
        model[key] = value

    def _qt_del(self, name: str, key):
        with self._lock:
            model = self._models[name]['model']
        del model[key]

    def _handle_init(self, model) -> dict:
        """Serialise the full backing_store for a new client."""
        # Keys are stringified for JSON; client restores type via key_type.
        snapshot = {str(k): v for k, v in model.backing_store.items()}
        return proto.ok(snapshot)

    # ── Broadcast ─────────────────────────────────────────────────────────────

    def _broadcast(self, msg: dict):
        """
        Publish a change to all subscribed clients.
        Topic = model_name so clients can subscribe selectively.
        Uses two-frame multipart: [topic_bytes, payload_bytes]
        """
        topic = msg['model_name'].encode('utf-8')
        self._pub.send_multipart([topic, proto.encode(msg)])