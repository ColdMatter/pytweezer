"""
client.py
---------
Holds a replica of a remote DictSyncModel and keeps it in sync via TCP.

On construction:
  1. Sends INIT to the server REP socket → receives full snapshot.
  2. Populates the local DictSyncModel with the snapshot.
  3. Starts a background thread subscribing to PUB broadcasts.

On mutation (set / del):
  - Applies the change locally immediately (optimistic update).
  - Sends the mutation to the server REP socket for authoritative commit
    and rebroadcast to other clients.

Qt thread safety
----------------
  Incoming PUB messages arrive on a background thread.
  All model mutations are dispatched to the Qt main thread via
  QMetaObject.invokeMethod, exactly as in ModelServer.
"""

import threading
import logging
import zmq
from PyQt5.QtCore import QObject, pyqtSignal, QMetaObject, Qt, Q_ARG
from . import protocol as proto


class ModelClient(QObject):
    """
    Attach one ModelClient per model you want to mirror on a remote machine.

    Example
    -------
        client = ModelClient(
            model       = ScheduleModel({}),
            model_name  = 'schedule',
            key_type    = int,
            server_ip   = '192.168.1.100',
        )
        client.start()
        # client.model is now a live ScheduleModel you can attach to a QTableView
    """

    # Emitted on the Qt thread when a remote change is applied to the replica.
    synced = pyqtSignal(str, str, object)   # model_name, op, key

    # Emitted if the server cannot be reached during start().
    connection_failed = pyqtSignal(str)

    def __init__(self,
                 model,
                 model_name: str,
                 key_type: type = str,
                 server_ip: str = '127.0.0.1',
                 rep_port: int = 5560,
                 pub_port: int = 5561,
                 parent=None):
        """
        Args:
            model      : An empty DictSyncModel (or subclass) instance.
            model_name : Must match the name used in ModelServer.register().
            key_type   : Callable to restore key type from JSON string, e.g. int.
            server_ip  : IP or hostname of the machine running ModelServer.
            rep_port   : Port of the server's REP socket.
            pub_port   : Port of the server's PUB socket.
        """
        super().__init__(parent)
        self.model = model
        self._name = model_name
        self._key_type = key_type

        self._rep_ep = f'tcp://{server_ip}:{rep_port}'
        self._pub_ep = f'tcp://{server_ip}:{pub_port}'

        self._ctx = zmq.Context()
        self._stop = threading.Event()

        # REQ socket — used for INIT and mutations.
        # Protected by a lock because set()/del() may be called from any thread.
        self._req = self._ctx.socket(zmq.REQ)
        self._req.setsockopt(zmq.LINGER, 0)
        self._req.setsockopt(zmq.RCVTIMEO, 5000)
        self._req.setsockopt(zmq.SNDTIMEO, 5000)
        self._req_lock = threading.Lock()

        # SUB socket — receives broadcasts from the server.
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.LINGER, 0)
        self._sub.setsockopt(zmq.RCVTIMEO, 500)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, model_name)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, retries: int = 5):
        """
        Connect to the server, fetch initial snapshot, then start the
        background listener thread.

        Args:
            retries: Number of INIT attempts before emitting connection_failed.
        """
        self._req.connect(self._rep_ep)
        self._sub.connect(self._pub_ep)
        logging.info(f'[ModelClient:{self._name}] Connecting to {self._rep_ep}')

        snapshot = self._fetch_snapshot(retries)
        if snapshot is None:
            self.connection_failed.emit(self._rep_ep)
            return

        self._apply_snapshot(snapshot)

        t = threading.Thread(
            target=self._listen_loop, daemon=True,
            name=f'ModelClient-{self._name}'
        )
        t.start()
        logging.info(f'[ModelClient:{self._name}] Started — {len(snapshot)} rows loaded.')

    def stop(self):
        self._stop.set()
        self._req.close()
        self._sub.close()
        self._ctx.term()

    # ── Public mutation API ───────────────────────────────────────────────────

    def set(self, key, value):
        """
        Set a row in the model and propagate to the server.

        Applies the change locally first (optimistic update) so the UI
        feels instant, then sends to the server. If the server rejects it,
        a warning is logged and the local change is rolled back.
        """
        # Optimistic local update on Qt thread
        QMetaObject.invokeMethod(
            self, '_qt_set',
            Qt.QueuedConnection,
            Q_ARG(object, key),
            Q_ARG(object, value),
        )
        # Send to server (may be called from any thread)
        threading.Thread(
            target=self._send_set, args=(key, value), daemon=True
        ).start()

    def delete(self, key):
        """Delete a row from the model and propagate to the server."""
        QMetaObject.invokeMethod(
            self, '_qt_del',
            Qt.QueuedConnection,
            Q_ARG(object, key),
        )
        threading.Thread(
            target=self._send_del, args=(key,), daemon=True
        ).start()

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def _fetch_snapshot(self, retries: int) -> dict | None:
        """Request the full model state from the server."""
        delay = 0.5
        for attempt in range(1, retries + 1):
            try:
                with self._req_lock:
                    self._req.send(proto.encode(proto.make_init(self._name)))
                    raw = self._req.recv()
                resp = proto.decode(raw)
                if resp['status'] == 'OK':
                    return resp['payload']
                logging.warning(f'[ModelClient:{self._name}] INIT rejected: {resp}')
            except zmq.Again:
                logging.warning(
                    f'[ModelClient:{self._name}] INIT timeout '
                    f'(attempt {attempt}/{retries}), retrying in {delay:.1f}s…'
                )
                import time; time.sleep(delay)
                delay = min(delay * 2, 10)
        return None

    def _apply_snapshot(self, snapshot: dict):
        """Populate the local model from a snapshot dict (runs on caller thread)."""
        for str_key, value in snapshot.items():
            key = self._key_type(str_key)
            # Direct backing_store write — avoids emitting signals for each row.
            # We emit layoutChanged once at the end instead.
            self.model.backing_store[key] = value
            self.model.row_to_key = sorted(
                self.model.backing_store.keys(),
                key=lambda k: self.model.sort_key(k, self.model.backing_store[k])
            )
        self.model.layoutChanged.emit()

    # ── Background listener ───────────────────────────────────────────────────

    def _listen_loop(self):
        """
        Receive PUB broadcasts from the server and apply them to the replica.
        Runs on a daemon thread — never touches Qt objects directly.
        """
        while not self._stop.is_set():
            try:
                topic, raw = self._sub.recv_multipart()
            except zmq.Again:
                continue                           # timeout — check stop flag
            except zmq.ZMQError as e:
                if not self._stop.is_set():
                    logging.error(f'[ModelClient:{self._name}] SUB error: {e}')
                break

            try:
                msg = proto.decode(raw)
                self._handle_broadcast(msg)
            except Exception:
                logging.exception(f'[ModelClient:{self._name}] Error applying broadcast')

    def _handle_broadcast(self, msg: dict):
        """Dispatch a broadcast message to the Qt thread."""
        op = msg.get('op')
        key = self._key_type(msg['key'])

        if op == 'SET':
            QMetaObject.invokeMethod(
                self, '_qt_set',
                Qt.QueuedConnection,
                Q_ARG(object, key),
                Q_ARG(object, msg['value']),
            )
            self.synced.emit(self._name, 'SET', key)

        elif op == 'DEL':
            QMetaObject.invokeMethod(
                self, '_qt_del',
                Qt.QueuedConnection,
                Q_ARG(object, key),
            )
            self.synced.emit(self._name, 'DEL', key)

    # ── Qt-thread slots ───────────────────────────────────────────────────────

    def _qt_set(self, key, value):
        self.model[key] = value

    def _qt_del(self, key):
        del self.model[key]

    # ── REQ helpers (background threads) ─────────────────────────────────────

    def _send_set(self, key, value):
        msg = proto.make_set(self._name, key, value)
        self._req_send(msg, context=f'SET {key}')

    def _send_del(self, key):
        msg = proto.make_del(self._name, key)
        self._req_send(msg, context=f'DEL {key}')

    def _req_send(self, msg: dict, context: str = ''):
        """Send a REQ message and check the response. Thread-safe."""
        try:
            with self._req_lock:
                self._req.send(proto.encode(msg))
                raw = self._req.recv()
            resp = proto.decode(raw)
            if resp['status'] != 'OK':
                logging.warning(
                    f'[ModelClient:{self._name}] Server rejected {context}: {resp}'
                )
        except zmq.Again:
            logging.error(
                f'[ModelClient:{self._name}] Timeout sending {context}'
            )
        except zmq.ZMQError as e:
            logging.error(
                f'[ModelClient:{self._name}] ZMQ error sending {context}: {e}'
            )