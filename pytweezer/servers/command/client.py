"""
client.py
---------
Sends commands to a remote CommandServer.

Uses ZMQ DEALER (async-capable counterpart to ROUTER).
Each CommandClient instance maintains one persistent connection.

Usage
-----
    cmd_client = CommandClient(server_ip='192.168.1.100')
    cmd_client.start()

    # Fire-and-forget
    cmd_client.send('pause')

    # With arguments
    cmd_client.send('set_priority', args={'task_nr': 5, 'priority': 10})

    # With a reply callback
    cmd_client.send('restart', on_reply=lambda r: print('Server replied:', r))
"""

import threading
import logging
import zmq
from . import protocol as proto


class CommandClient:

    def __init__(self,
                 server_ip: str = '127.0.0.1',
                 port: int = 5562):
        """
        Args:
            server_ip : IP or hostname of the CommandServer machine.
            port      : Must match CommandServer's bind port.
        """
        self._endpoint = f'tcp://{server_ip}:{port}'

        self._ctx = zmq.Context()
        self._dealer = self._ctx.socket(zmq.DEALER)
        self._dealer.setsockopt(zmq.LINGER, 0)
        self._dealer.setsockopt(zmq.RCVTIMEO, 5000)
        self._dealer.setsockopt(zmq.SNDTIMEO, 2000)

        # Protects the DEALER socket — not thread-safe in ZMQ
        self._lock = threading.Lock()

        # msg_id -> callback; populated by send(), consumed by _recv_loop()
        self._pending: dict[str, callable] = {}
        self._pending_lock = threading.Lock()

        self._stop = threading.Event()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        self._dealer.connect(self._endpoint)
        logging.info(f'[CommandClient] Connected to {self._endpoint}')
        t = threading.Thread(
            target=self._recv_loop, daemon=True, name='CommandClient-recv'
        )
        t.start()

    def stop(self):
        self._stop.set()
        self._dealer.close()
        self._ctx.term()

    # ── Public API ────────────────────────────────────────────────────────────

    def send(self,
             cmd: str,
             args: dict = None,
             on_reply: callable = None,
             timeout: float = 5.0) -> str:
        """
        Send a command to the server.

        Args:
            cmd      : Command name, e.g. 'pause'.
            args     : Optional dict of arguments for the handler.
            on_reply : Optional callback(reply_dict) called when the server
                       responds. Called on the recv thread, not the Qt thread —
                       use Qt signals if you need to update the UI.
            timeout  : Seconds to wait for a reply before logging a warning.
                       Only meaningful if on_reply is provided.

        Returns:
            msg_id (str): The UUID of the sent message.
        """
        msg = proto.make_command(cmd, args)

        if on_reply is not None:
            with self._pending_lock:
                self._pending[msg['msg_id']] = (on_reply, timeout)

        threading.Thread(
            target=self._send_msg,
            args=(msg,),
            daemon=True,
        ).start()

        return msg['msg_id']

    # ── Internal send (background thread) ────────────────────────────────────

    def _send_msg(self, msg: dict):
        """
        DEALER framing: [empty_delimiter, payload]
        The empty delimiter mirrors what ROUTER expects on the other side.
        """
        try:
            with self._lock:
                self._dealer.send_multipart(
                    [b'', proto.encode(msg)]
                )
            logging.debug(f'[CommandClient] Sent: {msg["cmd"]} ({msg["msg_id"]})')
        except zmq.Again:
            logging.error(
                f'[CommandClient] Timeout sending "{msg["cmd"]}"'
            )
            # Remove pending callback so we don't leak it
            with self._pending_lock:
                self._pending.pop(msg['msg_id'], None)
        except zmq.ZMQError as e:
            logging.error(f'[CommandClient] ZMQ error: {e}')

    # ── Recv loop (background thread) ─────────────────────────────────────────

    def _recv_loop(self):
        """
        Receive replies from the server and dispatch to registered callbacks.
        DEALER recv frames: [empty_delimiter, payload]
        """
        while not self._stop.is_set():
            try:
                with self._lock:
                    frames = self._dealer.recv_multipart()
            except zmq.Again:
                continue
            except zmq.ZMQError as e:
                if not self._stop.is_set():
                    logging.error(f'[CommandClient] Recv error: {e}')
                break

            if len(frames) != 2:
                logging.warning(f'[CommandClient] Unexpected frame count: {len(frames)}')
                continue

            _delimiter, raw = frames

            try:
                reply = proto.decode(raw)
            except Exception as e:
                logging.error(f'[CommandClient] Failed to decode reply: {e}')
                continue

            msg_id = reply.get('msg_id', '')
            with self._pending_lock:
                entry = self._pending.pop(msg_id, None)

            if entry is not None:
                callback, _ = entry
                try:
                    callback(reply)
                except Exception:
                    logging.exception('[CommandClient] Reply callback raised')
            else:
                logging.debug(f'[CommandClient] Received unsolicited reply: {msg_id}')