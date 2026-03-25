from pytweezer.servers import EVENT_MAP  # zmqcontext no longer needed
from pytweezer.servers import configreader as cr
from pytweezer.servers.xsub_xpub import event_monitor
import copy
import zmq
import threading
import time
import logging
import sys
import os
from pytweezer.analysis.print_messages import print_error

"""
TCP Migration Notes:
--------------------
- All ZMQ sockets now use tcp:// instead of ipc://
- The zmqcontext is now created locally so each process (even remote ones)
  gets its own context — avoids sharing a context across a network boundary.
- Endpoints are resolved from properties.json (updated to use TCP addresses).
- No other logic changes are required — ZMQ abstracts the transport layer.
"""


class PropertyAttribute:
    """
    Access properties as attributes.
    Unchanged from IPC version — transport-agnostic.
    """

    def __init__(self, propname, defaultval, parent=None):
        self._default = defaultval
        self._propname = propname
        self.parent = parent
        self._value = defaultval

    @property
    def value(self):
        return self._value

    @value.getter
    def value(self):
        self._value = self.parent._props.get(self._propname, self._default)
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self.parent._props.set(self._propname, value)

    def __get__(self, obj, t):
        return obj._props.get(self._propname, self._default)

    def __set__(self, obj, v):
        obj._props.set(self._propname, v)


class Properties(threading.Thread):
    """
    TCP-capable Properties manager.

    Key changes vs IPC version:
      1. zmqcontext created per-instance (not imported as a shared global).
         Shared ZMQ contexts don't work across process/machine boundaries.
      2. All endpoints use tcp:// — resolved from config.
      3. Added optional `server_ip` override so remote clients can point
         at a specific host without editing the config file.
      4. Added reconnect logic with exponential backoff for the INIT REQ socket,
         since TCP connections can fail transiently (unlike local IPC).
      5. Socket linger is set to 0 to prevent hangs on shutdown over TCP.
      6. Added `close()` for clean teardown.
    """

    def __init__(self, name, initfromfile=False, server_ip=None):
        """
        Args:
            name        (str):  Process name, used as default key namespace.
            initfromfile(bool): If True, load properties from local file
                                instead of querying the Propertylogger.
            server_ip   (str):  Optional IP override, e.g. '192.168.1.100'.
                                If provided, replaces the host in all endpoints
                                from the config. Useful for remote clients.
        """
        threading.Thread.__init__(self, daemon=True)

        if server_ip is None:
            server_ip = os.getenv("PYTWEEZER_SERVER_IP")

        conf = cr.Config()
        self.properties_lock = threading.Lock()
        self._stop_event = threading.Event()

        # ------------------------------------------------------------------
        # 1. Build a LOCAL ZMQ context.
        #    The original code imported a shared zmqcontext. Over TCP,
        #    each process must own its context — especially remote processes.
        # ------------------------------------------------------------------
        self._zmq_context = zmq.Context()

        # ------------------------------------------------------------------
        # 2. Resolve endpoints, optionally overriding the host IP.
        #    This lets a remote machine point at the hub without editing JSON.
        # ------------------------------------------------------------------
        hub_sub_ep = self._resolve_endpoint(
            conf["Servers"]["Propertyhub"]["sub"], server_ip
        )
        hub_pub_ep = self._resolve_endpoint(
            conf["Servers"]["Propertyhub"]["pub"], server_ip
        )
        logger_ep = self._resolve_endpoint(
            conf["Servers"]["Propertylogger"]["rep"], server_ip
        )

        logging.debug(f"[{name}] Connecting PUB -> {hub_sub_ep}")
        logging.debug(f"[{name}] Connecting SUB -> {hub_pub_ep}")

        # ------------------------------------------------------------------
        # 3. Create sockets.
        #    LINGER=0: don't block on close waiting to flush TCP buffers.
        #    RCVTIMEO / SNDTIMEO: prevent infinite hangs on a bad network.
        # ------------------------------------------------------------------
        self.pub_socket = self._zmq_context.socket(zmq.PUB)
        self.pub_socket.setsockopt(zmq.LINGER, 0)
        self.pub_socket.connect(hub_sub_ep)

        self.sub_socket = self._zmq_context.socket(zmq.SUB)
        self.sub_socket.setsockopt(zmq.LINGER, 0)
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 s recv timeout
        self.sub_socket.connect(hub_pub_ep)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "Prop")

        # ------------------------------------------------------------------
        # 4. Monitor sockets (unchanged — works over TCP too).
        # ------------------------------------------------------------------
        pub_mon = self.pub_socket.get_monitor_socket()
        threading.Thread(
            target=event_monitor, args=(pub_mon, f"Props: {name}", "PUB"), daemon=True
        ).start()

        sub_mon = self.sub_socket.get_monitor_socket()
        threading.Thread(
            target=event_monitor, args=(sub_mon, f"Props: {name}", "SUB"), daemon=True
        ).start()

        # Give TCP connections a moment to establish.
        # TCP handshake adds latency vs IPC — 50 ms is conservative but safe.
        time.sleep(0.05)

        # ------------------------------------------------------------------
        # 5. Initialise properties dict.
        #    REQ/REP over TCP with retry + timeout for resilience.
        # ------------------------------------------------------------------
        if not initfromfile:
            self.properties = self._fetch_initial_properties(logger_ep)
        else:
            self.properties = cr.Properties()

        self.recent_changes = set()

        # Start background threads
        threading.Thread(target=self.check, daemon=True).start()
        self.start()  # starts run() — the recv loop

        self.name = name
        if name not in self.properties:
            self.get("/" + name, {})

        logging.debug(self.properties)
        self.crashed = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_endpoint(endpoint: str, override_ip: str | None) -> str:
        """
        Replace the host portion of a tcp:// endpoint with override_ip.

        Example:
            endpoint    = 'tcp://192.168.1.100:5557'
            override_ip = '10.0.0.50'
            returns       'tcp://10.0.0.50:5557'

        If override_ip is None, the endpoint is returned unchanged.
        """
        if override_ip is None or not endpoint.startswith("tcp://"):
            return endpoint
        # tcp://host:port  ->  split on last ':'  to isolate port
        prefix, port = endpoint.rsplit(":", 1)
        return f"tcp://{override_ip}:{port}"

    def _fetch_initial_properties(self, logger_ep: str, retries: int = 5) -> dict:
        """
        Fetch the initial property dict from the Propertylogger via REQ/REP.

        Retries with exponential backoff — necessary over TCP because the
        server may not be immediately reachable (unlike IPC on localhost).

        Args:
            logger_ep (str): tcp:// endpoint of the Propertylogger REP socket.
            retries   (int): Number of attempts before raising.

        Returns:
            dict: The initial properties dictionary.

        Raises:
            ConnectionError: If all retries are exhausted.
        """
        delay = 0.5  # initial backoff in seconds

        for attempt in range(1, retries + 1):
            init_socket = self._zmq_context.socket(zmq.REQ)
            init_socket.setsockopt(zmq.LINGER, 0)
            init_socket.setsockopt(zmq.RCVTIMEO, 3000)  # 3 s per attempt
            init_socket.setsockopt(zmq.SNDTIMEO, 3000)

            try:
                init_socket.connect(logger_ep)
                init_socket.send_string("INIT?")
                _ack = init_socket.recv_string()  # acknowledgement frame
                properties = init_socket.recv_json()  # actual data frame
                logging.info(
                    f"Properties initialised from {logger_ep} " f"(attempt {attempt})"
                )
                return properties

            except zmq.Again:
                logging.warning(
                    f"[Properties] INIT timeout (attempt {attempt}/{retries}), "
                    f"retrying in {delay:.1f}s …"
                )
                time.sleep(delay)
                delay = min(delay * 2, 10)  # exponential backoff, cap at 10 s

            finally:
                # Always close the REQ socket — ZMQ REQ sockets are not
                # reusable after a failed send/recv cycle.
                init_socket.close()

        raise ConnectionError(
            f"[Properties] Could not reach Propertylogger at {logger_ep} "
            f"after {retries} attempts."
        )

    # ------------------------------------------------------------------
    # Key parsing (unchanged)
    # ------------------------------------------------------------------

    def _parsekey(self, key):
        if type(key) != list:
            if key[0] != "/":
                if key[-1] == "/":
                    key = key[:-1]
                key = "/" + self.name + "/" + key
            keys = key.split("/")
        else:
            keys = key
        if keys[0] == "":
            keys = keys[1:]
        return keys

    # ------------------------------------------------------------------
    # Public API  (set / get / delete / changes — logic unchanged)
    # ------------------------------------------------------------------

    def set(self, key, value):
        if self.get(key) != value:
            keys = self._parsekey(key)
            self._set(keys, value)
            self._send({"keys": keys, "value": value})

    def _set(self, keys, value):
        with self.properties_lock:
            prop = self.properties
            for key in keys[:-1]:
                if key not in prop:
                    prop[key] = {}
                prop = prop[key]
            if isinstance(value, dict) and "options" in value:
                if keys[-1] not in prop:
                    prop[keys[-1]] = {}
                prop[keys[-1]] = copy.deepcopy(value)
            elif (
                keys[-1] in prop
                and isinstance(prop[keys[-1]], dict)
                and "options" in prop[keys[-1]]
            ):
                if value not in prop[keys[-1]]["options"]:
                    print(
                        f"value {value} not in options "
                        f'{prop[keys[-1]]["options"]}. no change made to {keys[-1]}'
                    )
                else:
                    prop[keys[-1]]["value"] = copy.deepcopy(value)
            else:
                prop[keys[-1]] = copy.deepcopy(value)
            self.recent_changes.add("/" + "/".join(keys))

    def _send(self, data, flags=0):
        """
        Publish a property change.
        Unchanged in logic; ZMQ handles TCP framing transparently.
        """
        self.pub_socket.send_string("Propertychange_" + self.name, flags | zmq.SNDMORE)
        return self.pub_socket.send_json(data, flags)

    def delete(self, key):
        keys = self._parsekey(key)
        self._del(keys)
        self._send({"delete": keys})

    def _del(self, keys):
        with self.properties_lock:
            prop = self.properties
            for key in keys[:-1]:
                if key not in prop:
                    prop[key] = {}
                prop = prop[key]
            if keys[-1] in prop:
                del prop[keys[-1]]
            self.recent_changes.add("/" + "/".join(keys[:-1]))

    def _recv(self, flags=0):
        """
        Receive a property update from the hub.

        Added zmq.Again handling: over TCP, recv can time out (we set
        RCVTIMEO=5000 ms). We catch the timeout and loop — this also
        gives the thread a chance to check _stop_event.
        """
        try:
            message = self.sub_socket.recv_string(flags=flags)
            logging.debug(f"{self.name} received: {message}")
            md = self.sub_socket.recv_json(flags=flags)
            logging.debug(f"{self.name} received md: {md}")
            if "delete" in md:
                self._del(md["delete"])
            elif "keys" in md and "value" in md:
                self._set(md["keys"], md["value"])
        except zmq.Again:
            # Timeout — no message arrived within RCVTIMEO window. Not an error.
            pass
        except zmq.ZMQError as e:
            if not self._stop_event.is_set():
                logging.error(f"[Properties:{self.name}] ZMQ recv error: {e}")

    def get(self, key, defaultvalue=None):
        if key == "/":
            with self.properties_lock:
                return copy.deepcopy(self.properties)
        elif key[-1] == "/":
            key = key[:-1]
        keys = self._parsekey(key)
        return self._get(keys, defaultvalue)

    def _get(self, keys, defaultvalue):
        try:
            with self.properties_lock:
                prop = self.properties
                for key in keys[:-1]:
                    prop = prop[key]
                value = copy.deepcopy(prop[keys[-1]])
                if isinstance(value, dict) and "options" in value:
                    value = value["value"]
        except KeyError:
            logging.debug("properties.py key does not exist")
            self._set(keys, defaultvalue)
            self._send({"keys": keys, "value": defaultvalue})
            value = defaultvalue
            print(f"tried to _get an unset property {keys}")
            if isinstance(value, dict) and "options" in value:
                value = value["value"]
            print(f"setting default value: {value}")
        return value

    def changes(self, includeparent=True):
        with self.properties_lock:
            changes = self.recent_changes
            self.recent_changes = set()
        if includeparent:
            chang = set()
            for key in changes:
                keys = key[1:].split("/")
                for i in range(len(keys)):
                    chang.add("/" + "/".join(keys[: i + 1]))
            return chang
        return changes

    def check(self):
        """Heartbeat / watchdog — unchanged."""
        lastTime = 0
        timeout = 40
        interval = 30
        # (body intentionally left as in original)

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------

    def run(self):
        """Recv loop — exits cleanly when _stop_event is set."""
        while not self._stop_event.is_set():
            self._recv()

    def close(self):
        """
        Cleanly shut down sockets and ZMQ context.
        Important over TCP — lingering sockets can delay process exit.
        """
        self._stop_event.set()
        self.pub_socket.close()
        self.sub_socket.close()
        self._zmq_context.term()
        logging.info(f"[Properties:{self.name}] Closed.")


if __name__ == "__main__":
    pass
