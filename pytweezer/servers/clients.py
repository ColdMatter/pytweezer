"""
.. module:: clients
    :synopsis:  pytweezers standard interface to handle interprocess communication


Handels interaction with the datachannel in a convenient way

Currently there are 4 clients:

* **CommandClient** (used for controlling drivers and processes)
* **DataClient**    (used for small data like 1d curves)
* **ImageClient**   (used for larger chunks of 2d data, mainly images)
* **MessageClient** (used for error,warning,info and debug messages)


Examples
--------

Receiving image data from a stream::

    from pytweezer.servers import ImageClient
    self.imageq=ImageClient(name)       #the name only has an effect if sending data
    self.imageq.subscribe('teststream')

    msg=self.imageq.recv()
        if msg!= None:
            msgstr,dict_info,array_data=msg

Sending data over a stream::

    from pytweezer.servers import DataClient
    self.dataq=DataClient('teststream')
    self.dataq.send(info_dictionary,data_array,'substream1_')

Classes
=======

"""

import ipaddress
import json
import os
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import numpy
import numpy as np
import zmq

from pytweezer.configuration import config
from pytweezer.configuration.config import get_config
from pytweezer.logging_utils import get_logger
from pytweezer.servers import zmqcontext

logger = get_logger(__name__)

#: Put in front of every stream name a simulated client sends or subscribes to.
SIMULATED_STREAM_PREFIX = "sim."
#: ``"1"`` makes this process, and every process it starts, a simulation; the
#: test suite sets it.
SIMULATION_ENV = "PYTWEEZER_SIMULATION"

_simulations = 0
_simulations_lock = threading.Lock()


@contextmanager
def simulation() -> Iterator[None]:
    """Make every stream client created inside a simulated one (see :class:`GenericClient`)."""
    global _simulations
    with _simulations_lock:
        _simulations += 1
    try:
        yield
    finally:
        with _simulations_lock:
            _simulations -= 1


def in_simulation() -> bool:
    """Whether stream clients created now are simulated."""
    return (
        config.SIMULATING or os.environ.get(SIMULATION_ENV) == "1" or _simulations > 0
    )


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(socket.gethostbyname(host)).is_loopback
    except (OSError, ValueError):
        return False


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.floating):
            return float(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return super().default(obj)


class GenericClient:
    """Sends and receives on one hub, named by ``_hub`` in ``CONFIG["Servers"]``.

    A simulated client can never reach the lab's streams: it connects only to a
    hub on this PC's loopback, and every stream name it sends or subscribes to
    carries :data:`SIMULATED_STREAM_PREFIX`, which :meth:`recv` strips again.
    A client is simulated if made with ``simulated=True`` (as a simulated
    camera's are), or while :func:`in_simulation`: in a simulation session,
    in a process with ``PYTWEEZER_SIMULATION=1``, or inside :func:`simulation`.
    """

    _hub = None

    def __init__(
        self, name="noname", recvtimeout=1000, subscribe=None, simulated=False
    ):
        """send or receive data over balis ZMQ ports
        standardizes datatransfer and connects to sockets according to the central configuration
        Use this class for small (<10k) datasets this data will be logged stored etc,
        there is the Imageclient for larger (imagesize) data.

        Args:
            name (string):
                used when sending data. The data will be published under a channel always starting with the  name
            recvtimeout(int):
                time in ms the receive should wait for a timeout
            simulated (bool):
                make this client simulated whether or not the process is
        """
        self.name = name
        self.simulated = simulated or in_simulation()
        self.subscriptions = []
        # initialize sockets
        self.pub_socket = zmqcontext.socket(zmq.PUB)
        self.sub_socket = zmqcontext.socket(zmq.SUB)
        self.sub_socket.RCVTIMEO = recvtimeout
        self.pub_socket.setsockopt(zmq.SNDBUF, 1024 * 1024 * 100)
        self.pub_socket.setsockopt(zmq.SNDHWM, 0)
        self.sub_socket.setsockopt(zmq.RCVBUF, 1024 * 1024 * 100)
        self.sub_socket.setsockopt(zmq.RCVHWM, 0)
        self._connect()
        if subscribe is not None:
            self.subscribe(subscribe)

    def _connect(self):
        conf = get_config()
        c = conf["Servers"][self._hub]
        host = c["host"]
        if self.simulated and not _is_loopback(host):
            # Unconnected, sends are dropped and nothing arrives.
            logger.info(
                "Simulated %s client %r left unconnected: %s is not on this PC",
                self._hub,
                self.name,
                host,
            )
            return
        # In the xsub/xpub hub, publishers must connect to XSUB (sub_port)
        # and subscribers must connect to XPUB (pub_port).
        self.pub_socket.connect(f"tcp://{host}:{c['sub_port']}")
        self.sub_socket.connect(f"tcp://{host}:{c['pub_port']}")

    def _topic(self, stream):
        """``stream`` as it goes over the wire."""
        return SIMULATED_STREAM_PREFIX + stream if self.simulated else stream

    def _stream(self, topic):
        """The stream name of a received ``topic``."""
        if self.simulated:
            return topic.removeprefix(SIMULATED_STREAM_PREFIX)
        return topic

    def unsubscribe(self, channel=None):
        """unsubscribe from specific channel

        Args:
            channel (string) :
        """
        if channel is None:
            for entry in self.subscriptions:
                self.sub_socket.setsockopt_string(zmq.UNSUBSCRIBE, self._topic(entry))
        else:
            self.sub_socket.setsockopt_string(zmq.UNSUBSCRIBE, self._topic(channel))
        # self.sub_socket.setsockopt_string(zmq.UNSUBSCRIBE, bytes(channel,'utf8'))

    def subscribe(self, channel):
        """subscribe to a channel

        Args:
            channel : string
                channel to subscribe to
            channel ([string]):
                you can alternatively give a list of channels
        """
        if type(channel) == str:
            self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, self._topic(channel))
            self.subscriptions.append(channel)
        elif type(channel) == list:
            for ch in channel:
                self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, self._topic(ch))
                self.subscriptions.append(ch)

    def send(
        self, datadict, A=None, channel="", flags=0, copy=True, track=False, prefix=None
    ):
        """distribute ove ZMQ

        Args:
            datadict (dict) :
                data to be send (must be json serializable)
            A (np.array)    :
                optional array of data
            channel (string):
                subchannel (will be appended to name when sending)
            flags (int) :
                see ZMQ send flags in the ZMQ doku
        """
        if prefix is None:
            topic = self._topic(self.name + channel)
        else:
            topic = self._topic(prefix + "." + channel)
        self.pub_socket.send_string(topic, flags | zmq.SNDMORE)

        if not isinstance(A, np.ndarray):
            return self.pub_socket.send_json(datadict, flags, cls=NpEncoder)
        else:
            # recv() rebuilds the array in C order from the raw buffer.
            A = np.ascontiguousarray(A)
            datadict["dtype"] = str(A.dtype)
            datadict["shape"] = A.shape
            self.pub_socket.send_json(datadict, flags | zmq.SNDMORE, cls=NpEncoder)
            return self.pub_socket.send(A, flags, copy=copy, track=track)

    def recv(self, flags=0, copy=True, track=False):
        """recv a dictionary (and a numpy array)

        Args:
            flags (bitmap):
            copy  (bool):
            track (bool): ZMQ recieving options (leave as defaults unless you RTFM of ZMQ)
        """
        try:
            message = self._stream(self.sub_socket.recv_string(flags=flags))
            if self.sub_socket.getsockopt(zmq.RCVMORE):
                dict_ = self.sub_socket.recv_json(flags=flags)
                if not self.sub_socket.getsockopt(zmq.RCVMORE):
                    return message, dict_
                msg = self.sub_socket.recv(flags=flags, copy=copy, track=track)
                buf = memoryview(msg)
                A = numpy.frombuffer(buf, dtype=dict_["dtype"])
                return message, dict_, A.reshape(dict_["shape"])
        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                # print(e)
                return None
            else:
                raise e

    def has_new_data(self, timeout_ms=1):
        """check whether there is unprocessed data in the stream

        Args:
            timeout_ms: How long to wait for data. The default paces loops that
                do nothing else; a caller already paced by a timer passes 0,
                since on Windows any wait lasts a whole timer tick (~16 ms).

        Returns:
            bool: Data is available. (the next recv call will not have to wait or timeout)
        """
        return self.sub_socket.poll(timeout_ms) == zmq.POLLIN

    """
    try:
    raw_message = self.pull_socket.recv(zmq.NOBLOCK)
    except zmq.ZMQError as e:
    if e.errno != zmq.EAGAIN:
    raise
    """


class DataClient(GenericClient):
    _hub = "Datahub"

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)


class CommandClient(DataClient):
    _hub = "Commandhub"
    """used for sending commands to drivers
    we could have used dataclient but having a separate channel gives the possibility of setting addreses separately
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def send(
        self, cmd="", data=None, A=None, flags=0, copy=True, track=False, prefix=None
    ):
        """distribute over ZMQ

        Args:
            datadict (dict) :
                data to be send (must be json serializable)
            A (np.array)    :
                optional array of data
            channel (string):
                subchannel (will be appended to name when sending)
            flags (int) :
                see ZMQ send flags in the ZMQ doku
        """
        if data is None:
            data = {}
        super().send(data, A, " " + cmd, flags, copy, track, prefix)


class ImageClient(DataClient):
    _hub = "Imagehub"

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def send(self, header, data, channel="", flags=0, copy=True, track=False):
        """distribute ove ZMQ

        Args:
            header (dict) :
                dictionary with additional info about the data (must be json serializable)
            data (np.array):
                data array to be send (must be numpy array, will be converted to bytes and reconstructed on the other side according to the header info)
            channel (string):
                subchannel (will be appended to name when sending)
            flags (int):
                see ZMQ send flags in the ZMQ doku
        """
        self.pub_socket.send_string(
            self._topic(self.name + channel), flags | zmq.SNDMORE
        )

        # recv() rebuilds the array in C order from the raw buffer.
        data = np.ascontiguousarray(data)
        header["dtype"] = str(data.dtype)
        header["shape"] = data.shape
        self.pub_socket.send_json(header, flags | zmq.SNDMORE, cls=NpEncoder)
        return self.pub_socket.send(data, flags, copy=copy, track=track)


if __name__ == "__main__":
    pass
