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

import sys

sys.path.append("../")
sys.path.append("../../")
from pytweezer.servers import zmqcontext
import pytweezer as bc
import zmq
import time
import numpy
import json
from pytweezer.servers.configreader import ConfigReader
import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.floating):
            return float(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class GenericClient:
    _hub = None
    def __init__(self, name="noname", recvtimeout=1000, subscribe=None):
        """send or receive data over balis ZMQ ports
        standardizes datatransfer and connects to sockets according to the central configuration
        Use this class for small (<10k) datasets this data will be logged stored etc,
        there is the Imageclient for larger (imagesize) data.

        Args:
            name (string):
                used when sending data. The data will be published under a channel always starting with the  name
            recvtimeout(int):
                time in ms the receive should wait for a timeout
        """
        self.name = name
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
        conf = ConfigReader.getConfiguration()
        c = conf["Servers"][self._hub]
        host = c["host"]
        pub_port = c["pub_port"]
        sub_port = c["sub_port"]
        # In the xsub/xpub hub, publishers must connect to XSUB (sub_port)
        # and subscribers must connect to XPUB (pub_port).
        publish_endpoint = f"tcp://{host}:{sub_port}"
        subscribe_endpoint = f"tcp://{host}:{pub_port}"
        self.pub_socket.connect(publish_endpoint)
        self.sub_socket.connect(subscribe_endpoint)

    def unsubscribe(self, channel=None):
        """unsubscribe from specific channel

        Args:
            channel (string) :
        """
        if channel is None:
            for entry in self.subscriptions:
                self.sub_socket.setsockopt_string(zmq.UNSUBSCRIBE, entry)
        else:
            self.sub_socket.setsockopt_string(zmq.UNSUBSCRIBE, channel)
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
            self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, channel)
            self.subscriptions.append(channel)
        elif type(channel) == list:
            for ch in channel:
                self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, ch)
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
            self.pub_socket.send_string(self.name + channel, flags | zmq.SNDMORE)
        else:
            self.pub_socket.send_string(prefix + "." + channel, flags | zmq.SNDMORE)

        if not isinstance(A, np.ndarray):

            return self.pub_socket.send_json(datadict, flags, cls=NpEncoder)
        else:
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
            message = self.sub_socket.recv_string(flags=flags)
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

    def has_new_data(self):
        """check whether there is unprocessed data in the stream

        Returns:
            bool: Data is available. (the next recv call will not have to wait or timeout)
        """
        return self.sub_socket.poll(1) == zmq.POLLIN

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
        self, cmd="", data={}, A=None, flags=0, copy=True, track=False, prefix=None
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
        super().send(data, A, " " + cmd, flags, copy, track, prefix)


class ImageClient(DataClient):
    _hub = "Imagehub"
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def send(self, data, header={}, channel="", flags=0, copy=True, track=False):
        """distribute ove ZMQ

        Args:
            data (np.array):
                data array to be send (must be numpy array, will be converted to bytes and reconstructed on the other side according to the header info)
            header (dict) :
                optionsl dictionary with additional info about the data (must be json serializable)
            channel (string):
                subchannel (will be appended to name when sending)
            flags (int):
                see ZMQ send flags in the ZMQ doku
        """
        self.pub_socket.send_string(self.name + channel, flags | zmq.SNDMORE)

        header["dtype"] = str(data.dtype)
        header["shape"] = data.shape
        self.pub_socket.send_json(header, flags | zmq.SNDMORE, cls=NpEncoder)
        return self.pub_socket.send(data, flags, copy=copy, track=track)


if __name__ == "__main__":
    pass
