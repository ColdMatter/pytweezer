import zmq
zmqcontext = zmq.Context()
EVENT_MAP = {}
# print("Event names:")
for name in dir(zmq):
    if name.startswith('EVENT_'):
        value = getattr(zmq, name)
        # print("%21s : %4i" % (name, value))
        EVENT_MAP[value] = name
from .properties import Properties
from .properties import PropertyAttribute
from .clients import DataClient
from .clients import ImageClient
from .clients import CommandClient
from .configreader import balipath
from .messageclient import send_error,send_warning,send_info,send_debug
