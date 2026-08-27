import zmq

from .sipyco_patches import apply_patches as _apply_sipyco_patches

_apply_sipyco_patches()

zmqcontext = zmq.Context()
EVENT_MAP = {}
# print("Event names:")
for name in dir(zmq):
    if name.startswith("EVENT_"):
        value = getattr(zmq, name)
        # print("%21s : %4i" % (name, value))
        EVENT_MAP[value] = name
from pytweezer.configuration.paths import icon_path, tweezerpath

from .clients import CommandClient, DataClient, ImageClient
from .messageclient import send_debug, send_error, send_info, send_warning
from .properties import Properties, PropertyAttribute
