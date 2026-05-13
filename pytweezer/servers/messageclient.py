import logging
import time

import zmq

from pytweezer.logging_utils import get_logger
from pytweezer.servers.clients import GenericClient
from pytweezer.servers.configreader import ConfigReader

class MessageClient(GenericClient):
    ''' Standardized way of sending messages over the message stream
    '''
    def __init__(self,name):
        '''

        Args:
            name (str): name of your message stream (will be in front of the message)
        '''
        super().__init__(name)

    def _connect(self):
        conf=ConfigReader.getConfiguration()
        c = conf['Servers']['Messagehub']
        host = c['host']
        pub_port = c['pub_port']
        sub_port = c['sub_port']
        publish_endpoint = f"tcp://{host}:{sub_port}"
        subscribe_endpoint = f"tcp://{host}:{pub_port}"
        self.pub_socket.connect(publish_endpoint)
        self.sub_socket.connect(subscribe_endpoint)


    #beware packets might be lost when send via pub and connection is not establishedgtgt
    def send(self,channel='info',message='', flags=0):
            ''' distribute over ZMQ

            Args:
                datadict (dict) : data to be send (must be json serializable)
                A (np.array)    : optional array of data
                channel (string): subchannel (will be appended to name when sending)
                flags (int) : see ZMQ send flags in the ZMQ doku
            '''
            self.pub_socket.send_string(channel+': '+message,flags)

    def send_json(self, channel, payload, flags=0):
            """Publish a topic plus structured JSON payload."""
            self.pub_socket.send_string(channel, flags | zmq.SNDMORE)
            return self.pub_socket.send_json(payload, flags)


    def recv(self, flags=0,copy=True,track=False):
            """recv a dictionary (and a numpy array)

            Args:
                flags,copy,track : ZMQ recieving options (leave as defaults unless you RTFM of ZMQ)
            """
            try:
                message = self.sub_socket.recv_string(flags=flags)
                if self.sub_socket.getsockopt(zmq.RCVMORE):
                    payload = self.sub_socket.recv_json(flags=flags)
                    return message, payload
                return message, {}
            except zmq.ZMQError as e:
                if e.errno == zmq.EAGAIN:
                    return None
                else:
                    raise e

    def has_new_data(self):
        ''' check whether a new message has arrived (without reading it)
        '''
        #print('hnd')
        return self.sub_socket.poll(1)==zmq.POLLIN

_message_client = None
_LEVELS = {
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def _get_default_client_name():
    """Read default stream name from config, keeping legacy fallback."""
    conf = ConfigReader.getConfiguration()
    messagehub = conf.get('Servers', {}).get('Messagehub', {})
    return messagehub.get('stream_name', 'testname')


def get_message_client(name=None):
    """Return the shared MessageClient, creating it on first use."""
    global _message_client
    if _message_client is None:
        if name is None:
            name = _get_default_client_name()
        _message_client = MessageClient(name)
        # Give PUB/SUB sockets a brief moment to establish before first send.
        time.sleep(0.01)
    return _message_client



def send_msg(msg, errlevel):
    logger = get_logger("pytweezer")
    level = _LEVELS.get(str(errlevel).lower())
    if level is None:
        logger.error("messageclient.py: %s is no valid key! Message = %s", errlevel, msg)
        return
    logger.log(level, str(msg), stacklevel=2)

#Convenience functions
#----------------------------
#send error message
def send_error(msg):
    ''' send error message

    Args:
        msg (str): message
    '''
    send_msg(msg,'Error')
#send warning message
def send_warning(msg):
    ''' send warning message

    Args:
        msg (str): message
    '''
    send_msg(msg,'Warning')
#send info message
def send_info(msg):
    send_msg(msg,'Info')
#send debug message
def send_debug(msg):
    send_msg(msg,'Debug')



