from pytweezer.servers.clients import GenericClient
from pytweezer.servers.configreader import ConfigReader
from pytweezer.servers import zmqcontext
import zmq
import time
import inspect

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
        imgsub=conf['Servers']['Messagehub']['pub']
        imgpub=conf['Servers']['Messagehub']['sub']
        #print(imgpub)
        self.pub_socket.connect(imgpub)
        self.sub_socket.connect(imgsub)


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


    def recv(self, flags=0,copy=True,track=False):
            """recv a dictionary (and a numpy array)

            Args:
                flags,copy,track : ZMQ recieving options (leave as defaults unless you RTFM of ZMQ)
            """
            try:
                message = self.sub_socket.recv_string(flags=flags)
                return message,{}


            except zmq.ZMQError as e:
                print('error')
                if e.errno == zmq.EAGAIN:
                    return None
                else:
                    raise e

    def has_new_data(self):
        ''' check whether a new message has arrived (without reading it)
        '''
        #print('hnd')
        return self.sub_socket.poll(1)==zmq.POLLIN

#print('init Messageclient')
mgsc=MessageClient('testname')
time.sleep(0.01)



def send_msg(msg,errlevel):
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    filename = module.__file__.split('/')[-1] 
    mgsc.send(errlevel,filename+' '+msg)

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



