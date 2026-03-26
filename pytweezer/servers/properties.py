from pytweezer.servers import zmqcontext, EVENT_MAP
from pytweezer.servers import configreader as cr
from pytweezer.servers.xsub_xpub import event_monitor
import copy
import zmq
import threading
import time
import logging, sys
from zmq.utils import jsonapi
# logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
from pytweezer.analysis.print_messages import print_error
"""Configuration:  deep, fundamental property of the system.
                    (which hardware is running, drivers available)

   Properties:     usually refer to objects (position of ROI)

   Settings/Options: (currently not used)

   Preferences:    minor choices, no influence on data !! (shape of mouse pointer, positions of Windows,...)
"""


class PropertyAttribute:
    """
    access properties as Attributes.
    The class object using this attribute must have an attribute called _props of type Properties.
    The value property can be used to do basic manipulation of the property e.g. get, set, addition
    att = PropertyAttribute(propname, defaultval, parent=self)
    exmaples:
    getting: x = att.value. gets the property value and sets x with it
    addition: att.value += 1. this changes both the property and self.att
    indexing: element_2 = att.value[2]. this gets the second element of the property
    att.value.append() doesn't work! do:
    lst = att.value
    lst.append()
    att.value = lst

    x = att gets the property value and sets x with it
    att = x sets the property value BUT OVERWRITES att (i.e. if x is an int, att will also become an int),
    so it only works once. Therefore, using att.value is preferred.
    """

    def __init__(self, propname, defaultval, parent=None):
        """ init

        Args:
            propname (str): name of the property the attribut should access
            defaultval : default value of the property in case it doesn't exist yet
                        can be any type a dictionary entry can be (hashable)
                        (needs to be compatible with json)
            parent: usually self. the object which has _props. required for the value property to function
        """
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

    # these work but I've disabled them because they could be confusing
    # def append(self, value):
    #     self._value.append(value)
    #     self.parent._props.set(self._propname, self._value)
    #
    # def extend(self, values):
    #     self._value.extend(values)
    #     self.parent._props.set(self._propname, self._value)

    def __get__(self, obj, t):
        return obj._props.get(self._propname, self._default)

    def __set__(self, obj, v):
        obj._props.set(self._propname, v)


class Properties(threading.Thread):

    @staticmethod
    def _configure_tcp_socket(socket, endpoint):
        if isinstance(endpoint, str) and endpoint.startswith('tcp://'):
            socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
            if hasattr(zmq, 'TCP_KEEPALIVE_IDLE'):
                socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
            if hasattr(zmq, 'TCP_KEEPALIVE_INTVL'):
                socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 30)
            if hasattr(zmq, 'TCP_KEEPALIVE_CNT'):
                socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, 5)
            socket.setsockopt(zmq.RECONNECT_IVL, 100)
            socket.setsockopt(zmq.RECONNECT_IVL_MAX, 2000)
        socket.setsockopt(zmq.LINGER, 0)

    def __init__(self, name, initfromfile=False):
        threading.Thread.__init__(self, daemon=True)
        """ managing configuration and properties within bali control center
        Keeps an updated and synchronized copy of the property dictionary

        Args:
            name (string): name of process (will be used as default in all set, get, etc.
            initfromfile (bool): Leave as default(False). Was used in the earlier versions.
                Currently only the propertylogger itself inits from a file on start up.
                All others should ask the propertylogger for an up to date set

        """

        conf = cr.Config()
        # iniitialize dictionaries
        self.properties_lock = threading.Lock()
        # initialize sockets
        hub_sub_endpoint = conf['Servers']['Propertyhub']['sub']
        hub_pub_endpoint = conf['Servers']['Propertyhub']['pub']
        self.pub_socket = zmqcontext.socket(zmq.PUB)
        self._configure_tcp_socket(self.pub_socket, hub_sub_endpoint)
        self.pub_socket.connect(hub_sub_endpoint)
        self.sub_socket = zmqcontext.socket(zmq.SUB)
        self._configure_tcp_socket(self.sub_socket, hub_pub_endpoint)
        self.sub_socket.connect(hub_pub_endpoint)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "Prop")

        pub_mon = self.pub_socket.get_monitor_socket()
        pub_mon_thread = threading.Thread(target=event_monitor, args=(pub_mon, 'Props: ' + name, 'PUB'))
        pub_mon_thread.start()
        sub_mon = self.sub_socket.get_monitor_socket()
        sub_mon_thread = threading.Thread(target=event_monitor, args=(sub_mon, 'Props: ' + name, 'SUB'))
        sub_mon_thread.start()

        time.sleep(0.01)
        if not initfromfile:
            self.init_socket = zmqcontext.socket(zmq.REQ)
            logger_endpoint = conf['Servers']['Propertylogger']['rep']
            self._configure_tcp_socket(self.init_socket, logger_endpoint)
            self.init_socket.setsockopt(zmq.RCVTIMEO, 5000)
            self.init_socket.setsockopt(zmq.SNDTIMEO, 5000)
            self.init_socket.connect(logger_endpoint)
            try:
                self.init_socket.send_string('INIT?')
                self.init_socket.recv_string()
                self.properties = self.init_socket.recv_json()
            except Exception as error:
                print_error(f'properties.py init handshake failed ({error}); falling back to property file', 'warning')
                self.properties = cr.Properties()
        else:
            self.properties = cr.Properties()
        self.recent_changes = set()  # keeps recent changes

        checkThread = threading.Thread(target=self.check)
        checkThread.start()

        # start socket listening thread
        self.start()
        self.name = name
        if not name in self.properties:
            n = self.get('/' + name, {})
        logging.debug(self.properties)

        self.crashed = False

    def _parsekey(self, key):
        ''' transform key into list of strings
        each entry is a key for one level of a multilayer dictionary

        Args:
            key (str): key of a pytweezer multylayer dictionary.
                (follows similar nomenclature like unix file system)

        Returns:
            [(str)] : list of strings
        '''
        if type(key) != list:
            if key[0] != '/':
                if key[-1] == '/':
                    key = key[:-1]
                key = '/' + self.name + '/' + key
            keys = key.split('/')
        else:
            keys = key
        # if the key started with '/'  an empty entry has to be removed
        if keys[0] == '':
            keys = keys[1:]
        return keys

    def set(self, key, value):
        """set property

        Args:
            key (string) or (list of strings) :  property to be set
                adressing is similar to filesystem
                if the string is starting with '/ start from dictionary root
                else start from entry=name


            value (anything): Anything you consider a useful value.
                Must be json serializable! (no self defined classes unless you know ...)

            _global (bool) access parameters of other programs
                    keys will the have to be a list with the first entry being the name of the program whose
                    Parameters should be modified

        """
        if self.get(key) != value:
            keys = self._parsekey(key)
            self._set(keys, value)
            self._send({'keys': keys, 'value': value})
        # print('properties.py Sending: {} with value {}'.format(key, value))

    def _set(self, keys, value):
        with self.properties_lock:
            prop = self.properties
            # if the property does not exist we iteratively create it
            for key in keys[:-1]:
                if not key in prop:
                    prop[key] = {}
                prop = prop[key]
            # else:
            if isinstance(value, dict) and 'options' in value.keys():
                if not keys[-1] in prop: prop[keys[-1]] = {}
                # print('_set received a dict')
                # print('now setting:', value)
                prop[keys[-1]] = copy.deepcopy(value)
            elif keys[-1] in prop and isinstance(prop[keys[-1]], dict) and 'options' in prop[keys[-1]].keys():
                # print('_set found a dict')
                # print('which was:', prop[keys[-1]])
                # print('now setting:', value)
                if value not in prop[keys[-1]]['options']:
                    print('value {} not in options {}. no change made to {}'.format(value, prop[keys[-1]]['options'],
                                                                                    keys[-1]))
                else:
                    prop[keys[-1]]['value'] = copy.deepcopy(value)
            else:
                prop[keys[-1]] = copy.deepcopy(value)
            # for i in range(len(keys)):
            #    self.recent_changes.add('/'+'/'.join(keys[:i+1]))
            self.recent_changes.add('/' + '/'.join(keys))

    def _send(self, data, flags=0):
        # self.pub_socket.send(bytes('Propertychange_'+self.name,'utf8'),flags|zmq.SNDMORE)
        self.pub_socket.send_string('Propertychange_' + self.name, flags | zmq.SNDMORE)
        # print('properties.py sending _send')
        return self.pub_socket.send_json(data, flags)

    def delete(self, key):
        ''' delete an entry including its subentries

        Args:
            key  (str):  entry to be deleted

        Returns:
            None.
        '''
        keys = self._parsekey(key)
        self._del(keys)
        self._send({'delete': keys})

    def _del(self, keys):
        # print(self.name,' properties._del deleting',keys)
        ''' delete enty from dictionary '''
        with self.properties_lock:
            prop = self.properties
            for key in keys[:-1]:
                if not key in prop:
                    prop[key] = {}
                prop = prop[key]
            # logging.debug(prop)
            if keys[-1] in prop:
                del prop[keys[-1]]
            self.recent_changes.add('/' + '/'.join(keys[:-1]))
            # for i in range(len(keys)):
            #    self.recent_changes.add('/'+'/'.join(keys[:i+1]))

    def _recv(self, flags=0):
        """recv a numpy array"""
        parts = self.sub_socket.recv_multipart(flags=flags)
        if len(parts) < 2:
            return
        message = parts[0].decode('utf-8', errors='ignore')
        logging.debug(self.name,' received: ',message)
        try:
            md = jsonapi.loads(parts[1])
        except Exception as error:
            print_error(f'properties.py malformed property payload: {error}', 'warning')
            return
        # in _default_decoder.decode(s)
        # Triggered by pushing experiment with scan
        """
            File "/usr/lib/python3.10/json/__init__.py", line 346, in loads
                return self._deserialize(msg, lambda buf: jsonapi.loads(buf, **kwargs))
            File "/home/bali/.local/lib/python3.10/site-packages/zmq/utils/jsonapi.py", line 34, in loads
                return _default_decoder.decode(s)
            File "/usr/lib/python3.10/json/decoder.py", line 337, in decode
                return load(recvd)
            File "/home/bali/.local/lib/python3.10/site-packages/zmq/sugar/socket.py", line 940, in <lambda>
                obj, end = self.raw_decode(s, idx=_w(s, 0).end())
            File "/usr/lib/python3.10/json/decoder.py", line 355, in raw_decode
                self._recv()
                self._recv()
                self.run()
            raise JSONDecodeError("Expecting value", s, err.value) from None
            json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
        """
        logging.debug(self.name,' received: ',message,md)
        if 'delete' in md:
            # print(self.name,' properties._recv deleting',md['delete'])
            self._del(md['delete'])
        elif 'keys' in md and 'value' in md:
            self._set(md['keys'], md['value'])
            # logging.debug('setting: ',md['keys'],md['value'])

    def get(self, key, defaultvalue=None):
        """ returns values from dictionary
        for details see set(). Values are always deep copies

        Args:
            key (str): key of the dictionary
            defaultvalue: used to create the entry in case it is not yet existing

        Returns:
            the entry from the property dictionary

        It is a good habit to always give reasonable default values so the property tree can
        create and maintain itself.
        """
        if key == '/':
            with self.properties_lock:
                return copy.deepcopy(self.properties)
        elif key[-1] == '/':
            key = key[:-1]
        keys = self._parsekey(key)
        return self._get(keys, defaultvalue)

    def _get(self, keys, defaultvalue):
        # print('_get:')
        # print(keys)
        try:
            with self.properties_lock:
                prop = self.properties
                for key in keys[:-1]:
                    prop = prop[key]
                value = copy.deepcopy(prop[keys[-1]])
                if isinstance(value, dict) and 'options' in value.keys():
                    # print('_get found a dict')
                    # print('which is:',value)
                    value = value['value']

                    # prop[keys[-1]]=copy.deepcopy(value['value'])

        except KeyError:
            logging.debug('properties.py key does not exist')
            self._set(keys, defaultvalue)
            self._send({'keys': keys, 'value': defaultvalue})
            value = defaultvalue
            print('tried to _get an unset property {}'.format(keys))
            if isinstance(value, dict) and 'options' in value.keys():
                value = value['value']
            print('setting default value: {}'.format(value))
        return value

    def changes(self, includeparent=True):
        ''' return keys of all entries that have changed since the last call of this function

        Args:
            includeparent (bool): if True the parent classes will be included in the list of changes

        Returns:
            set(): set of changes since last call of this function
        '''
        with self.properties_lock:
            changes = self.recent_changes
            self.recent_changes = set()

        if includeparent:
            chang = set()
            for key in changes:
                keys = key[1:].split('/')
                for i in range(len(keys)):
                    chang.add('/' + '/'.join(keys[:i + 1]))
            return chang
        else:
            return changes

    def check(self):
        lastTime = 0
        timeout = 40
        interval = 30
        # while True:
        #     time.sleep(0.5)
        #     t = time.time()
        #     if t - lastTime > interval:
        #         lastTime = t
        #         check_time = self.get('/Checker/check_time', 0)
        #         if abs(check_time - t) > timeout:
        #             print_error('Property check timeout {}; time.time {}, check_time {}'.format(self.name, t, check_time))
        #             if self.name == 'ADwindriver' and not self.crashed:
        #                 self.send_alert_to_mattermost()
        #             self.crashed = True
        #         # else:
        #             # self.crashed = False
        


    def run(self):
        while True:
            try:
                self._recv()
            except Exception as error:
                print_error(f'properties.py receive loop error: {error}', 'warning')
                time.sleep(0.05)


if __name__ == "__main__":
    pass
