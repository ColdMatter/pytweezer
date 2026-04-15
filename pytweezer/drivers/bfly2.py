from imageio import imread
from balic.servers import ImageClient
from balic.servers import CommandClient,DataClient
from balic.servers import Properties,PropertyAttribute
from balic.analysis.print_messages import print_error
import time
from termcolor import colored
import copy
import argparse
import numpy as np
from scipy.ndimage import rotate
import logging, sys
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
from sipyco.pc_rpc import Client as RPCClient, simple_server_loop

try:
    import rotpy
    from rotpy.system import SpinSystem
    from rotpy.camera import CameraList
except:
    class pc2():
        ''' dummy class in case driver is not installed '''
        def __init__(self):
            print_error('blackfly driver not installed', 'error')

LOGGER = logging.getLogger(__name__)

class Blackfly():
    ''' Driver based on rotpy's python bindings of of the Spinnaker SDK
    Allows for setting of multiple settings using a single dictionary

    Args:
        serial (int)     : serial number of camera
    '''

    def __init__(self, serial, system: SpinSystem):
        '''
        Initialize a camera with a certain serial number. Creates a System that
        serves as the interface to the camera.

        Args:
            serial: str, the serial number of the camera
            system: SpinSystem, the interface to access all cameras
        '''

        # Exclude non-writable params from printed warnings:

        self.serial = serial
        self.system = system
        cameras = CameraList.create_from_system(self.system, update_cams=True, update_interfaces=True)
        self.camera: rotpy.camera.Camera = cameras.create_camera_by_serial(str(self.serial))
        self.running = False

        self._image_handler = None

    def _synchronize_timestamp(self):
        '''
        Stores the current CPU time and resets the internal clock of the
        cam. This is needed to synchronize the image timestamp with experiment
        control soft timestamps.
        '''
        reset = self.camera.get_node_map().get_node_by_name("GevTimestampControlReset")

        self.cpu_time = time.time()
        self.tick_tock = self._get_camera_node_value_by_name("GevTimestampTickFrequency")
        # tested: resetting + latching + reading takes 100us real time
        reset.execute_node()

    def _get_camera_node_value_by_name(self, name):
        """
        Quickly access a node (property) value by its name.
        """
        value = self.camera.get_node_map().get_node_by_name(name).get_node_value()
        return value

    def set_props(self, props):
        ''' configuring camera

        Args:
            props (dict) : dictionary containing the properties
                keys correspond to funtions called,
                the entries correspond to arguments

        Returns:
            props (dict): the properties that have been set. May differ from the input in case of properties
            incompatible with the camer
        '''
        # stop the cam if it's running, to set properties
        cam_status = self.running
        self.stop()
        nodemap = self.camera.get_node_map()
        self._iter_property_dict(props, nodemap)
        props = self.get_props()

        # restart the cam if it has been running before
        if cam_status:
            self.start()
        return props

    def _iter_property_dict(self, dic, nodemap):
        """
        Helper function to recurse into a dictionary of properties and write them
        to the camera.
        """
        for key, val in dic.items():
            print("Setting {} to {}".format(key, val))
            # I. we're at the lowest level, set the value in the node
            if not isinstance(val, dict):
                node = nodemap.get_node_by_name(key)
                if node is None: continue
                if node.is_writable():
                    try:
                        node.set_node_value(val)
                    except rotpy.system.SpinnakerAPIException as e:
                        print(e)
                else:
                    print_error('bfly2.py {}: Node {} not writable.'.format(self.serial, key), 'warning')

            # II. It's a SpinEnumNode, set the 'value' from the dict
            elif "options" in val:
                node = nodemap.get_node_by_name(key)
                if node.is_writable():
                    enum_item = node.get_entry_by_name(val['value'])
                    node.set_node_value(enum_item)
                else:
                    print_error('bfly2.py {}: Node {} not writable.'.format(self.serial, key), 'warning')

            # III. it's a SpinTreeNode, we iterate on
            else:
                self._iter_property_dict(val, nodemap)

    def _get_camera_properties(self):
        '''
        Returns a dictionary of all available nodes of the camera in hierarchical
        order.
        '''
        nodemap = self.camera.get_node_map()
        root_node = nodemap.get_node_by_name("Root")
        props = self._iterate_nodes(root_node)
        return props["Root"]

    def _get_system_properties(self):
        """
        Returns a dictionary of all available nodes of `self.system` in
        hierarchical order.
        """
        sysnm = self.system.get_tl_node_map()
        sys_root = sysnm.get_node_by_name("Root")
        props = self._iterate_nodes(sys_root)
        return props["Root"]

    def get_props(self):
        ''' Get all current properties (entries that can be configured) from camera

        Returns:
            dict:
                Properties of the camera
        '''
        props = self._get_camera_properties()
        return props

    def get_image(self):
        ''' Read image from camera/

        Returns:
            Blackfly image. Not a numpy array!
            None.   In case of error
        '''
        try:
            im = self.camera.get_next_image()
            imc = im.deep_copy_image(im)
            im.release()
            return imc

        except Exception as e:
            print_error('Cam {0}: Error get image: {1}'.format(self.serial, e), 'warning')
            return None

    def initialize_cam(self):
        """
        Initialize the camera.
        :return:
        """
        self.camera.init_cam()
        self._synchronize_timestamp()

    def __enter__(self):
        '''
        Safely initialize the camera runtime environment.
        '''

        self.initialize_cam()
        return self

    def start(self, callback=None):
        """
        Attach an image callback and start capturing images.
         """
        if not self.running:
            if callback is not None:
                self._image_handler = self.camera.attach_image_event_handler(callback)
            self.camera.begin_acquisition()
            self.running = True

    def stop(self):
        """
        Stop capturing images and detach the image callback.
        """
        if self.running:
            self.camera.end_acquisition()
            if self._image_handler is not None:
                self.camera.detach_image_event_handler(self._image_handler)
                self._image_handler = None
            self.running = False

    def finalize_cam(self):
        """
        Stop acquisition and release the cam handler.
        :return: None
        """
        self.stop()
        self.camera.deinit_cam()
        self.camera.release()

    def __exit__(self,type,value,traceback):
        '''exit runtime context
        ensure even in the event of exit the camera is disconnected
        '''
        self.finalize_cam()

    def _iterate_nodes(self, node, lvl=0, dic={}):
        """
        Iterate through the nodes (properties) of a camera recursively and store
        the values in a dictionary.
        """
        # SpinTreeNodes are not properties themselves but represent categories.
        if type(node) == rotpy.node.SpinTreeNode:
            dic[node.get_name()] = {}
            for n in node.get_children():
                if n.is_available():
                    # print(lvl*"    ", n.get_name())
                    self._iterate_nodes(n, lvl=lvl+1, dic=dic[node.get_name()])

        # all available nodes that represent properties get stored with their values
        elif node.is_available():
            if type(node) is rotpy.node.SpinEnumNode:
                dic[node.get_name()] = {"value":node.get_node_value().get_enum_name(), "options": node.get_entries_names()}

            elif type(node) is not rotpy.node.SpinCommandNode:
                dic[node.get_name()] = node.get_node_value()

        return dic




def run_server(
    host: str,
    port: int,
    stream_name: str = "imagemx2",
    image_dir: str | None = None,
    timeout: float = 5.0,
    simulate: bool = False,
):
    camera = Blackfly(serial=12345678, system=SpinSystem())

    LOGGER.info(
        "Starting ImagEM X2 RPC server host=%s port=%s stream=%s simulate=%s",
        host,
        port,
        stream_name,
        simulate,
    )
    simple_server_loop(
        {"camera": camera},
        host=host,
        port=int(port),
        description="ImagEM X2 RPC server",
    )

def main():

    parser = argparse.ArgumentParser(description="Run ImagEM X2 sipyco RPC server")
    parser.add_argument("--host", default="127.0.0.1", help="RPC bind host")
    parser.add_argument("--port", type=int, default=3251, help="RPC bind port")
    parser.add_argument("--stream-name", default="imagemx2", help="Image stream name")
    parser.add_argument(
        "--image-dir", default=None, help="optional TIFF autosave directory"
    )
    parser.add_argument(
        "--timeout", type=float, default=5.0, help="frame wait timeout in seconds"
    )
    parser.add_argument(
        "--simulate", action="store_true", help="use simulated camera backend"
    )
    parser.add_argument("--log-level", default="INFO", help="Python log level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    run_server(
        host=args.host,
        port=args.port,
        stream_name=args.stream_name,
        image_dir=args.image_dir,
        timeout=args.timeout,
        simulate=args.simulate,
    )


if __name__ == "__main__":
    main()
