import time
import numpy as np
import subprocess
import copy
from termcolor import colored
from scipy.ndimage import rotate

from pytweezer.GUI.pytweezerQt import BMainWindow
from pytweezer.analysis.print_messages import print_error

import PyQt5.QtCore as Qt
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import qApp, QApplication, QWidget, QTableView, QPushButton
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QMainWindow, QDialog, QDockWidget
from PyQt5.QtWidgets import QMdiArea, QTreeView, QDirModel, QHeaderView, QSpinBox
from PyQt5.QtWidgets import QAbstractItemView, QAction, QMdiSubWindow, QLabel
from PyQt5.QtWidgets import QSizePolicy, QComboBox, QGridLayout, QDialogButtonBox
from PyQt5.QtWidgets import QMenu, QCheckBox, QLineEdit, QDoubleSpinBox, QTabWidget
from PyQt5.QtWidgets import QFileDialog

import pytweezer
#from pytweezer.drivers.blackfly import Blackfly
from pytweezer.servers import Properties, tweezerpath, PropertyAttribute, DataClient, ImageClient, CommandClient


try:
    import rotpy
    from rotpy.system import SpinSystem
    from rotpy.camera import CameraList
    sim = False
except:
    class SpinSystem:
        ''' dummy class in case driver is not installed '''
        def __init__(self):
            print_error('camerahub.py - blackfly driver not installed', 'error')
    class CameraList:
        def __init__(self):
            pass

    sim = True
#buttonStyleSheet = "QPushButton {" "color: blue;" "background-color: rgb(0, 255, 127);" "}"


class CameraHub:
    """
    a manager for the cameras
    TODO: work out audoconfigure
    TODO: how to load configuration with minimal interruption to acquisition
    TODO: can the cameras be started without doing the configuration?
    TODO: have camera hub detect and display when a camera crashes
    """

    def __init__(self, props, parent=None, cameras_for_startup={}):
        self.props = props
        self.parent = parent
        self.system = SpinSystem()
        self.camservers = {}
        self.start_cam(None, None, startup=True, cameras_for_startup=cameras_for_startup)

    def start_cam(self, name, serial, startup=False, cameras_for_startup=None):
        print_error('CameraHub - start_cam()', 'weak')
        if name in self.camservers:
            self.stop_cam(name)
        self.camlist, self.serials = self.get_camlist()

        if cameras_for_startup is None:
            cameras_for_startup = {name: serial}

        for name, serial in cameras_for_startup.items():
            serial_str = str(serial)
            if serial_str not in self.serials:
                print_error('CameraHub - start_cam(): Camera {0} not connected or in use by another'
                            ' instance.'.format(name), 'error')
                camWidget = self.parent.camWidgets[name]
                camWidget.label.setStyleSheet("background-color: red")
            else:
                cam = CamServer(name, serial, self.system)
                if not cam.cam.error:
                    self.camservers[name] = cam
                    cam.run()
                    print_error('CameraHub - start_cam(): Camera {0} started.'.format(name),
                                'success')

    def stop_cam(self, name):
        if name in self.camservers:
            cam = self.camservers[name]
            cam.cam.stop()
            cam.stop()
            del cam
            print_error('CameraHub - stop_cam(): Camera {0} stopped.'.format(name), 'info')
        else:
            print_error('CameraHub - stop_cam(): Camera {0} not running.'.format(name), 'info')

    def force_ip(self, name, serial):
        print_error('CameraHub - force_ip(): Not implemented yet.', 'info')
        return  # not working yet
        # try:
        #     cam = CamServer(name, serial, self.system)
        #     cam.cam.stop()
        #     cam.cam.initialize_cam()
        #     cam.cam.camera.force_ip()
        # except Exception as e:
        #     print_error('camerahub.py - force_ip(): Camera can\'t be re-configured, {0}.'.format(e), 'error')

    def config_cam(self, name):
        if name in self.camservers:
            cam = self.camservers[name]
            cam.configure()
        else:
            print_error('CameraHub - config_cam(): Camera {0} not running.'.format(name), 'info')

    def get_camlist(self):
        '''
        Detects connected cameras from SpinSystem
        '''
        if not sim:
            camlist = CameraList.create_from_system(self.system, update_cams=True, update_interfaces=True)
            serials = []
            for i in range(camlist.get_size()):
                try:
                    c = camlist.create_camera_by_index(i)
                    c.init_cam()
                    serial = c.camera_nodes.DeviceSerialNumber.get_node_value()
                    serials.append(serial)
                    c.deinit_cam()
                    c.release()
                except Exception as e:
                    # get autoforce IP working
                    # c.force_ip()
                    print_error("CameraHub - get_camlist(): " + str(e), 'error')
            return camlist, serials
        else:
            return [], []

class CamServer():
    '''
    Controls the camera and handles the communication with the Image and Data bus.
    '''
    _resolution = PropertyAttribute('resolution [m per pixel]',(0.00001,0.00001))
    _conf_name = PropertyAttribute('Configuration_name','DefaultConfig')

    _autoreconfigure = PropertyAttribute('Autroreconfigure',True)

    def __init__(self, name, serial, system):
        '''
        initializes the driver.  Set the properties to the camera. If configuration does not exist,
        create new configuration with current values from camera

        Args:
            name (str): Name of the camera. Will be used to identify the properties and to create a named
                        imagestream
            serial    : Serial number of the camera
        '''
        name = name.split('/')[-1]
        self.imstream = ImageClient(name)
        self.cmdstream = CommandClient(name)
        self.cmdstream.subscribe(name)
        self.name = 'Cameras/'+name
        self.indexq = DataClient(self.name)
        self.indexq.subscribe(['Experiment.start'])
        self.experiment_timings = [[0,{'_run':0,'_repetition':0,'_task':0}]]
        self._props = Properties(self.name)
        self.confname = copy.copy(self._conf_name)
        self.system = system

        self.lastinfo = (0,0,0)
        self.imgindex = 0

        self.cam = Blackfly(serial, self.system, self._props, name=self.name)
        if self.cam.error:
            return
        self.configure()
        self.connected = True


    def configure(self):
        # write the properties of the configuration to the cam
        cam_props = self._props.get(self._conf_name, 'Empty')
        with self.cam:
            # if the configuration doesn't exist, we create it from current settings
            if cam_props == 'Empty':
                cam_props = self.cam.getProps()
                self._props.set(self._conf_name, cam_props)

            else:
                cam_props = self.cam.setProps(cam_props)
                self._props.set(self._conf_name, cam_props)
        self._cam_props = cam_props

    def update_experiment_indices(self):
        """  Keep list of experiment starts updated. Copied from bfly.py"""

        while self.indexq.has_new_data():
            #print('datamgr.py data incomming!')
            recvmsg=self.indexq.recv()
            A=None
            if len(recvmsg)==2:
                msg,msg_dic=recvmsg
            elif len(recvmsg)==3:
                msg,msg_dic,A=recvmsg
            else:
                print_error('CamServer() - datamgr message error', 'error')
                return
            #print(msg)
            if msg=='Experiment.start':
                indices=dict((k,msg_dic[k]) for k in ('_run','_repetition','_task') if k in msg_dic)
                timestamp=msg_dic['_starttime']
                self.experiment_timings.append([timestamp,indices])
                while len(self.experiment_timings)>20:
                    del self.experiment_timings[0]

    def run(self):
        """
        Sends images from the camera to the imagestream and initiates reconfiguration of the camera in case
        properties have changed.

        """

        # Start the cam with a callback function for image acquisition
        self.cam.initialize_cam()
        self.cam.start(self._image_callback)

        # TODO: implement autoreconfigure if we really want it. Best way would be a timed callback instead of infinite loop.

    def stop(self):
        """
        Stops image acquisition and finalizes the camera.
        """
        self.cam.finalize_cam()

    def _image_callback(self, handler, camera, image):
        """
        Callback function for camera image handling that sends the image via the imagestream.
        :param handler: rotpy.ImageEventHandler
        :param camera: camera object
        :param image: image object (not an array!)
        :return: None
        """
        try:
            imagec = image.deep_copy_image(image)
            image.release()
            self.sendimage(imagec)
        except Exception as e:
            print_error('CamServer() - _image_callback() - {0}: {1}'.format(self.name, e))

    def _image_to_array(self, im):
        ''' Convert blackfly image into numpy array.

        Args:
            im: Image from blackfly driver

        Returns:
            numpy.array(uintn).    The image as a numpy array, with bit depth n.
        '''
        h = im.get_height()
        w = im.get_width()
        bit_depth = im.get_bits_per_pixel()
        if bit_depth == 8:
            dt = np.uint8
        elif bit_depth == 16:
            # for some reason the buffer order is changed to big endian in MONO16 pixel format
            dt = np.dtype(np.uint16)
            dt = dt.newbyteorder('>')
        ima = np.frombuffer(im.get_image_data(), dtype=dt)
        ima = ima.reshape(h, w)
        ima = np.flipud(ima)
        ima = rotate(ima, -90)

        return ima

    def generate_image_index(self, _task=0, _repetition=0, _run=0):
        """
        in the current measurement count the images. If the next run has started (either task,repetition or run changes)
        reset the counter(imgindex).
        Directly copied from bfly.py
        """
        measIndex = (_task, _repetition, _run)
        if self.lastinfo == measIndex:
            self.imgindex += 1
        else:
            self.imgindex = 0
            self.lastinfo = measIndex

    def sendimage(self, im):
        ''' send blackfly camera image via the imagestream
        The image is converted to numpy array, additional information like binning,
        timestamp, ... is added to a dictionary and both are sent to the imagestream


        Args:
            im: Blackfly image object

        Returns:
            None

        '''
        im_arr = self._image_to_array(im)
        image_timestamp = self.cam.cpu_time + im.get_frame_timestamp()/self.cam.tick_tock # timestamp of cam fpga clock
        run_task = self.find_experiment_indices(image_timestamp)
        self.generate_image_index(run_task)
        if time.time() - image_timestamp > 5:
            print_error('CamServer() - Image readout delayed of {1} by {:.0f}s'.format(time.time()-image_timestamp, self.name), 'warning')

        # collect image metadata and store it in info dict
        gain = self.cam._get_camera_node_value_by_name("Gain")
        offset = [im.get_offset_x(),
                  im.get_offset_y()]
        binning = [self.cam._get_camera_node_value_by_name("BinningHorizontal"),
                   self.cam._get_camera_node_value_by_name("BinningVertical")]
        img_resolution = [res*bin for res,bin in zip(self._resolution, binning)]
        info = {}
        info['timestamp'] = image_timestamp
        info['_imgresolution'] = img_resolution
        info['_offset'] = offset
        info['_imgindex'] = self.imgindex
        info['camgain'] = gain
        info['bits_per_pixel'] = im.get_bits_per_pixel()
        info['pixel_format'] = im.get_pix_fmt()
        info.update(run_task)

        self.imstream.send(im_arr, info)


    def find_experiment_indices(self,timestamp):
        ''' find the run,task and repetition for a given timestamp
            Copied from bfly.py
        '''
        self.update_experiment_indices()
        tstamps=[i[0] for  i in self.experiment_timings]
        tstamps = np.array(tstamps)

        index = 0
        if timestamp > tstamps[0]:
            index = np.argwhere(tstamps-timestamp < 0)[-1][0]
        return self.experiment_timings[int(index)][1]


class Blackfly():
    ''' Wrapper class for proprietary blackfly driver.
    Allows for setting of multiple settings using a single dictionary

    Args:
        serial (int)     : serial number of camera
    '''
    def __init__(self, serial, system, props, name):
        '''
        Initialize a camera with a certain serial number. Creates a System that
        serves as the interface to the camera.

        Args:
            serial: str, the serial number of the camera
            system: SpinSystem, the interface to access all cameras
        '''

        self._props = props
        self._name = name.split('/')[-1]

        # Exclude non-writable params from printed warnings:
        self.non_writables = np.array(['AutoGainLowerLimit', 'AutoGainUpperLimit', 'DeviceVendorName',
                                       'DeviceModelName', 'DeviceVersion', 'DeviceSerialNumber',
                                       'DeviceID', 'pgrSensorDescription', 'DeviceSVNVersion', 'DeviceFirmwareVersion',
                                       'DeviceScanType', 'DeviceTemperature', 'pgrDeviceUptime',
                                       'pgrPowerSupplyVoltage', 'pgrPowerSupplyCurrent', 'DeviceMaxThroughput',
                                       'AutoExposureTimeLowerLimit', 'DecimationHorizontal',
                                       'AutoExposureTimeUpperLimit', 'pgrAutoExposureCompensationLowerLimit',
                                       'pgrAutoExposureCompensationUpperLimit', 'LineMode',
                                       'AcquisitionFrameRate', 'AcquisitionStatus', 'PixelCoding', 'SensorWidth',
                                       'SensorHeight', 'WidthMax', 'HeightMax', 'BinningHorizontal', 'PixelSize',
                                       'PixelColorFilter', 'PixelDynamicRangeMin', 'PixelDynamicRangeMax',
                                       'pgrDefectPixelCorrectionType', 'UserSetCurrent', 'LineStatus', 'LineStatusAll',
                                       'DataFlashPageSize', 'DataFlashPageCount', 'GevCurrentIPAddress',
                                       'GevCurrentSubnetMask', 'GevCurrentDefaultGateway', 'GevGVCPPendingTimeout',
                                       'GevTimestampTickFrequency', 'GevTimestampValue', 'GevStreamChannelCount',
                                       'GevSupportedOption', 'GevVersionMajor', 'GevVersionMinor',
                                       'GevDeviceModeIsBigEndian', 'GevDeviceModeCharacterSet', 'GevInterfaceSelector',
                                       'GevMACAddress', 'GevFirstURL', 'GevSecondURL', 'GevNumberOfInterfaces',
                                       'GevLinkSpeed', 'GevMessageChannelCount', 'GevPrimaryApplicationSocket',
                                       'GevPrimaryApplicationIPAddress', 'GevSCPDirection', 'GevSCSP', 'PayloadSize',
                                       'TransmitFailureCount', 'GainSelector', 'ExposureAuto', 'AcquisitionFrameRateAuto'])

        self.serial = serial
        self.system = system

        self.error = True
        for trial in range(1):
            try:
                cameras = CameraList.create_from_system(self.system, update_cams=True, update_interfaces=True)
                self.camera = cameras.create_camera_by_serial(str(self.serial))
                self.error = False
                break
            except Exception as e:
                print_error('camerahub.py - Blackfly(): Error while initializing camera {0}, trial {2}:\n{1}'.format(self._name, e, trial), 'error')
                self.error = True

        self._props.set('init_error', self.error)
        self._props.set('init_done', True)
        print_error('camerahub.py - Blackfly(): Done with camera {0}, init error={1}'.format(self._name, self.error), 'info')

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

    def setProps(self, props):
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
        props = self.getProps()

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
            # print("Setting {} to {}".format(key, val))
            # I. we're at the lowest level, set the value in the node
            if not isinstance(val, dict):
                node = nodemap.get_node_by_name(key)
                if node is None: continue
                if node.is_writable():
                    try:
                        node.set_node_value(val)
                    except rotpy.system.SpinnakerAPIException as e:
                        print(e)
                #else:
                #    print_error('Blackfly() - bfly2.py {}: Node {} not writable.'.format(self.serial, key),
                #                'warning' if key not in self.non_writables else 'info')

            # II. It's a SpinEnumNode, set the 'value' from the dict
            elif "options" in val:
                node = nodemap.get_node_by_name(key)
                if node.is_writable():
                    enum_item = node.get_entry_by_name(val['value'])
                    node.set_node_value(enum_item)
                elif key not in self.non_writables:
                    print_error('Blackfly() - bfly2.py {}: Node {} not writable.'.format(self.serial, key),
                                'warning')

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

    def getProps(self):
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
            print_error('Blackfly - get_image(): Error getting an image from camera {0}; {1}'.format(
                self.serial, e), 'error')
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
            print_error('Blackfly - start(): Starting camera acquisition ...', 'info')
            if callback is not None:
                # print_error('Blackfly - start(): Camera started: {0}'.format(callback), 'success')
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
                    self._iterate_nodes(n, lvl=lvl+1, dic=dic[node.get_name()])

        # all available nodes that represent properties get stored with their values
        elif node.is_available():
            if type(node) is rotpy.node.SpinEnumNode:
                dic[node.get_name()] = {"value":node.get_node_value().get_enum_name(), "options": node.get_entries_names()}

            elif type(node) is not rotpy.node.SpinCommandNode:
                dic[node.get_name()] = node.get_node_value()

        return dic

class CameraGUI(BMainWindow):
    _cam_serials = {"Axial": 16292944, "RadBa": 18085363,
                    "Beamprofiler": 18085402, "Vertical": 17428576}

    def __init__(self):
        super().__init__(name='Servers/CameraHub')
        self.props = self._props
        _cam_serials_startup = {k: v for k, v in self._cam_serials.items() if k != 'Vertical'}
        self.hub = CameraHub(self.props, parent=self, cameras_for_startup=_cam_serials_startup)
        self.camWidgets = {}
        self.initUI()
        g = self.geometry()
        geo = self._props.get('Geometry', g.getRect())
        self.setGeometry(*geo)

        for cam in _cam_serials_startup.keys():
            self.camWidgets[cam].start_cam(startup=True)

    def initUI(self):
        self.mainWidget = QWidget()
        # self.mainWidget.setStyleSheet("CameraGUI {background-color: rgb(195,205,230);color:blue; margin:0px; border:5px solid rgb(0, 0, 80);} ")

        self.qlayout = QHBoxLayout()
        for cam in self._cam_serials.keys():
            camWidget = CamWidget(cam, parent=self)
            self.camWidgets[cam] = camWidget
            self.qlayout.addWidget(camWidget)
        self.mainWidget.setLayout(self.qlayout)
        self.setCentralWidget(self.mainWidget)
        # self.mainWidget.setAttribute(QtCore.Qt.WA_StyledBackground)

    def closeEvent(self, event):
        ''' on shutdown close all cameras first '''
        for cam in self._cam_serials:
            self.hub.stop_cam(cam)
        g = self.geometry()
        self._props.set('Geometry', g.getRect())
        event.accept()

class CamWidget(QWidget):
    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.hub = parent.hub
        self.name = name
        self.serial = parent._cam_serials[name]
        layout = QVBoxLayout()
        self.label = QLabel(name)
        layout.addWidget(self.label)
        self.running = False

        startButton = QPushButton("Start")
        startButton.pressed.connect(self.start_cam)
        # startButton.setStyleSheet(buttonStyleSheet)
        layout.addWidget(startButton)

        stopButton = QPushButton("Stop")
        # stopButton.setStyleSheet(buttonStyleSheet)
        stopButton.clicked.connect(self.stop_cam)
        layout.addWidget(stopButton)

        configButton = QPushButton("Config")
        # configButton.setStyleSheet(buttonStyleSheet)
        configButton.pressed.connect(self.config_cam)
        #layout.addWidget(configButton)

        self.setLayout(layout)

    # self.setStyleSheet("CamWidget {background-color: rgb(210,230,240);color:blue; margin:7px; border:7px solid rgb(220, 240, 255); } QPushButton {background-color: rgb(210,230,240);color:#000000; margin:1px; border:0px solid rgb(20, 240, 255);} ")
    # self.setStyleSheet("CamWidget {background-color: rgb(210,230,240);color:blue; margin:1px; border:2px solid rgb(220, 240, 255); } ")

    def start_cam(self, startup=False):
        print_error('CamWidget - start_cam(): Starting camera {} with serial {}.'.format(
            self.name, self.serial), 'info')
        self.label.setStyleSheet("background-color: green")
        if not startup:
            self.hub.start_cam(self.name, self.serial)
        self.running = True

    def stop_cam(self):
        print_error('CamWidget - stop_cam(): Stopping camera {} with serial {}.'.format(
            self.name, self.serial), 'info')
        self.hub.stop_cam(self.name)
        self.label.setStyleSheet("background-color: red")
        self.running = False

    def config_cam(self):
        print_error('CamWidget - config_cam(): Configuring camera {} with serial {}.'.format(
            self.name, self.serial), 'info')
        cs = self.hub.camservers[self.name]
        self.hub.config_cam(self.name)
        cs.run()

    def force_ip(self):
        print_error('camerahub.py - check_force_ip(): Forcing IP for {0}.'.format(self.name), 'warning')
        self.hub.force_ip(self.name, self.serial)


def main():
    qApp = QApplication(sys.argv)
    Win = CameraGUI()
    Win.show()
    qApp.exec_()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    #if (sys.flags.interactive != 1) or not hasattr(Qt, 'PYQT_VERSION'):
    main()
