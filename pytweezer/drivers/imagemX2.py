import pylablib.devices.DCAM as dcam
import tifffile as tiff

from pytweezer.servers import ImageClient
from pytweezer.servers import CommandClient, DataClient
from pytweezer.servers import Properties, PropertyAttribute
import time
import copy
import argparse
import numpy as np
import logging, sys

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)



IMAGE_DIRECTORY = (
    "C:/Users/CaFMOT/OneDrive - Imperial College London/caftweezers/HamCamImages"
)


class ImagEMX2Camera:
    """Class to control the Hamamatsu ImagEM X2 camera via DCAM API."""

    def __init__(self, name, image_dir=None, timeout=5):
        try:
            self.dcam = dcam.DCAMCamera()
        except Exception as e:
            raise RuntimeError("Could not connect to ImagEM X2 camera") from e
        try:
            self.dcam.open()
            print("Connected to ImagEM X2 camera.")
        except Exception as e:
            raise RuntimeError("Could not open connection to ImagEM X2 camera") from e

        self.image_dir = (
            image_dir
            or "C:\\Users\\CaFMOT\\OneDrive - Imperial College London\\caftweezers\\HamCamImages\\"
        )
        self.timeout = timeout
        self._connect_clients()
        
    def _connect_clients(self):
        self.imstream = ImageClient('dummy_cam')
        self.cmdstream = CommandClient('dummy_cam')
        self.cmdstream.subscribe('dummy_cam')

    def set_roi(self, x0, width, y0, height):
        """Set the region of interest (ROI) for image acquisition.

        Parameters:
            x0 (int): Starting x-coordinate of the ROI.
            width (int): Width of the ROI.
            y0 (int): Starting y-coordinate of the ROI.
            height (int): Height of the ROI.
        """
        self.dcam.set_roi(x0, x0 + width - 1, y0, y0 + height - 1)

    def set_ccd_mode(self, mode):
        """Set the CCD mode.

        Parameters:
            mode (int): 2 for EM gain mode, 1 for normal mode.
        """
        if mode:
            self.dcam.set_attribute_value("ccd_mode", 2)
        else:
            self.dcam.set_attribute_value("ccd_mode", 1)

    def enable_em_gain(self, enable=True):
        """Enable or disable EM gain.

        Parameters:
            enable (bool): True to enable EM gain, False to disable.
        """
        if enable:
            self.set_ccd_mode(2)
        else:
            self.set_ccd_mode(1)

    def set_direct_em_gain_mode(self, mode):
        """Set the direct EM gain mode.

        Parameters:
            mode (int): 1 for relative, 2 for absolute.
        """
        self.dcam.set_attribute_value("direct_em_gain_mode", mode)

    def enable_direct_em_gain(self, enable=True):
        """Enable or disable direct EM gain.

        Parameters:
            enable (bool): True to enable direct EM gain, False to disable.
        """
        if enable:
            self.set_direct_em_gain_mode(2)
        else:
            self.set_direct_em_gain_mode(1)

    def set_sensitivity(self, sensitivity):
        """Set the camera sensitivity (EM gain).

        Parameters:
            sensitivity (int): Sensitivity value (1-16).
        """
        self.dcam.set_attribute_value("sensitivity", sensitivity)

    def set_trigger_source(self, source: str):
        """Set the trigger source.

        Parameters:
            source (str): Trigger source, options are "software", "external", "internal".
        """
        self.dcam.set_trigger_mode(source)

    def set_exposure_time(self, exposure):
        """Set the exposure time.

        Parameters:
            exposure_ms (float): Exposure time in as yet unknown units.
        """
        self.dcam.set_exposure(exposure)

    def set_external_exposure_mode(self):
        """Set the camera to external exposure mode. Also known as "level trigger mode"."""
        self.set_trigger_source("ext")
        self.dcam.set_attribute_value("trigger_active", 2)  # level
        self.dcam.setup_ext_trigger(invert=True)

    def setup_acquisition(self, acq_mode, nframes):
        """Setup image acquisition.

        Parameters:
            acq_mode (str): Acquisition mode. "snap" or "sequence".
            nframes (int): Number of frames to acquire in snap mode, or buffer size in sequence mode.
        """
        self.dcam.setup_acquisition(acq_mode, nframes)

    def start_acquisition(self):
        """Start image acquisition."""
        self.dcam.start_acquisition()

    def stop_acquisition(self):
        """Stop image acquisition."""
        self.dcam.stop_acquisition()

    def acquire_n_frames(self, nframes, exp_info=None, start_frame=0, autosave=False):
        """Acquire a specified number of frames.

        Parameters:
            nframes (int): Number of frames to acquire.
        """
        self.dcam.wait_for_frame(nframes=nframes, timeout=self.timeout)
        imgs, infos = self.dcam.read_multiple_images((start_frame, start_frame + nframes), return_info=True)
        imgs: np.ndarray
        infos: list[dcam.DCAM.TFrameInfo]
        for i, img in enumerate(imgs):
            if exp_info is not None:
                self.broadcast_image(img, task=self.exp_info["task"], run=self.exp_info["run"], rep=self.exp_info["rep"], index=i, timestamp=time.time())
            if autosave:
                self.save_tiff(img, IMAGE_DIRECTORY, run_no=self.exp_info["run"])
            
        return imgs
    
    def broadcast_image(self, im, task, run, rep, index, timestamp):
    
        info = {
            "timestamp": timestamp,
            "task": task,
            "run": run,
            "rep": rep,
            "index": index,
            "_imageresolution" : [1,1],
            "_offset": [0,0]
        }  # TODO all cam settings

        self.imstream.send(im, info)
        # print('camera time stamp:',image_stamp)


    def clear_buffer(self):
        """Clear the camera buffer."""
        self.dcam.read_multiple_images()

    @staticmethod
    def save_tiff(image, image_dir=None, run_no=0):
        import os
        import tifffile as tiff

        """Save image to a TIFF file.

        Parameters:
            filename (str): Path to the output TIFF file.
            images (ndarray): Array of images to save.
        """
        i = 1
        while os.path.exists(image_dir + "/HamTweezer%s_%s.tif" % (f"{run_no:04d}", i)):
            i += 1

        run_no_str = f"{run_no:04d}"
        filename = os.path.join(image_dir, "HamTweezer%s_%s.tif" % (run_no_str, i))
        tiff.imwrite(filename, image)




class Camserver:
    """Controls the camera and handles the communication with the Image and Data bus."""

    # _task       =PropertyAttribute('/Experiments/_task',0)                #task number (a task contains multiple repetitions of scans)
    # _repetition =PropertyAttribute('/Experiments/_repetition',0)    #repetition index (a scan can be repeated multiple times)
    # _run        =PropertyAttribute('/Experiments/_run',0)                  #run number in a scan (a scan has multiple experiment runs)
    _resolution = PropertyAttribute(
        "resolution [m per pixel]", (0.00001, 0.00001)
    )  # run number in a scan (a scan has multiple experiment runs)
    _conf_name = PropertyAttribute("Configuration_name", "DefaultConfig")

    _autoreconfigure = PropertyAttribute("Autroreconfigure", True)

    def __init__(self, name, serial):
        """
        initializes the driver.  Set the properties to the camera. If configuration does not exist,
        create new configuration with current values from camera

        Args:
            name (str): Name of the camera. Will be used to identify the properties and to create a named
                        imagestream
            serial    : Serial number of the camera
        """
        # global defaultproperties
 
        self.imstream = ImageClient(name)
        self.cmdstream = CommandClient(name)
        self.cmdstream.subscribe(name)
        self.name = "Cameras/" + name
        self._props = Properties(self.name)
        self.connected = False

        self.lastinfo = (0, 0, 0)
        self.imgindex = 0  # number of image in the current experiment

        self.cam = ImagEMX2Camera(serial)
        props = self._props
        self.running = False
        self.connected = True

        self.indexq = DataClient(self.name)
        self.indexq.subscribe(["Experiment.start"])
        self.experiment_timings = [
            [0, {"_run": 0, "_repetition": 0, "_task": 0}]
        ]  # list containing timings of the experiment

        self.configure()

    def configure(self):
        with self.cam:
            while True:
                self.cam.start()
                changes = self._props.changes()
                print(changes)
                # if any(s.startswith(self.confname) for s in changes): #some subentry has changed
                if "/" + self.name in changes:  # some subentry has changed
                    print(self.name, "reconfiguring")
                    cam_props = self._props.get(self.confname, {})
                    self.cam.stop()
                    if cam_props == {}:
                        cam_props = self.cam.getProps()
                        self._props.set(self._conf_name, cam_props)
                        self._cam_props = cam_props
                    else:
                        newprops = self.cam.setProps(cam_props)
                        self._props.set(
                            self.confname, newprops
                        )  # update changed properties
                        self._cam_props = newprops
                    self._props.changes()  # clear changes
                else:
                    self.cam.stop()
                    break

    def update_experiment_indices(self):
        """keep list of experiment starts updatesd"""

        while self.indexq.has_new_data():
            # print('datamgr.py data incomming!')
            recvmsg = self.indexq.recv()
            A = None
            if len(recvmsg) == 2:
                msg, msg_dic = recvmsg
            elif len(recvmsg) == 3:
                msg, msg_dic, A = recvmsg
            else:
                print("datamgr message error")
                return
            # print(msg)
            if msg == "Experiment.start":
                indices = dict(
                    (k, msg_dic[k])
                    for k in ("_run", "_repetition", "_task")
                    if k in msg_dic
                )
                timestamp = msg_dic["_starttime"]
                self.experiment_timings.append([timestamp, indices])
                while len(self.experiment_timings) > 20:
                    del self.experiment_timings[0]

    def run(self):
        """runloop running forever
        Sends images from the camera to the imagestream and initiates reconfiguration of the camera in case
        properties have changed.
        """
        if not self.connected:
            return
        with self.cam:
            self.ii = 0
            self.cam.start()
            self.running = True
            while self.running == True:  # poll camera and commands
                self.sendimage(
                    self.cam.get_image()
                )  # if no image is available waits Config.grabTimeout milliseconds
                if self._autoreconfigure:
                    changes = self._props.changes()
                    # print(changes)
                    # if any(s.startswith(self.confname) for s in changes): #some subentry has changed
                    if "/" + self.name in changes:  # some subentry has changed
                        print(self.name, "reconfiguring")
                        cam_props = self._props.get(self.confname, {})
                        self.cam.stop()
                        if cam_props == {}:
                            cam_props = self.cam.getProps()
                            self._props.set(self._conf_name, cam_props)
                            self._cam_props = cam_props
                        else:
                            newprops = self.cam.setProps(cam_props)
                            self._props.set(
                                self.confname, newprops
                            )  # update changed properties
                            self._cam_props = newprops
                        self.cam.start()
                        self._props.changes()  # clear changes
                self.ii += 1
                # print(self.ii)
                # if self.ii>300 : self.running=False

                if self.cmdstream.has_new_data():
                    cmd = self.cmdstream.recv()
            self.cam.stop()
        return 0

    def _image_to_array(self, im):
        """Convert blackfly image into numpy array

        Args:
            im: Image from blackfly driver

        Returns:
            numpy.array(float64).    The image as a numpy array
        """
        # ii=im.getData()
        # im2=im.convert(pc2.PIXEL_FORMAT.RAW16)
        imarr = np.array(im.getData())  ## This is much too slow (try struct.pack)

        rows = im.getRows()
        cols = im.getCols()
        if imarr.shape[0] == rows * cols * 2:
            imarr2 = imarr[::2] / 16 + 16 * imarr[1::2]
        else:
            imarr2 = imarr
        imarr2 = np.reshape(imarr2, (rows, cols))
        imarr2 = imarr2.T
        # imarr=np.rot90(imarr)
        # imarr = imarr[::,::-1]
        imarr2 = imarr2.astype(np.float64)
        return imarr2.copy()

    def generateImageIndex(self, _task=0, _repetition=0, _run=0):
        """
        in the current measurement count the images. If the next run has started (either task,repetition or run changes)
        reset the counter(imgindex).
        """
        measIndex = (_task, _repetition, _run)
        if self.lastinfo == measIndex:
            self.imgindex += 1
        else:
            self.imgindex = 0
            self.lastinfo = measIndex

    def sendimage(self, im):
        """send blackfly camera image via the imagestream
        The image is converted to numpy array, additional information like binning,
        timestamp, ... is added to a dictionary and both are send to the imagestream


        Args:
            im: Blackfly image object

        Returns:
            None

        """
        # print('sending image')
        if im == None:
            return
        if self.confname != "DefaultConfig":
            print("{} sending image.".format(self.name))
        A = self._image_to_array(im)

        timestamp_dict = im.getTimeStamp().__dict__
        image_stamp = (
            float(timestamp_dict["seconds"]) + timestamp_dict["microSeconds"] * 1e-6
        )
        if time.time() - image_stamp > 5:
            print(
                "Warning: bfly.py ",
                self.name,
                " image readout delayed by{:.0f}s".format(time.time() - image_stamp),
            )

        run_task = self.find_experiment_indices(image_stamp)
        self.generateImageIndex(**run_task)

        binning = self._cam_props["GigEBinningSettings"][0]

        info = {
            "timestamp": image_stamp,
            "_imgresolution": [x * binning for x in self._resolution],
            "_imgindex": self.imgindex,
            "camgain": -1,
        }  # TODO CAMgain   self._cam_props[' }
        info.update(run_task)
        imgsettings = self._cam_props["GigEImageSettings"]
        # print('Pixelformat{0:b}'.format(imgsettings['pixelFormat']))
        # print(pc2.PIXEL_FORMAT.__dict__)
        info["_offset"] = [imgsettings["offsetX"], imgsettings["offsetY"]]
        self.imstream.send(A, info)
        # print('camera time stamp:',image_stamp)

    def find_experiment_indices(self, timestamp):
        """find the run,task and repetition for a given timestamp"""
        self.update_experiment_indices()
        tstamps = [i[0] for i in self.experiment_timings]
        tstamps = np.array(tstamps)

        index = 0
        if timestamp > tstamps[0]:
            index = np.argwhere(tstamps - timestamp < 0)[-1][0]
        return self.experiment_timings[int(index)][1]


def run(name):
    """initialize a Camserver and run it"""
    cs = Camserver(name)
    cs.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs=1, help="name of this program instance")
    args = parser.parse_args()
    name = args.name[0]
    run(name)
