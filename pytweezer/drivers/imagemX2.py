import pylablib.devices.DCAM as dcam
import tifffile as tiff

IMAGE_DIRECTORY  = "C:/Users/CaFMOT/OneDrive - Imperial College London/caftweezers/HamCamImages"

class ImagEMX2Camera():
    """Class to control the Hamamatsu ImagEM X2 camera via DCAM API.

    
    """

    def __init__(self, image_dir=None):
        try:
            self.cam = dcam.DCAMCamera()
        except Exception as e:
            raise RuntimeError("Could not connect to ImagEM X2 camera") from e
        try:
            self.cam.open()
            print("Connected to ImagEM X2 camera.")
        except Exception as e:
            raise RuntimeError("Could not open connection to ImagEM X2 camera") from e
        
        self.image_dir = image_dir or "C:\\Users\\CaFMOT\\OneDrive - Imperial College London\\caftweezers\\HamCamImages\\"
       

    def set_roi(self, x0, width, y0, height):
        """Set the region of interest (ROI) for image acquisition.

        Parameters:
            x0 (int): Starting x-coordinate of the ROI.
            width (int): Width of the ROI.
            y0 (int): Starting y-coordinate of the ROI.
            height (int): Height of the ROI.
        """
        self.cam.set_roi(x0, x0 + width - 1, y0, y0 + height - 1)

    def set_ccd_mode(self, mode):
        """Set the CCD mode.

        Parameters:
            mode (int): 2 for EM gain mode, 1 for normal mode.
        """
        if mode:
            self.cam.set_attribute_value("ccd_mode", 2)
        else:
            self.cam.set_attribute_value("ccd_mode", 1)
    
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
        self.cam.set_attribute_value("direct_em_gain_mode", mode)

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
        self.cam.set_attribute_value("sensitivity", sensitivity)

    def set_trigger_source(self, source: str):
        """Set the trigger source.

        Parameters:
            source (str): Trigger source, options are "software", "external", "internal".
        """
        self.cam.set_trigger_mode(source)

    def set_exposure_time(self, exposure):
        """Set the exposure time.

        Parameters:
            exposure_ms (float): Exposure time in as yet unknown units.
        """
        self.cam.set_exposure(exposure)

    def set_external_exposure_mode(self):
        """Set the camera to external exposure mode. Also known as "level trigger mode".
        """
        self.set_trigger_source("ext")
        self.cam.set_attribute_value("trigger_active", 2)  # level
        self.cam.setup_ext_trigger(invert=True)

    def setup_acquisition(self, acq_mode, nframes):
        """Setup image acquisition.

        Parameters:
            acq_mode (str): Acquisition mode. "snap" or "sequence".
            nframes (int): Number of frames to acquire in snap mode, or buffer size in sequence mode.
        """
        self.cam.setup_acquisition(acq_mode, nframes)

    def start_acquisition(self):
        """Start image acquisition.
        """
        self.cam.start_acquisition()

    def stop_acquisition(self):
        """Stop image acquisition.
        """
        self.cam.stop_acquisition()

    def acquire_n_frames(self, nframes, start_frame=0, autosave=False, run_no=0):
        """Acquire a specified number of frames.

        Parameters:
            nframes (int): Number of frames to acquire.
        """
        self.cam.wait_for_frame(nframes=nframes)
        imgs = self.cam.read_multiple_images((start_frame, start_frame + nframes))
        if autosave:
            for img in imgs:
                self.save_tiff(img, IMAGE_DIRECTORY, run_no=run_no)
        return imgs
    
    def clear_buffer(self):
        """Clear the camera buffer.
        """
        self.cam.read_multiple_images()

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
        while os.path.exists(image_dir+"/HamTweezer%s_%s.tif" % (f"{run_no:04d}", i)):
            i += 1

        run_no_str = f"{run_no:04d}"
        filename = os.path.join(image_dir, "HamTweezer%s_%s.tif" % (run_no_str, i))
        tiff.imwrite(filename, image)