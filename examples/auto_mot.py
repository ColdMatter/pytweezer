from typing import Any, Tuple, Union
import time
import os
import gc
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from .experiment import Experiment


sns.set()


def _get_image_from_file(
    filepath: str
) -> np.ndarray:
    # BUG FIX: Python was not realeasing the file resource for deletion
    with Image.open(filepath, 'r') as imagefile:
        image = np.array(imagefile, dtype=float)
    return image


class MOTAutoTuner():
    def __init__(
        self,
        experiment: Experiment,
        script: str = "AMOTBasic",
        imgs_dirpath: str = "C:\\Users\\cafmot\\Documents\\TempCameraImages",
        field_parameter: str = "MOTCoilsCurrentValue",
        bg_field_value: float = 0.0,
        mot_field_value: float = 1.0
    ) -> None:
        self.expt = experiment
        self.script = script
        self.imgs_dirpath = imgs_dirpath
        self.field_parameter = field_parameter
        self.bg_field_value = bg_field_value
        self.mot_field_value = mot_field_value

    def __setattr__(
        self,
        __name: str,
        __value: Any
    ) -> None:
        self.__dict__[__name] = __value

    def _clear_imgs_dir(
        self
    ) -> None:
        for filename in os.listdir(self.imgs_dirpath):
            if '.tif' in filename:
                os.remove(os.path.join(self.imgs_dirpath, filename))
        return None

    def __call__(
        self,
        display_results=True
    ) -> None:
        self._clear_imgs_dir()
        results = self.expt.scan_motmaster_parameter(
            self.script,
            self.field_parameter,
            [self.bg_field_value],
            False,
            self.fl_number_detector
        )
        images_bg = np.mean(results[0][1], axis=0)
        time.sleep(1)
        for laser in self.expt.config["lasers"]:
            if hasattr(self, f"{laser}_isscan"):
                if getattr(self, f"{laser}_isscan"):
                    results = self.expt.scan_tcl_laser_set_points(
                        self.script,
                        laser,
                        getattr(self, f"{laser}_scan_range"),
                        False,
                        self.fl_number_detector,
                        self.field_parameter,
                        self.mot_field_value
                    )
                    n, ne = np.array([]), np.array([])
                    for item in results:
                        images_mot = item[1] - images_bg
                        _n = np.sum(images_mot, axis=(1, 2))
                        n = np.append(n, np.mean(_n))
                        ne = np.append(ne, np.std(_n)/np.sqrt(len(_n)))
                    _scan_range = getattr(self, f"{laser}_scan_range")
                    set_point = _scan_range[np.argmax(n)]
                    setattr(self, f"{laser}_setpoint", set_point)
                    setattr(self, f"{laser}_numbers_mean", n)
                    setattr(self, f"{laser}_numbers_error", ne)
                    results = self.expt.scan_tcl_laser_set_points(
                        self.script,
                        laser,
                        [set_point],
                        False,
                        self.fl_number_detector,
                        self.field_parameter,
                        self.mot_field_value
                    )
            else:
                print(
                    f"{laser} not found in MOTAutoTuner Object"
                )
        if display_results:
            self.display_results()
        return self

    def fl_number_detector(
        self,
        value: Union[int, float]
    ) -> Tuple[float, np.ndarray]:
        images, filepaths = [], []
        for filename in os.listdir(self.imgs_dirpath):
            if '.tif' in filename:
                filepaths.append(os.path.join(self.imgs_dirpath, filename))
        for filepath in filepaths:
            image = _get_image_from_file(filepath)
            images.append(image)
        gc.collect()
        self._clear_imgs_dir()
        return (
            value,
            np.array(images)
        )

    def display_results(
        self
    ) -> None:
        fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
        fig.subplots_adjust(
            left=0.125,
            bottom=0.1,
            right=0.9,
            top=0.9,
            wspace=0.05,
            hspace=0.35
        )
        named_ax = \
            {
                "v00Lock": ax[0, 0],
                "bXLock": ax[0, 1],
                "v10Lock": ax[0, 2],
                "v21Lock": ax[1, 0],
                "v32Lock": ax[1, 1],
                "bXBeastLock": ax[1, 2]
            }
        named_ax["v00Lock"].set_ylabel("Norm. Number")
        named_ax["v21Lock"].set_ylabel("Norm. Number")
        for laser in self.expt.config["lasers"]:
            if hasattr(self, f"{laser}_isscan"):
                if getattr(self, f"{laser}_isscan"):
                    f = getattr(self, f"{laser}_scan_range")
                    n = getattr(self, f"{laser}_numbers_mean")
                    ne = getattr(self, f"{laser}_numbers_error")
                    named_ax[laser].errorbar(
                        f,
                        n/np.max(n),
                        yerr=ne/np.max(n),
                        fmt='ok'
                    )
                    named_ax[laser].arrow(
                        getattr(self, f"{laser}_setpoint"),
                        1.15, 0, -0.1,
                        head_width=0.002,
                        head_length=0.03,
                        width=0.0005,
                        fc='r', ec='r')
                    named_ax[laser].set_title(
                        f"{laser} set @ {getattr(self, f'{laser}_setpoint')} V"
                    )
            else:
                named_ax[laser].set_title(f"{laser} not scanned")
            named_ax[laser].set_xlabel("TCL Volatge [V]")
            named_ax[laser].set_ylim((0, 1.2))
        return None
