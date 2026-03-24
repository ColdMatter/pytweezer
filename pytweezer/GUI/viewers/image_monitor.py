import argparse
import datetime
import sys
import os
import time
import logging
from PyQt5 import QtWidgets
from requests import head
import zmq
import numpy as np
from imageio import imwrite

from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QPushButton, QDialog
from PyQt5.QtCore import Qt
import PyQt5
import pyqtgraph as pg
from pyqtgraph.dockarea import DockArea, Dock
import matplotlib.image as mpli
import matplotlib.pyplot as plt

# import pyximport; pyximport.install(setup_args={"script_args":["--compiler=unix"],"include_dirs":np.get_include()}, reload_support=True)
# from gaussfit import gaussian_fit
# from pytweezer import servers
from pytweezer.servers import ImageClient, Properties, tweezerpath, PropertyAttribute
from pytweezer.GUI.viewers.zmq_ROI import zmq_ROI
from pytweezer.GUI.property_editor import PropEdit
from pytweezer.GUI.subscription_editor import SubscriptionEditor
import imageio
import cProfile


class ImageDisplay(QWidget):
    """self updating Image display Window

    Args:
        imagestreams (list of strings): names of the image streams the wiever should display
        parent (QWidget-like):      see Qt documentation

    Returns:
        None
    """

    _maxlevels = PropertyAttribute("maxlevels", 255.0)
    _autolevels = PropertyAttribute("autolevels", False)
    _imagestreams = PropertyAttribute("imagestreams", [""])
    _imgpath = PropertyAttribute("/Servers/Imagepath", tweezerpath + "/../Data")
    _roilist = PropertyAttribute("ROIs", ["slice"])
    _imagestreams = PropertyAttribute("imagestreams", ["--None--"])
    _maskstreams = PropertyAttribute("maskstreams", ["--None--"])
    _invert_colors = PropertyAttribute("invert_colors", False)

    def __init__(self, name, parent=None):
        self._props = Properties(name)
        self.props = self._props
        self.name = name
        self.imDataDict = {}
        self.imgresolution = [1, 1]
        self.parent = parent
        super().__init__(parent)
        btn = QPushButton("SaveImage")
        layout = QVBoxLayout()
        layout.addWidget(btn)
        # btn.clicked.connect(self.saveCurrentImage)

        self.imgstream = ImageClient("imageviewer")
        self.maskstream = ImageClient("imageviewermask")
        for stream in self._imagestreams:
            self.imgstream.subscribe(stream)
        for stream in self._maskstreams:
            self.maskstream.subscribe(stream)
        print("subscribed")
        if self.imgstream.has_new_data():
            imgdata, head = self.imgstream.recv()
            print("recieved")
        else:
            imgdata = np.ones((100, 100))

        colors = np.array(plt.cm.magma.colors) * 255
        print(self.name)
        if (
            self._invert_colors
            or "Radial_Li_abs" in self.name
            or "Vertical_Li_abs" in self.name
        ):
            colors = colors[::-1]

        cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), color=colors)
        win = pg.GraphicsLayoutWidget()
        plot = win.addPlot()
        plot.setLabel("left", text="y", units="m")
        plot.setLabel("bottom", text="x", units="m")
        ii = pg.ImageItem(imgdata)
        self.image = ii
        ii.setRect(PyQt5.QtCore.QRect(0, 0, imgdata.shape[1], imgdata.shape[0]))
        ii.setLookupTable(cmap.getLookupTable())
        print(cmap.getLookupTable())
        plot.addItem(ii)
        # add Image mask
        maskdat = np.zeros(imgdata.shape)
        # maskdat[20:25]=1
        mask = pg.ImageItem(maskdat)
        mask.setLookupTable(np.array([[0, 0, 255, 0], [0, 255, 255, 255]]))
        plot.addItem(mask)
        self.mask = mask
        for region in self._roilist:
            roi = zmq_ROI(self.name + "/ROI/" + region, invertible=True)
            roi.addScaleHandle((1, 1), center=(0, 0))
            plot.addItem(roi)
        # \plot.hideAxis('left')
        # plot.hideAxis('bottom')
        layout.addWidget(win)

        self.setLayout(layout)
        # self.show()
        vb = plot.getViewBox()
        # we want to keep the aspect ratio correct when plotting images
        vb.setAspectLocked(True)
        subscribe_menu = vb.menu.addAction("subscriptions")
        subscribe_menu.triggered.connect(self.subscribe_window)
        subscribe_menu = vb.menu.addAction("Image masks")
        subscribe_menu.triggered.connect(self.subscribe_mask)
        subscribe_menu = vb.menu.addAction("configure")
        subscribe_menu.triggered.connect(self.configureWindow)

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_image)
        timer.start(10)
        self.timer = timer

    def sizeHint(self):
        return QtCore.QSize(800, 700)

    def update_image(self):
        if self.imgstream.has_new_data():
            print("new data")
            while self.imgstream.has_new_data():
                msg, head, imgdata = self.imgstream.recv()
                self.image.setImage(
                    imgdata, autoLevels=False, levels=(0, self._maxlevels)
                )
                # self.image.setImage(imgdata, autoLevels=True, levels=(0, self._maxlevels))
                self.image.resetTransform()
                try:
                    self.imgresolution = head["_imgresolution"]
                    offset = np.array(head["_offset"]) * np.array(self.imgresolution)
                except Exception:
                    self.imgresolution = [1, 1]
                    offset = [0, 0]

                h, w = imgdata.shape
                self.image.setRect(
                    QtCore.QRectF(
                        offset[0],
                        offset[1],
                        w * self.imgresolution[0],
                        h * self.imgresolution[1],
                    )
                )
                # print(imgdata)
                self.imgdata = imgdata
                img_flat = imgdata.flatten()
                imMin = imgdata.min()
                self.imDataDict["imMin"] = imMin
                imMax = imgdata.max()
                self.imDataDict["imMax"] = imMax
                imMean = imgdata.mean()
                self.imDataDict["imMean"] = imMean
                nDead = len(np.where(img_flat > 0.9 * self._maxlevels)[0])
                self.imDataDict["nDead"] = nDead
                self.parent.imDataBox.setNewData(self.imDataDict)
        self.update_mask()

    def update_mask(self):
        if self.maskstream.has_new_data():
            while self.maskstream.has_new_data():
                msg, head, imgdata = self.maskstream.recv()
                self.imgresolution = head["_imgresolution"]
                offset = tuple(np.array(head["_offset"]) * np.array(self.imgresolution))

                h, w = imgdata.shape
                self.mask.setRect(
                    QtCore.QRectF(
                        offset[0],
                        offset[1],
                        w * self.imgresolution[0],
                        h * self.imgresolution[1],
                    )
                )
                self.mask.setImage(imgdata, autoLevels=False, levels=(0, 1))
                self.mask.setImage(imgdata, autoLevels=False, levels=(0, 1))

    def saveCurrentImage(self):
        """Save the current image as png using current time as filename"""
        # img=self.imgdata.T[::-1]
        # img=img+img.min()
        # img=img*2**16/img.max()
        # imageio.imwrite(self._imgpath+'/image%i.png'%time.time(),img.astype('uint16'))
        # path = self._imgpath+'/image%i.png'%time.time()
        # print('Viewer saving image',self._imgpath+'/'+path)
        # img = self.imgdata
        # imwrite(path, img)
        pass

    def subscribe_window(self):
        self._subscribe_win("imagestreams")
        self.imgstream.unsubscribe()
        self.imgstream.subscribe(self._imagestreams)

    def subscribe_mask(self):
        self._subscribe_win("maskstreams")
        self.maskstream.unsubscribe()
        self.maskstream.subscribe(self._maskstreams)

    def _subscribe_win(self, key):
        d = QDialog()
        layout = QVBoxLayout()
        d.setWindowTitle("Dialog")
        editor = SubscriptionEditor(self._props, "Image", streamkey=key)
        layout.addWidget(editor)
        d.setLayout(layout)
        # d.setWindowModality(QtGui.ApplicationModal)
        d.exec_()

    def configureWindow(self):
        d = QDialog()
        d.setWindowTitle("Dialog")
        layout = QVBoxLayout()
        editor = PropEdit("/" + self.name + "/")
        layout.addWidget(editor)
        d.setLayout(layout)
        d.exec_()


def main(name):
    logging.info("Started.")
    app = QtWidgets.QApplication(sys.argv)
    camWin = ImageDisplay(name)
    camWin.show()
    app.exec_()


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == "__main__":
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        parser = argparse.ArgumentParser()
        parser.add_argument("name", nargs=1, help="name of this program instance")
        args = parser.parse_args()
        name = args.name[0]
        main(name)
