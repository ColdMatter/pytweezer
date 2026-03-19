import pyximport; pyximport.install()

from pytweezer.analysis.src.twoDgaussian import twoDgaussfit
import numpy as np
from astropy.modeling import models as apm
from astropy.modeling import fitting as apf

from pytweezer.servers import DataClient,ImageClient
from pytweezer.servers import *

import argparse

import matplotlib.pyplot as plt

class TwoDGaussFit():
    _imagestreams = PropertyAttribute('imagestreams',['None'])
    _roi_name = PropertyAttribute('Region_of_Interest','/ROI/name')

    def __init__(self,name):
        self.name = name
        self._props = Properties(name)

        self.dataq = DataClient(name.split('/')[-1])
        self.imageq = ImageClient(name.split('/')[-1])
        self.imageq.subscribe(self._imagestreams)
        print('2D_gauss_fit.py subscriptions: ',self._imagestreams)

    def run(self):
        prop = self._props
        while True:
            # try to get an image from the imagestream
            msg = self.imageq.recv()
            if msg!= None:
                msgstr, head, img = msg
                try:
                    scale = head['_imgresolution']
                    offs = head['_offset']
                except:
                    scale = [1,1]
                    head['_imgresolution'] = scale
                    offs = [0,0]
                    head['_offset'] = offs

                # get position and size of the ROI
                roi_position = prop.get(self._roi_name + '/pos', [0,0])
                roi_size = prop.get(self._roi_name + '/size', [img.shape])

                param_names, param_vals, contour = twoDgaussfit(img, roi_size, roi_position, scale, offs) 
                for i, key in enumerate(param_names):
                    head[key] = param_vals[i]
                self.dataq.send(head,np.array([0,0]), name.split('/')[-1],prefix=msgstr[:-1])

                # send the contour out
                self.imageq.send(contour, head)

def main_run(name):
    twodgauss = TwoDGaussFit(name)
    twodgauss.run()    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs=1, help='name of this program instance')
    args = parser.parse_args()
    name=args.name[0]
    main_run(name)

