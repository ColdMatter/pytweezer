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

                # calculate the indices of the lower left edge of the ROI
                pos_idx = np.array([int(i/scale[n]-offs[n]) for n,i  in enumerate(roi_position)])

                # calculate the indices of the upper right edge of the ROI
                size_idx = np.array([int(i/scale[n]) for n,i in enumerate(roi_size)])
                pos2_idx = pos_idx + size_idx

                # cut the image to the ROI for fitting
                img_cropped = img[pos_idx[0]:pos2_idx[0], pos_idx[1]:pos2_idx[1]]
                
                # calculate x and y axis of the new image
                x_axis = (np.arange(pos_idx[0], pos2_idx[0]))*scale[0]
                y_axis = (np.arange(pos_idx[1], pos2_idx[1]))*scale[1]

                # guess initial parameters
                # amplitude
                A0 = np.max(img_cropped) - np.min(img_cropped)
                # offset
                c0 = np.min(img_cropped)
                # position
                max_idx = np.unravel_index(np.argmax(img_cropped), img_cropped.shape)
                x0 = x_axis[max_idx[0]]
                y0 = y_axis[max_idx[1]]
                # sigma, adjust ROI such, that it is around 4 sigma
                sigmax0 = (x_axis[-1] - x_axis[0])/4
                sigmay0 = (y_axis[-1] - y_axis[0])/4

                print('A0 => ', A0, 'c0 => ', c0, 'x0 => ', x0, 'y0 => ', y0) 
                
                # create a 2D gaussian + offset fit model
                # put constraints on the fitting parameters
                gauss_bounds = {'amplitude':[0, np.inf], 'x_mean':[x_axis[0], x_axis[-1]], 'y_mean':[y_axis[0], y_axis[-1]]}
                fitmodel = apm.Gaussian2D(amplitude=A0, x_mean=x0, y_mean=y0, bounds=gauss_bounds,
                        x_stddev=sigmax0, y_stddev=sigmay0) + apm.Const2D(amplitude=c0)


                # Fit the model using levenberg marquardt
                # the fit model requires x and y to be shaped such that the outcome of
                # f(x[i,j], y[i,j]) = img[i,j], so we need to shape x_fit and y_fit accordingly
                fitter = apf.LevMarLSQFitter()
                x_fit, y_fit = np.ones_like(img_cropped), np.ones_like(img_cropped).T
                x_fit = (x_fit.T * x_axis).T # use meshgrid instead?
                y_fit = y_fit.T * y_axis
                popt = fitter(fitmodel, x_fit, y_fit, img_cropped)
                
                ## evaluate the model on the original image
                # create x and y values for evaluation
                x = np.arange(0, img.shape[0])*scale[0]
                y = np.arange(0, img.shape[1])*scale[1]
                x, y = np.meshgrid(x,y)

                # evaluate the function and normalize it in order to calculate 1/e and 1e2 contours
                # amplitude_0 and _1 are amplitude of the gaussian and the offset function respectively
                img_result_normalized = (popt(x, y) - popt.amplitude_1) / popt.amplitude_0

                # calculate the contours, maybe the tolerance can be fine tuned
                tolerance = 1e-2
                contour1 = np.where(np.abs(img_result_normalized-1/np.e) < tolerance, 1, 0)
                contour2 = np.where(np.abs(img_result_normalized-1/np.e**2) < tolerance/2, 1, 0)
                contour = contour1 + contour2
                contour = contour.T # image format of pytweezer is transposed

                # send the fit results out via dataq
                param_names = popt.param_names
                param_vals = popt.parameters
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
