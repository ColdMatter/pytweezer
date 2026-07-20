import cython
import numpy as np
cimport numpy as np

from astropy.modeling import models as apm
from astropy.modeling import fitting as apf

#DTYPE = np.int
#DTYPEf = np.float64
#ctypedef np.int_t DTYPE_y
#ctypedef np.float64_t DTYPEf_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def twoDgaussfit(np.ndarray img, list roi_size, list roi_position, list scale, list offs):
    cdef np.ndarray ipos_idx, pos2_idx, size_idx, mg_cropped, x_axis, y_axis, x, y, x_fit, y_fit, img_result_normalized, contour, contour1, contour2, param_vals
    cdef np.float64_t A0, c0, x0, y0, sigmax0, sigmay0, tolerance
    cdef dict gauss_bounds
    cdef tuple param_names
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
    cdef tuple cropped_shape = img_cropped.shape # this is still a tuple!! [0] is due to cython!
    #stackoverflow 51737512
    max_idx = np.unravel_index(np.argmax(img_cropped), cropped_shape)
    x0 = x_axis[max_idx[0]]
    y0 = y_axis[max_idx[1]]
    # sigma, adjust ROI such, that it is around 4 sigma
    sigmax0 = (x_axis[len(x_axis)-1] - x_axis[0])/4
    sigmay0 = (y_axis[len(y_axis)-1] - y_axis[0])/4

    # create a 2D gaussian + offset fit model
    # put constraints on the fitting parameters
    gauss_bounds = {'amplitude':[0, np.inf], 'x_mean':[x_axis[0], x_axis[len(x_axis)-1]], 'y_mean':[y_axis[0], y_axis[len(y_axis)-1]]}
    fitmodel = apm.Gaussian2D(amplitude=A0, x_mean=x0, y_mean=y0, bounds=gauss_bounds,
            x_stddev=sigmax0, y_stddev=sigmay0) + apm.Const2D(amplitude=c0)

    # Fit the model using levenberg marquardt
    # the fit model requires x and y to be shaped such that the outcome of
    # f(x[i,j], y[i,j]) = img[i,j], so we need to shape x_fit and y_fit accordingly
    fitter = apf.LevMarLSQFitter()
    x_fit, y_fit = np.meshgrid(y_axis, x_axis)
    #x_fit = (x_fit.T * x_axis).T # use meshgrid instead?
    #y_fit = y_fit.T * y_axis
    popt = fitter(fitmodel, x_fit, y_fit, img_cropped)
    
    ## evaluate the model on the original image
    # create x and y values for evaluation
    x = np.arange(0, img.shape[0])*scale[0]
    y = np.arange(0, img.shape[1])*scale[1]
    x, y = np.meshgrid(y,x)

    # evaluate the function and normalize it in order to calculate 1/e and 1e2 contours
    # amplitude_0 and _1 are amplitude of the gaussian and the offset function respectively
    img_result_normalized = (popt(x, y) - popt.amplitude_1) / popt.amplitude_0

    # calculate the contours, maybe the tolerance can be fine tuned
    tolerance = 1e-2
    contour1 = np.where(np.abs(img_result_normalized-1/np.e) < tolerance, 1, 0)
    contour2 = np.where(np.abs(img_result_normalized-1/np.e**2) < tolerance/2, 1, 0)
    contour = contour1 + contour2

    # send the fit results out via dataq
    param_names = popt.param_names
    param_vals = popt.parameters

    return param_names, param_vals, contour
