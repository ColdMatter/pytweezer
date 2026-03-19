import cython
import numpy as np
cimport numpy as np
from scipy.optimize import curve_fit
import argparse
from pytweezer.servers import DataClient,ImageClient,Properties

#DTYPE = np.int
#DTYPEf = np.float64
#ctypedef np.int_t DTYPE_y
#ctypedef np.float64_t DTYPEf_t

cdef np.float64_t A = 1/np.sqrt(2*np.pi)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def gaussian_estimate(np.ndarray x, np.ndarray y, np.int N, np.float64_t threshold_mu, np.float64_t threshold_sigma):
    cdef np.ndarray y_normalized, y_smooth, cdf, y_grounded
    cdef np.float64_t mu_0, sigma_0, C0, A0, delta_x, popt, pcov
    cdef int sigma_0_ind, mu_0_ind
    cdef list p0

    # I In the first part we want to get stable estimates for the fit
    # parameters. For this we smooth the data, calculate the cdf and extract the values for
    # mean and standard deviation


    # extract and substract offset
    C0 = min(y)
    y_grounded = y - C0

    # 1. apply a moving average to data
    y_smooth = np.convolve(y_grounded, np.ones((N,))/float(N), mode='same')

    # 2. calculate cdf and extract scaling factor
    delta_x = x[1] - x[0] # TODO: Fix value when working with pixels
    cdf = np.cumsum(y_smooth)*delta_x
    A0 = np.max(cdf)
    cdf /= A0

    # 3. extract stdev and mean
    sigma_0_ind = np.argmax(cdf > threshold_sigma)
    mu_0_ind = np.argmax(cdf > threshold_mu)
    mu_0_ind = min(len(x)-1, max(0, mu_0_ind))
    sigma_0_ind = min(len(x)-1, max(0, sigma_0_ind))
    mu_0 = x[mu_0_ind]
    sigma_0 = mu_0 - x[sigma_0_ind]

    A0 *= 0.2
    sigma_0 *= 0.5

    # II. In the second part we calculate the estimate

    y_fit = gaussianC(x, mu_0, sigma_0, C0, A0)

    return y_fit, mu_0, sigma_0, C0, A0

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef gaussianC(np.ndarray x, np.float64_t mu, np.float64_t sigma, np.float64_t C, np.float64_t A0):
    return A0 * A / np.abs(sigma) * np.exp(-(x-mu)**2/(2*sigma**2)) + C

def gaussianC_py(x, mu, sigma, C, A0):
    return A0 * A / np.abs(sigma) * np.exp(-(x-mu)**2/(2*sigma**2)) + C

def gaussian_fit(x, y, verbose=False, avg_window_size=20, threshold_mu=0.5, threshold_sigma=0.1586):
    estimate = True
    p0 = gaussian_estimate(x, y, avg_window_size, threshold_mu, threshold_sigma)[1:]
    if verbose:
        print('p0 (mu_0, sigma_0, C0, A0):', p0)
    try:
        popt, pcov = curve_fit(gaussianC_py, x, y, p0)#, bounds=([np.min(x), -np.inf, -np.inf, -np.inf], [np.max(x), np.inf, np.inf, np.inf]))
        estimate = False
    except RuntimeError:
        popt = p0
    x_fit = np.linspace(np.min(x), np.max(x), 500)
    y_fit = gaussianC(x_fit, popt[0], popt[1], popt[2], popt[3])
    return x_fit, y_fit, estimate, popt
