import cython
import numpy as np
cimport numpy as np
from scipy.optimize import curve_fit
from scipy.stats import skewnorm
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


    # II. In the second part we calculate the estimate

    y_fit = gaussianC(x, mu_0, sigma_0, C0, A0)

    return y_fit, mu_0, sigma_0, C0, A0

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef gaussianC(np.ndarray x, np.float64_t mu, np.float64_t sigma, np.float64_t C, np.float64_t A0):
    return A0 * A / np.abs(sigma) * np.exp(-(x-mu)**2/(2*sigma**2)) + C

def gaussianC_py(x, mu, sigma, C, A0):
    return A0 * A / np.abs(sigma) * np.exp(-(x-mu)**2/(2*sigma**2)) + C

def gaussian_skewed(x, mu, sigma, C, A0, a):
    return A0 * skewnorm.pdf(x, a, mu, sigma) + C

def gaussian_fit_skewed(x, y, verbose=False, avg_window_size=20, threshold_mu=0.5, threshold_sigma=0.1586):
    estimate = True
    p0 = gaussian_estimate(x, y, avg_window_size, threshold_mu, threshold_sigma)[1:]
    p0 = [p for p in p0] + [-3]
    p0[1] = p0[1]/4
    p0[-2] = p0[-2]/2
    if verbose:
        print('p0 (mu_0, sigma_0, C0, A0, skewness):', p0)
    try:
        popt, pcov = curve_fit(gaussian_skewed, x, y, p0)
        estimate = False
    except RuntimeError:
        popt = p0
    x_fit = np.linspace(np.min(x), np.max(x), 500)
    y_fit = gaussian_skewed(x_fit, popt[0], popt[1], popt[2], popt[3], popt[4])
    #print('mu, sigma, C, A0, a')
    #print(p0)
    #print(popt)
    # y_fit = gaussian_skewed(x_fit, p0[0], p0[1], p0[2], p0[3], p0[4])

    return x_fit, y_fit, estimate, popt
