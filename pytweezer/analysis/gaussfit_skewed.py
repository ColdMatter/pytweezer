''' Perform a 1D skew normal fit on a datataset

Output:

    The output is a datastream containing a dictionary with the fit results and a 2d array containing the
    fitted curve.

    Fitparameter nomenclature
        *   A0 :    :math:`A_0`
        *   sigma:  :math:`\\sigma`
        *   pos:    :math:`\\mu`
        *   offset: :math:`C`
        *   height: :math:`\\frac{A_0}{\sqrt{2\pi}\sigma}`
        *   peakdensity_3D: :math:`\\frac{A_0}{\sqrt{2\pi}^3\sigma^3}`
        *   skewness: :math: `a`


Format of the input datastream:

    The fit can handle two types of data:

    *   1D numpy.array (data.ndim ==1):
        the y-values are taken from the data, and the x values are a linear spaced index from 0 to ny.
    *   2D numpy array:

        *   x=data[0]
        *   y=data[1]

Output datastream:



Properties controlling the program:

    datastreams:    []
        list of input datastreams


'''

# build and import cython gauss fit routine
import pyximport
pyximport.install()
from pytweezer.analysis.src.gaussskewfit import gaussian_fit_skewed

import numpy as np
import argparse
from pytweezer.servers import DataClient,ImageClient,Properties,PropertyAttribute


class GaussfitSkewed:
    _datastreams   =PropertyAttribute('datastreams',['None'])
    _avg_window_size = PropertyAttribute('_avg_window_size', 20)
    _threshold_mu = PropertyAttribute('_threshold_mu', 0.5)
    _threshold_sigma = PropertyAttribute('_threshold_sigma', 0.1586)
    _verbose = PropertyAttribute('_verbose', False)

    def __init__(self,name):
        self._props=Properties(name)
        self.dataq=DataClient(name)
        self.dataq.subscribe(self._datastreams)
        print('gaussfit_skewed.py subscriptions: ',self._datastreams)


    def run(self):
        while True:
            msg = self.dataq.recv()
            #print('gaussfit.py recv')
            if msg != None:
                #print(name+'received')
                msgstr,head,data = msg
                y_fit = data
                #print(data.shape)
                if data.ndim == 1:
                    y=data
                    y = y[not np.isnan(y)]
                    x=np.arange(y.shape[0])
                else:
                    x = data[0]
                    y = data[1]
                    x = x[np.invert(np.isnan(y))]
                    y = y[np.invert(np.isnan(y))]
                try:
                    x, y_fit, converged, parameters = gaussian_fit_skewed(x, y, verbose=self._verbose,
                                                                          avg_window_size=self._avg_window_size,
                                                                          threshold_mu=self._threshold_mu,
                                                                          threshold_sigma=self._threshold_sigma)
                except Exception as e:
                    print(e)
                    continue
                head['pos'] = parameters[0]
                head['sigma'] = np.abs(parameters[1])
                head['offset'] = parameters[2]
                head['A0'] = np.abs(parameters[3])
                head['a'] = parameters[4]
                head['converged'] = not converged
                head['height'] = parameters[3]/(parameters[1]*np.sqrt(2*np.pi))

                # estimate maximum (mode, source: wikipedia)
                xi = parameters[0]
                omega = parameters[1]
                alpha = parameters[4]
                delta = alpha / np.sqrt(1+alpha**2)
                mu_z = np.sqrt(2/np.pi)*delta
                sigma_z = np.sqrt(1-mu_z**2)
                gamma = (4-np.pi)/2 * (delta*np.sqrt(2/np.pi))**3 / (1-2*delta**2/np.pi)**(3/2)
                m0 = mu_z - gamma*sigma_z/2 - np.sign(alpha)/2 * np.exp(-2*np.pi/np.abs(alpha))
                head['mode'] = xi + omega*m0


                #print('gaussfit.py sending')
                self.dataq.send(head,np.vstack([x,y_fit]),name.split('/')[-1],prefix=msgstr[:-1])
                #print('sending on ',name+'/colsum')




def main_run(name):
    slc=GaussfitSkewed(name)
    slc.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs=1, help='name of this program instance')
    args = parser.parse_args()
    name=args.name[0]
    main_run(name)
