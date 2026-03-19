'''
Evaluates whether a single ion is present in a given image using the position, size and skewness of the ion.

Input:
    Data streams: gaussx, gaussy, bloblist

Output:
        Datastream:
            *   result          int, 1 (there is a single ion), 0 (there is no or multiple ions)

Properties:
    *   data streams    ([str])     Input streams
    *   min_sigma_x     (float)     minimum width of an ion
    *   max_sigma_x     (float)     maximum width of an ion
    *   min_sigma_y     (float)     minimum height of an ion
    *   max_sigma_y     (float)     maximum height of an ion
    *   min_pos_x       (float)     minimum x position of an ion
    *   max_pos_x       (float)     maximum x position of an ion
    *   min_pos_y       (float)     minimum y position of an ion
    *   max_pos_y       (float)     maximum y position of an ion
    *   min_skewness_x  (float)     if using a gauss_skew fit
    *   max_skewness_x  (float)     if using a gauss_skew fit
    *   check_skewness  (bool)      whether using a gauss_skew fit
'''


import numpy as np
from pytweezer.servers import DataClient, ImageClient
from pytweezer.servers import Properties, PropertyAttribute
import skimage.feature as sk
import argparse


class SingleIonPresent:
    _datastreams = PropertyAttribute('datastreams', ['Experiment.start', 'Ion_Countbloblist_', 'radba_yrowsumcut.radba_gaussy', 'radba_xcolsumcut.radba_gaussx_skewed'])
    _verbose_output = PropertyAttribute('verbose_output', True)
    _check_skewness_x = PropertyAttribute('check_skewness_x', True)
    _check_skewness_y = PropertyAttribute('check_skewness_y', False)
    _check_offset_x = PropertyAttribute('check_offset_x', False)
    _check_offset_y = PropertyAttribute('check_offset_y', False)
    _check_pos_x = PropertyAttribute('check_pos_x', False)
    _check_pos_y = PropertyAttribute('check_pos_y', False)
    _min_pos_x = PropertyAttribute('min_pos_x', 3.66e-3)
    _max_pos_x = PropertyAttribute('max_pos_x', 3.72e-3)
    _min_sigma_x = PropertyAttribute('min_sigma_x', 1.e-5)
    _max_sigma_x = PropertyAttribute('max_sigma_x', 5.e-5)
    _min_offset_x = PropertyAttribute('min_offset_x', 550.)
    _max_offset_x = PropertyAttribute('max_offset_x', 680.)
    _min_skew_x = PropertyAttribute('min_skewness_x', -7.5)
    _max_skew_x = PropertyAttribute('max_skewness_x', -1.3)
    _min_pos_y = PropertyAttribute('min_pos_y', 2.252e-3)
    _max_pos_y = PropertyAttribute('max_pos_y', 2.257e-3)
    _min_sigma_y = PropertyAttribute('min_sigma_y', 6.e-6)
    _max_sigma_y = PropertyAttribute('max_sigma_y', 1.1e-5)
    _min_offset_y = PropertyAttribute('min_offset_y', 350.)
    _max_offset_y = PropertyAttribute('max_offset_y', 450.)
    _min_skew_y = PropertyAttribute('min_skewness_y', -8.)
    _max_skew_y = PropertyAttribute('max_skewness_y', -1.)


    def __init__(self, name):
        self._props = Properties(name)
        self.dataq = DataClient(name.split('/')[-1])
        self.dataq.subscribe(self._datastreams)
        print('\n\033[1msingle_ion_present.py - subscriptions: {0}\n\033[0m'.format(self._datastreams))
        self.name = name

    def run(self):
        received_gauss_fit_x = False
        received_gauss_fit_y = False
        received_bloblist = False

        N_bright = 0
        pos_x = 0
        sigma_x = 0
        offset_x = 0
        skew_x = 0
        pos_y = 0
        sigma_y = 0
        offset_y = 0
        skew_y = 0

        while True:
            msg = self.dataq.recv()

            if msg is not None:
                msgstr, head, data = msg
                self.msg = msg
                self.head = head

                # print('\n\n\nsingle_ion_present.py - msgstr:', msgstr)
                # print('\n\n\nsingle_ion_present.py - head:', head)
                # print('\n\n\nsingle_ion_present.py - data:', data, '\n\n\n')

                if msgstr == 'Ion_Countbloblist_':
                    received_bloblist = True
                    N_bright = int(head['N_bright'])
                elif 'gaussx' in msgstr.replace('_', ''):
                    received_gauss_fit_x = True
                    pos_x = float(head['pos'])
                    sigma_x = float(head['sigma'])
                    offset_x = float(head['offset'])
                    if self._check_skewness_x:
                        #print(head)
                        skew_x = float(head['a'])
                elif 'gaussy' in msgstr.replace('_', ''):
                    received_gauss_fit_y = True
                    pos_y = float(head['pos'])
                    sigma_y = float(head['sigma'])
                    offset_y = float(head['offset'])
                    if self._check_skewness_y:
                        skew_y = float(head['a'])

                if received_gauss_fit_x and received_gauss_fit_y and received_bloblist:
                    # print('\n\n\nsingle_ion_present.py: got all\n\n\n')

                    result = N_bright
                    if result > 1:
                        result = 0

                    printed = False

                    if self._check_pos_x:
                        if not (self._min_pos_x <= pos_x <= self._max_pos_x):
                            result = 0
                            if self._verbose_output:
                                print('{3}single_ion_present.py - filtered out pos_x: {0} <= {1} <= {2}'.
                                      format(self._min_pos_x, pos_x, self._max_pos_x, '' if printed else '\n'))
                                printed = True
                    if not (self._min_sigma_x <= sigma_x <= self._max_sigma_x):
                        result = 0
                        if self._verbose_output:
                            print('{3}single_ion_present.py - filtered out sigma_x: {0} <= {1} <= {2}'.
                                  format(self._min_sigma_x, sigma_x, self._max_sigma_x, '' if printed else '\n'))
                            printed = True
                    if self._check_offset_x:
                        if not (self._min_offset_x <= offset_x <= self._max_offset_x):
                            result = 0
                            if self._verbose_output:
                                print('{3}single_ion_present.py - filtered out offset_x: {0} <= {1} <= {2}'.
                                      format(self._min_offset_x, offset_x, self._max_offset_x, '' if printed else '\n'))
                                printed = True
                    if self._check_skewness_x:
                        if not (self._min_skew_x <= skew_x <= self._max_skew_x):
                            result = 0
                            if self._verbose_output:
                                print('{3}single_ion_present.py - filtered out skew_x: {0} <= {1} <= {2}'.
                                      format(self._min_skew_x, skew_x, self._max_skew_x, '' if printed else '\n'))
                                printed = True

                    if self._check_pos_y:
                        if not (self._min_pos_y <= pos_y <= self._max_pos_y):
                            result = 0
                            if self._verbose_output:
                                print('{3}single_ion_present.py - filtered out pos_y: {0} <= {1} <= {2}'.
                                      format(self._min_pos_y, pos_y, self._max_pos_y, '' if printed else '\n'))
                                printed = True
                    if not (self._min_sigma_y <= sigma_y <= self._max_sigma_y):
                        result = 0
                        if self._verbose_output:
                            print('{3}single_ion_present.py - filtered out sigma_y: {0} <= {1} <= {2}'.
                                  format(self._min_sigma_y, sigma_y, self._max_sigma_y, '' if printed else '\n'))
                            printed = True
                    if self._check_offset_y:
                        if not (self._min_offset_y <= offset_y <= self._max_offset_y):
                            result = 0
                            if self._verbose_output:
                                print('{3}single_ion_present.py - filtered out offset_y: {0} <= {1} <= {2}'.
                                      format(self._min_offset_y, offset_y, self._max_offset_y, '' if printed else '\n'))
                                printed = True
                    if self._check_skewness_y:
                        if not (self._min_skew_y <= skew_y <= self._max_skew_y):
                            result = 0
                            if self._verbose_output:
                                print('{3}single_ion_present.py - filtered out skew_y: {0} <= {1} <= {2}\n'.
                                      format(self._min_skew_y, skew_y, self._max_skew_y, '' if printed else ''))
                                printed = True

                    if printed:
                        print()

                    self.head['result'] = result
                    self.dataq.send(self.head, result, '_result')
                    if self._verbose_output:
                        print('single_ion_present.py: exactly_one_ion = {0}'.format(result == 1))

                    received_gauss_fit_x = False
                    received_gauss_fit_y = False
                    received_bloblist = False


def main_run(name):
    slc = SingleIonPresent(name)
    slc.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs=1, help='name of this program instance')
    args = parser.parse_args()
    name = args.name[0]
    main_run(name)
