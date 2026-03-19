import datetime
import time

from pytiamo.atom_ion.feshbach_plotter import dateTime_to_unix

from pytweezer.servers import DataClient
from pytweezer.servers import Properties, PropertyAttribute
from pytweezer.analysis.print_messages import print_error
import pandas as pd
import numpy as np
import argparse

from pytweezer.servers.mattermost_interface import post_to_mattermost


class BaLaserLockCheck:
    _datastreams = PropertyAttribute('datastreams', ['Experiment.start',
                                                     'radba_xcolsumcut.radba_gaussx_skewed'])
    _experiment_name = PropertyAttribute('_experiment_name', '02_compensation_helper')
    _data_name = PropertyAttribute('_data_name', 'radba_xcolsumcut.radba_gaussx_skewed')
    _fit_name = PropertyAttribute('_fit_name', 'A0')
    _experiment_filters = PropertyAttribute('_experiment_filters',
                                            {'rf_set': 4, 'shutter_time': 3, 'cooling_power': 2.5,
                                             'take_images': True, 'use_trap_x_y_ax_default': True,
                                             'specify_camera_exposure': False})
    _binning = PropertyAttribute('_binning', 15)
    # _fluor_to_MHz = PropertyAttribute('_fluor_to_MHz', 0.003)  # MHz / A0, see Joplin, 2024-01-15
    _fluor_set = PropertyAttribute('_fluor_set', 0.0303)  # see Joplin, 2024-01-22
    _freq_set_MHz = PropertyAttribute('_freq_set_MHz', 95.0)  # see Joplin, 2024-01-22
    _max_drift_MHz_before_alert = PropertyAttribute('_max_drift_MHz_before_alert', 4)
    _max_drift_MHz_before_correction = PropertyAttribute('_max_drift_MHz_before_correction', 0.5)
    _max_single_correction_MHz = PropertyAttribute('_max_single_correction_MHz', 1.5)
    _freq_MHz_min_max = PropertyAttribute('_freq_MHz_min_max', [88.0, 100.0])
    _verbose_output = PropertyAttribute('_verbose_output', False)

    _check_sigma_min_max = PropertyAttribute('_check_sigma_min_max', True)
    _sigma_min_max = PropertyAttribute('_sigma_min_max', [1.0e-7, 0.02])
    _check_A0_min_max = PropertyAttribute('_check_A0_min_max', True)
    _A0_min_max = PropertyAttribute('_A0_min_max', [0.0001, 0.08])
    _check_pos_min_max = PropertyAttribute('_check_pos_min_max', True)
    _pos_min_max = PropertyAttribute('_pos_min_max', [0.0036, 0.0038])
    _check_converged = PropertyAttribute('_check_converged', True)

    def __init__(self, name):
        self._props = Properties(name)
        self.dataq = DataClient(name.split('/')[-1])
        self.dataq.subscribe(self._datastreams)
        self.name = name

        self.reset()

    def reset(self):
        self.results = pd.DataFrame(columns=['task', 'rep', 'run', 'experiment', 'timestamp'])
        # self.fluor_start_up = []
        self.fluor_measurement = []
        self.already_failed = False

    def run(self):
        while True:
            if self._props.get('/BaliBrowser/Looper/loop_starts_now', False):
                self._props.set('/BaliBrowser/Looper/loop_starts_now', False)
                self.reset()

            msg = self.dataq.recv()
            if msg is not None:
                if len(msg) == 2:
                    _, experiment = msg
                    if experiment['_expName'] == self._experiment_name and self._experiment_valid(experiment):
                        new_exp = pd.DataFrame([{'task': experiment['_task'], 'rep': experiment['_repetition'],
                                                 'run': experiment['_run'], 'experiment': experiment['_expName'],
                                                 'timestamp': datetime.datetime.now()}])
                        try:
                            self.results = pd.concat([self.results, new_exp], ignore_index=True).reset_index(drop=True)
                        except Exception as e:
                            print_error('BaLaserLockCheck.py: Error trying to concat data frame {0} with data frame '
                                        '{1}. Resetting \'results\' now ...\n{2}'.format(self.results, new_exp, e),
                                        'error')
                            self.results = pd.DataFrame(columns=['task', 'rep', 'run', 'experiment', 'timestamp'])
                        try:
                            self.results = self.results[self.results['timestamp'] > new_exp['timestamp'] - datetime.timedelta(hours=5)]
                        except Exception as e:
                            print_error('BaLaserLockCheck.py: Error trying to compare data frame {0} with data frame '
                                        '{1}. Resetting \'results\' now ...\n{2}'.format(self.results, new_exp, e),
                                        'error')
                            self.results = pd.DataFrame(columns=['task', 'rep', 'run', 'experiment', 'timestamp'])
                        if self._verbose_output:
                            print_error('BaLaserLockCheck.py: Got experiment, results so far: '
                                        '{0}.'.format(self.results), 'weak')
                elif len(msg) == 3:
                    msgstr, head, _ = msg
                    row = ((self.results['task'] == head['_task']) & (self.results['rep'] == head['_repetition'])
                           & (self.results['run'] == head['_run']))
                    if row.any() and msgstr == self._data_name and self._fit_name in head:
                        row = [not r for r in row]
                        self.results = self.results[row]
                        if self._ion_present(head):
                            fluor = head[self._fit_name]
                            if False:  # len(self.fluor_start_up) < self._binning:
                                self.fluor_start_up.append(fluor)
                                self._props.set('/BaliBrowser/Looper/loop_starts_now', False)
                            else:
                                self.fluor_measurement.append(fluor)
                            #if self._verbose_output:
                            #    print_error('BaLaserLockCheck.py: Fluorescence so far: \nfluor_start_up: {0},\nfluor_'
                            #                'measurement: {1}.'.format(self.fluor_start_up, self.fluor_measurement),
                            #                'weak')

                            if len(self.fluor_measurement) >= self._binning:
                                # fluor_0 = np.median(self.fluor_start_up)
                                fluor_fit = 0.013703  # See Joplin, 2024-01-22
                                fluor_1 = np.median(self.fluor_measurement)
                                freq_defaults = self._props.get('/Experiments/Defaults/ba_493_freq', 95.0e6)

                                # Estimate the fluorescence without AOM efficiency drop: See Joplin, 2024-01-29
                                fluor_1_comp = fluor_1 * AOM_fit(self._freq_set_MHz * 1e6) / AOM_fit(freq_defaults)
                                if self._verbose_output:
                                    print_error('Ba_laser_lock_check: Estimated an AOM efficiency free fluorescence, '
                                                '{0} instead of {1}'.format(fluor_1_comp, fluor_1), 'info')

                                # fluor_diff = np.abs(fluor_0 - fluor_1)
                                freq_meas = fluor_to_freq(fluor_1_comp * fluor_fit / self._fluor_set)  # Hz
                                # drift_MHz = fluor_diff / self._fluor_to_MHz
                                drift_MHz = freq_meas * 1e-6 - self._freq_set_MHz
                                print_error('Ba_laser_lock_check: Measured {0} MHz, set value is {1} MHz, defaults read'
                                            ' {2} MHz'.format(np.round(freq_meas * 1e-6, 1),
                                                              self._freq_set_MHz, freq_defaults * 1e-6), 'weak')
                                drift_MHz_step = (freq_meas - freq_defaults) * 1e-6
                                freq_new_MHz = freq_defaults * 1e-6 - drift_MHz
                                freq_to_defaults = freq_defaults

                                if np.abs(drift_MHz) > self._max_drift_MHz_before_correction or np.abs(drift_MHz) > self._max_drift_MHz_before_alert:
                                    correct = True
                                    if not np.abs(drift_MHz) > self._max_drift_MHz_before_correction:
                                        correct = False
                                        print_error('Ba_laser_lock_check: Won\'t correct, because abs(drift_MHz) = '
                                                    '{0} <= {1} = _max_drift_MHz_before_correction'.format(np.abs(drift_MHz),
                                                                                                           self._max_drift_MHz_before_correction), 'info')
                                    if not np.abs(drift_MHz_step) <= self._max_single_correction_MHz:
                                        correct = False
                                        print_error('Ba_laser_lock_check: Won\'t correct, because abs(drift_MHz_step) = '
                                                    '{0} > {1} = _max_single_correction_MHz'.format(np.abs(drift_MHz_step),
                                                                                                           self._max_single_correction_MHz), 'info')
                                    if not max(self._freq_MHz_min_max) >= freq_new_MHz >= min(self._freq_MHz_min_max):
                                        correct = False
                                        print_error('Ba_laser_lock_check: Won\'t correct, because freq_new_MHz = {0}'
                                                    ' is outside of {1} = _freq_MHz_min_max'.format(np.abs(drift_MHz),
                                                                                                           self._freq_MHz_min_max), 'info')
                                    if correct:
                                        freq_to_defaults = freq_new_MHz * 1e6
                                        print_error('Ba_laser_lock_check: Setting the 493 nm AOM frequency to {0} '
                                                    'MHz.'.format(np.round(freq_new_MHz, 1)), 'weak')
                                        print(freq_to_defaults)
                                        self._props.set('/Experiments/Defaults/ba_493_freq', freq_to_defaults)
                                    alert_mm = np.abs(drift_MHz) > self._max_drift_MHz_before_alert \
                                        and drift_MHz_step >= self._max_drift_MHz_before_correction \
                                        and not self.already_failed
                                    alert_incl_mattermost(drift_MHz, self._freq_set_MHz, freq_meas, correct=correct,
                                                          mattermost=alert_mm)
                                    if alert_mm:
                                        self.already_failed = True
                                else:
                                    if self._verbose_output:
                                        print_error('Ba_laser_lock_check: All fine! Fluorescence = {0}, frequency = {1} MHz'
                                                    '.'.format(fluor_1, np.round(freq_meas * 1e-6, 1)), 'success')
                                    self.already_failed = False
                                self.fluor_measurement = []

                                head = {'timestamp': dateTime_to_unix([datetime.datetime.now()])[0],
                                        'fluorescence': fluor_1, 'freq_meas': freq_meas,
                                        'freq_offset': drift_MHz * 1e6, 'freq_step': drift_MHz_step * 1e6,
                                        'freq_new': freq_to_defaults}
                                self.dataq.send(head)
                else:
                    print_error('Ba_laser_lock_check.py: Got invalid message length {0}'.format(len(msg)), 'error')

            time.sleep(0.1)  # sec

    def _experiment_valid(self, exp_params):
        for filter in self._experiment_filters:
            if filter not in exp_params or exp_params[filter] != self._experiment_filters[filter]:
                if self._verbose_output:
                    print_error('Ba_laser_lock_check - _experiment_valid(): Filtered out experiment because {0} = {1} '
                                '!= {2}'.format(filter, exp_params[filter], self._experiment_filters[filter]),
                                'weak')
                return False
        return True

    def _ion_present(self, head):
        pos = head['pos']
        sigma = head['sigma']
        A0 = head['A0']
        converged = head['converged']

        filtered_out = False
        if self._check_A0_min_max and not filtered_out:
            filtered_out = not self._check_limits(A0, self._A0_min_max, 'A0')
        if self._check_sigma_min_max and not filtered_out:
            filtered_out = not self._check_limits(sigma, self._sigma_min_max, 'sigma')
        if self._check_pos_min_max and not filtered_out:
            filtered_out = not self._check_limits(pos, self._pos_min_max, 'pos')
        if self._check_converged and not filtered_out:
            filtered_out = not self._check_bool(converged, 'converged')

        return not filtered_out

    def _check_limits(self, value, bounds, msg):
        inside = max(bounds) >= value >= min(bounds)

        if not inside and self._verbose_output:
            print_error('Ba_laser_lock_check.py: Filtered out image by {0}: {1} >= {2} >= '
                        '{3}.'.format(msg, max(bounds), value, min(bounds)), 'warning')

        return inside

    def _check_bool(self, value, msg):
        inside = value

        if self._verbose_output and not inside:
            print_error('Ba_laser_lock_check.py: Filtered out image by {0}: {1}.'.format(msg, value),
                        'warning')

        return inside


def freq_to_fluor(freq):  # Hz
    freq_offset = 95e6
    freq = (freq - freq_offset) * 1e-6  # freq in MHz w/o offset for easier fit
    center, fwhm, amplitude, y_0, slope_lin = [6.36287877, 12.23491772, 0.8020297, 0.04517133, 0.03183636]
    return lorentzian([center, fwhm, amplitude, y_0], freq) * slope_lin

def fluor_to_freq(fluor):
    freq_offset = 95e6
    center, fwhm, amplitude, y_0, slope_lin = [6.36287877, 12.23491772, 0.8020297, 0.04517133, 0.03183636]
    return lorentzian_inverse([center, fwhm, amplitude, y_0], fluor / slope_lin) * 1e6 + freq_offset  # Hz

def AOM_fit(freq):
    center = -8.75669804
    amplitude = -5.27228382e-05
    y_max = 2.94532076e-02

    freq_offset = 95e6
    freq = (freq - freq_offset) * 1e-6  # freq in MHz w/o offset for easier fit
    return amplitude * (freq - center) ** 2 + y_max

def lorentzian(p, x):
    mu = p[0]
    fwhm = p[1]
    A0 = p[2]
    c = p[3]
    y = (mu - x) / (fwhm / 2)
    return A0 / (1 + y ** 2) + c

def lorentzian_inverse(p, x):
    mu = p[0]
    fwhm = p[1]
    A0 = p[2]
    c = p[3]
    y = np.sqrt(A0 / (x - c) - 1)
    return mu - (fwhm / 2) * y

def alert_incl_mattermost(diff_MHz, freq_set, freq_meas, correct=False, mattermost=True):
    msg_mm = 'Ba_laser_lock_check alert:'
    msg_mm += '\nMeasured a probable frequency shift of {0} MHz,'.format(np.round(diff_MHz, 1))
    msg_mm += '\nset freq was {0} MHz,'.format(np.round(freq_set, 1))
    msg_mm += '\nmeasured freq was {0} MHz,'.format(np.round(freq_meas * 1e-6, 1))
    msg_mm += '\nwill{0} correct for it,'.format(' not' if not correct else '')
    msg_mm += '\ncheck the Ba laser(s) and wave meter!'
    if mattermost:
        post_to_mattermost(msg=msg_mm, channel_name="lab-status")
    print_error(msg_mm, 'error')


def main_run(name):
    slc = BaLaserLockCheck(name)
    slc.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs=1, help='name of this program instance')
    args = parser.parse_args()
    name = args.name[0]
    main_run(name)
