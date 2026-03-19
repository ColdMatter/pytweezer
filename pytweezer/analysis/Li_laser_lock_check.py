import time
from pytweezer.servers import DataClient
from pytweezer.servers import Properties, PropertyAttribute
from pytweezer.analysis.print_messages import print_error
import pandas as pd
import numpy as np
import argparse

from pytweezer.servers.mattermost_interface import post_to_mattermost


class LiLaserLockCheck:
    _datastreams = PropertyAttribute('datastreams', ['Experiment.start',
                                                     'abs_h_cutrowintcut.abs_h_gaussy'])
    _MOT_loading_limits = PropertyAttribute('_MOT_loading_limits', [1, 5])
    _check_MOT_loading_limits = PropertyAttribute('_check_MOT_loading_limits', True)
    _TOF_limits = PropertyAttribute('_TOF_limits', [50e-6, 500e-6])
    _check_TOF_limits = PropertyAttribute('_check_TOF_limits', True)
    _img_freq = PropertyAttribute('_img_freq', 318.8)
    _check_img_freq = PropertyAttribute('_check_img_freq', True)
    _check_B_spin_pol_B_img = PropertyAttribute('_check_B_spin_pol_B_img', True)
    _p_eva_limits = PropertyAttribute('_p_eva_limits', [0.9e-2, 5e-2])
    _check_p_eva_limits = PropertyAttribute('_check_p_eva_limits', True)
    _check_atom_shutter = PropertyAttribute('_check_atom_shutter', True)
    _check_odt_on = PropertyAttribute('_check_odt_on', True)
    _check_high_field_on = PropertyAttribute('_check_high_field_on', True)
    _check_on_horiz = PropertyAttribute('_check_on_horiz', True)
    _check_rf_pulse_off = PropertyAttribute('_check_rf_pulse_off', True)
    _decimating_duration_limits = PropertyAttribute('_decimating_duration_limits', [0, 15])
    _check_decimating_duration_limits = PropertyAttribute('_check_decimating_duration_limits', True)
    _totalint_on_A0_off = PropertyAttribute('_totalint_on_A0_off', False)

    _atom_number_limits = PropertyAttribute('_atom_number_limits', [5e3, 28e3])

    _verbose_output = PropertyAttribute('_verbose_output', False)

    def __init__(self, name):
        self._props = Properties(name)
        self.dataq = DataClient(name.split('/')[-1])
        self.dataq.subscribe(self._datastreams)
        self.name = name

        self.experiment_name = '00_Interaction_optical_dipole_trap'

        self.reset()

    def reset(self):
        self.results = pd.DataFrame(columns=['task', 'rep', 'run', 'experiment', 'timestamp'])
        self.fit_name = 'A0'
        if self. _totalint_on_A0_off:
            self.fit_name = 'totalint_roi'
        self.already_failed = False

    def run(self):
        while True:
            msg = self.dataq.recv()
            if msg is not None:
                self.fit_name = 'A0'
                if self. _totalint_on_A0_off:
                    self.fit_name = 'totalint_roi'
                if len(msg) == 2:
                    _, experiment = msg
                    if experiment['_expName'] == self.experiment_name and self._experiment_valid(experiment):
                        new_exp = pd.DataFrame([{'task': experiment['_task'], 'rep': experiment['_repetition'],
                                                 'run': experiment['_run'], 'experiment': experiment['_expName'],
                                                 'timestamp': None}])
                        try:
                            self.results = pd.concat([self.results, new_exp], ignore_index=True).reset_index(drop=True)
                        except Exception as e:
                            print_error('Li_laser_lock_check.py: Error trying to concat data frame {0} with data frame '
                                        '{1}. Resetting now ...'.format(self.results, new_exp), 'error')
                            self.reset()
                        self.results = self.results[self.results['task'] > new_exp['task'] - 10]
                        if self._verbose_output:
                            print_error('Li_laser_lock_check.py: Got experiment, results so far: '
                                        '{0}.'.format(self.results), 'weak')
                elif len(msg) == 3:
                    msgstr, head, _ = msg
                    row = ((self.results['task'] == head['_task']) & (self.results['rep'] == head['_repetition'])
                           & (self.results['run'] == head['_run']))
                    if row.any() and msgstr == 'abs_h_cutrowintcut.abs_h_gaussy' and self.fit_name in head:
                        self.results.loc[row, 'timestamp'] = head['timestamp']
                        atoms_present = (min(self._atom_number_limits) <= head[self.fit_name]
                                         <= max(self._atom_number_limits))
                        if not atoms_present:
                            if not self.already_failed:
                                send_to_mattermost()
                            self.already_failed = True
                            print_error('Li_laser_lock_check.py: Li laser(s) out of lock!', 'error')
                        else:
                            self.already_failed = False
                        if self._verbose_output:
                            print_error('Li_laser_lock_check.py: Atom number by total_int is N = {0}, limits are [{1},'
                                        ' {2}]'.format(head[self.fit_name], min(self._atom_number_limits),
                                                       max(self._atom_number_limits)), 'weak')
                        self.results.drop(row.index, inplace=True)
                        if self._verbose_output and atoms_present:
                            print_error('Li_laser_lock_check.py: Got atom image with atoms.', 'success')
                    else:  # No clear distinction possible, reset.
                        if self._verbose_output:
                            print_error('Li_laser_lock_check.py: No experiment found for image.',
                                        'warning')
                else:
                    print_error('Li_laser_lock_check.py: Got invalid message length {0}'.format(len(msg)), 'error')

            time.sleep(0.1)  # sec

    def _experiment_valid(self, exp_params):
        valid = True

        if self._check_MOT_loading_limits:
            valid = self._single_exp_param_limits_check(exp_params=exp_params, param='MOT_loading',
                                                        limits=self._MOT_loading_limits)
            if not valid:
                return valid

        if self._check_TOF_limits:
            valid = self._single_exp_param_limits_check(exp_params=exp_params, param='time_of_flight',
                                                        limits=self._TOF_limits)
            if not valid:
                return valid

        if self._check_p_eva_limits:
            valid = self._single_exp_param_limits_check(exp_params=exp_params, param='eva_ramp_target_p',
                                                        limits=self._p_eva_limits)
            if not valid:
                return valid

        if self._check_decimating_duration_limits:
            valid = self._single_exp_param_limits_check(exp_params=exp_params, param='decimating_duration',
                                                        limits=self._decimating_duration_limits)
            if not valid:
                return valid

        if self._check_img_freq:
            valid = self._single_exp_param_value_check(exp_params=exp_params, param='imaging_freq_shift',
                                                       set_val=self._img_freq)
            if not valid:
                return valid

        if self._check_atom_shutter:
            valid = self._single_exp_param_value_check(exp_params=exp_params, param='atoms_present',
                                                       set_val=True)
            if not valid:
                return valid

        if self._check_odt_on:
            valid = self._single_exp_param_value_check(exp_params=exp_params, param='odt_on',
                                                       set_val=True)
            if not valid:
                return valid

        if self._check_high_field_on:
            valid = self._single_exp_param_value_check(exp_params=exp_params, param='high_field_on',
                                                       set_val=True)
            if not valid:
                return valid

        if self._check_on_horiz:
            valid = self._single_exp_param_value_check(exp_params=exp_params, param='on_horiz_off_vertical',
                                                       set_val=True)
            if not valid:
                return valid

        if self._check_rf_pulse_off:
            valid = self._single_exp_param_value_check(exp_params=exp_params, param='rf_pulse',
                                                       set_val=False)
            if not valid:
                return valid

        if self._check_B_spin_pol_B_img:
            param = 'B_spin_pol'
            B_spin_pol = 0
            if param in exp_params:
                B_spin_pol = exp_params[param]
            else:
                print_error('Li_laser_lock_check.py - _experiment_valid(): Received corrupt '
                            'experiment parameters:\n{0}.'.format(exp_params), 'error')

            param = 'B_imaging'
            B_imaging = 0
            if param in exp_params:
                B_imaging = exp_params[param]
            else:
                print_error('Li_laser_lock_check.py - _experiment_valid(): Received corrupt '
                            'experiment parameters:\n{0}.'.format(exp_params), 'error')

            valid = B_spin_pol in [293, 345] and B_imaging in [293, 345] and B_imaging != B_spin_pol

        return valid

    def _single_exp_param_limits_check(self, exp_params, param, limits):
        if param in exp_params:
            if not (min(limits) <= exp_params[param] <= max(limits)):
                if self._verbose_output:
                    print_error('Li_laser_lock_check.py - _experiment_valid(): Filtered out {0} = {1}, it\'s '
                                'not within {2}.'.format(param, exp_params[param], limits), 'weak')
                return False
            else:
                return True
        else:
            print_error('Li_laser_lock_check.py - _single_exp_param_limits_check(): Received corrupt experiment '
                        'parameters:\n{0}.'.format(exp_params), 'error')
            return False

    def _single_exp_param_value_check(self, exp_params, param, set_val):
        if param in exp_params:
            if exp_params[param] != set_val:
                if self._verbose_output:
                    print_error('Li_laser_lock_check.py - _experiment_valid(): Filtered out {0} = {1}, it\'s '
                                'not equal to {2}.'.format(param, exp_params[param], set_val), 'weak')
                return False
            else:
                return True
        else:
            print_error('Li_laser_lock_check.py - _single_exp_param_value_check(): Received corrupt '
                        'experiment parameters:\n{0}.'.format(exp_params), 'error')
            return False


def send_to_mattermost():
    msg_mm = 'Li_laser_lock_check alert:'
    msg_mm += '\nCheck the Li laser(s), the lock(s) probably failed!'
    post_to_mattermost(msg=msg_mm, channel_name="lab-status")


def main_run(name):
    slc = LiLaserLockCheck(name)
    slc.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs=1, help='name of this program instance')
    args = parser.parse_args()
    name = args.name[0]
    main_run(name)
