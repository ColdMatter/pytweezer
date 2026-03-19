'''

Takes two images (should be by 02_compensation_helper.py), one with low and one with high rf confinement and calculates
the position shift in all three spatial dimensions.

Input:
    Datastreams

Output:
        Datastream:
            *   ionchecklist    Truth table for presence of ion in each image
            *   result          int corresponding to ion product state

Properties:
    *   imagestreams   ([str]) Input streams
    *   max_sigma   (float)    Parameter from blob_dog
    *   threshold   (float)    Parameter from blob_dog
    *   Region_of_Interest     A region of interest that can be used to sort out unwanted blobs
    *   Filter by ROI   (bool) If True, blobs outside the ROI will be sorted out


'''
from datetime import datetime, timedelta

import numpy as np

from pytweezer.experiment.trap_shim import Trap_Shim
from pytweezer.servers import DataClient
from pytweezer.servers import Properties, PropertyAttribute
import skimage.feature as sk
import argparse
from pytweezer.analysis.print_messages import print_error
import pandas as pd

from pytweezer.servers.mattermost_interface import post_to_mattermost


class StrayFieldCheck:
    _datastreams = PropertyAttribute('datastreams',
                                     ['axcolsumcut.axgaussx', 'ax_yrowsumcut.axgaussy',
                                      'radba_xcolsumcut.radba_gaussx_skewed', 'radba_yrowsumcut.radba_gaussy',
                                      'axba_allcolsumcut.axba_all_gauss_x', 'axba_allrowsumcut.axba_all_gauss_y',
                                      'radba_allcolsumcut.radba_all_gauss_x', 'radba_allrowsumcut.radba_all_gauss_y'])

    _conversion_factors = PropertyAttribute('conversion_factors', [5056.0, 7175.0, 2775.0])
    _verbose_output = PropertyAttribute('verbose_output', False)
    _auto_correct = PropertyAttribute('auto_correct', False)
    _use_radba_for_y = PropertyAttribute('use_radba_for_y', False)
    _rf_low = PropertyAttribute('rf_low', 0.35)
    _rf_high = PropertyAttribute('rf_high', 4.0)
    _binning = PropertyAttribute('binning', 3)

    # end cap voltage vs spatial ion displacement:
    _slopes_ax_gaussx_low_high = PropertyAttribute('slopes_ax_gaussx_low_high',
                                                   [-1.594e-03, -1.046e-05])
    _slopes_ax_gaussy_low_high = PropertyAttribute('slopes_ax_gaussy_low_high',
                                                   [3.7042e-03, 1.232e-05])
    _slopes_radba_gaussx_low_high = PropertyAttribute('slopes_radba_gaussx_low_high',
                                                      [6.3184e-04, 5.2199e-04])
    _slopes_radba_gaussy_low_high = PropertyAttribute('slopes_radba_gaussy_low_high',
                                                      [3.6243e-03, -1.160e-05])

    # For correction when the ion is outside the ROI:
    _slopes_rf_high_x_y_ax_rough = PropertyAttribute('_slopes_rf_high_x_y_ax_rough',
                                                     [590.1e-9, 918.7e-9, -1.160e-05])
    _pos_x_y_ax_rough = PropertyAttribute('_pos_x_y_ax_rough',
                                          [1.4208e-3, 2.0300e-3, 3.6963e-3])

    _sigma_min_max_rf_low_x = PropertyAttribute('_sigma_min_max_rf_low_x', [1.0e-7, 0.02])
    _sigma_min_max_rf_high_x = PropertyAttribute('_sigma_min_max_rf_high_x', [1.0e-7, 0.02])
    _sigma_min_max_rf_low_y = PropertyAttribute('_sigma_min_max_rf_low_y', [1.0e-7, 7e-6])
    _sigma_min_max_rf_high_y = PropertyAttribute('_sigma_min_max_rf_high_y', [1.0e-7, 7e-6])
    _sigma_min_max_rf_low_ax = PropertyAttribute('_sigma_min_max_rf_low_ax', [1.0e-7, 0.02])
    _sigma_min_max_rf_high_ax = PropertyAttribute('_sigma_min_max_rf_high_ax', [1.0e-7, 0.02])

    _check_sigma_rf_low_x = PropertyAttribute('_check_sigma_rf_low_x', True)
    _check_sigma_rf_high_x = PropertyAttribute('_check_sigma_rf_high_x', True)
    _check_sigma_rf_low_y = PropertyAttribute('_check_sigma_rf_low_y', True)
    _check_sigma_rf_high_y = PropertyAttribute('_check_sigma_rf_high_y', True)
    _check_sigma_rf_low_ax = PropertyAttribute('_check_sigma_rf_low_ax', True)
    _check_sigma_rf_high_ax = PropertyAttribute('_check_sigma_rf_high_ax', True)

    _A0_min_max_rf_low_x = PropertyAttribute('_A0_min_max_rf_low_x', [0.0001, 0.08])
    _A0_min_max_rf_high_x = PropertyAttribute('_A0_min_max_rf_high_x', [0.0001, 0.08])
    _A0_min_max_rf_low_y = PropertyAttribute('_A0_min_max_rf_low_y', [0.0001, 0.08])
    _A0_min_max_rf_high_y = PropertyAttribute('_A0_min_max_rf_high_y', [0.0001, 0.08])
    _A0_min_max_rf_low_ax = PropertyAttribute('_A0_min_max_rf_low_ax', [0.0001, 0.08])
    _A0_min_max_rf_high_ax = PropertyAttribute('_A0_min_max_rf_high_ax', [0.0001, 0.08])

    _check_A0_rf_low_x = PropertyAttribute('_check_A0_rf_low_x', True)
    _check_A0_rf_high_x = PropertyAttribute('_check_A0_rf_high_x', True)
    _check_A0_rf_low_y = PropertyAttribute('_check_A0_rf_low_y', True)
    _check_A0_rf_high_y = PropertyAttribute('_check_A0_rf_high_y', True)
    _check_A0_rf_low_ax = PropertyAttribute('_check_A0_rf_low_ax', True)
    _check_A0_rf_high_ax = PropertyAttribute('_check_A0_rf_high_ax', True)

    _pos_min_max_rf_low_x = PropertyAttribute('_pos_min_max_rf_low_x', [1.35e-3, 1.5e-3])
    _pos_min_max_rf_high_x = PropertyAttribute('_pos_min_max_rf_high_x', [1.35e-3, 1.5e-3])
    _pos_min_max_rf_low_y = PropertyAttribute('_pos_min_max_rf_low_y', [1.95e-3, 2.05e-3])
    _pos_min_max_rf_high_y = PropertyAttribute('_pos_min_max_rf_high_y', [1.95e-3, 2.05e-3])
    _pos_min_max_rf_low_ax = PropertyAttribute('_pos_min_max_rf_low_ax', [3.6e-3, 3.8e-3])
    _pos_min_max_rf_high_ax = PropertyAttribute('_pos_min_max_rf_high_ax', [3.6e-3, 3.8e-3])

    _check_pos_rf_low_x = PropertyAttribute('_check_pos_rf_low_x', True)
    _check_pos_rf_high_x = PropertyAttribute('_check_pos_rf_high_x', True)
    _check_pos_rf_low_y = PropertyAttribute('_check_pos_rf_low_y', True)
    _check_pos_rf_high_y = PropertyAttribute('_check_pos_rf_high_y', True)
    _check_pos_rf_low_ax = PropertyAttribute('_check_pos_rf_low_ax', True)
    _check_pos_rf_high_ax = PropertyAttribute('_check_pos_rf_high_ax', True)

    _check_converged_rf_low_x = PropertyAttribute('_check_converged_rf_low_x', True)
    _check_converged_rf_high_x = PropertyAttribute('_check_converged_rf_high_x', True)
    _check_converged_rf_low_y = PropertyAttribute('_check_converged_rf_low_y', True)
    _check_converged_rf_high_y = PropertyAttribute('_check_converged_rf_high_y', True)
    _check_converged_rf_low_ax = PropertyAttribute('_check_converged_rf_low_ax', True)
    _check_converged_rf_high_ax = PropertyAttribute('_check_converged_rf_high_ax', True)

    _fix_axial_pos_by_absolute_position = PropertyAttribute('_fix_axial_pos_by_absolute_position', True)
    _pos_ax_set = PropertyAttribute('_pos_ax_set', 3.6963e-3)
    _set_upper_limit_auto_correction = PropertyAttribute('_set_upper_limit_auto_correction', False)
    _upper_limit_auto_correction_V_per_m = PropertyAttribute('_upper_limit_auto_correction_V_per_m', 30e-3)

    _trap_offset_x_y_ax = PropertyAttribute('_trap_offsets', [0.0, 0.01, 8.0])

    def __init__(self, name):
        self._props = Properties(name)
        self.dataq = DataClient(name.split('/')[-1])
        self.dataq.subscribe(['Experiment.start'])
        for ds in self._datastreams:
            self.dataq.subscribe([ds])
            if self._verbose_output:
                print('stray_field_check.py: subscribed to', ds)

        self.name = name
        self._reset()

    def _reset(self):
        self._init_vars()
        self.last_2_radba_images = [datetime.now() - timedelta(seconds=2), datetime.now()]
        self.last_2_axial_images = [datetime.now() - timedelta(seconds=2), datetime.now()]
        self.axial_images_missing = False
        self.last_successful_compensation = datetime.now()
        self.last_3_failed_compensations = [datetime.now() for _ in range(3)]  # not for missing axial images
        self.dm = [np.diff(self._slopes_ax_gaussx_low_high)[0],
                   np.diff(self._slopes_ax_gaussy_low_high)[0] if not self._use_radba_for_y
                   else np.diff(self._slopes_radba_gaussy_low_high)[0], np.diff(self._slopes_radba_gaussx_low_high)[0]]
        self.error = False

        self.trap_x_offs0 = self._props.get('/Experiments/Defaults/trap_x_offs', self._trap_offset_x_y_ax[0])
        self.trap_y_offs0 = self._props.get('/Experiments/Defaults/trap_y_offs', self._trap_offset_x_y_ax[1])
        self.trap_ax_offs0 = self._props.get('/Experiments/Defaults/trap_ax_offs', self._trap_offset_x_y_ax[2])
        self.trap_offsets = [self.trap_x_offs0, self.trap_y_offs0, self.trap_ax_offs0]

    def _init_vars(self):
        if self._verbose_output:
            print_error('stray_field_check.py: Initializing...', 'warning')
        self.results = pd.DataFrame(columns=['task', 'rep', 'run', 'head', 'rf', 'pos_x_y_ax',
                                             'pos_x_y_ax_no_ROI', 'timestamp'])

    def run(self):
        while True:
            msg = self.dataq.recv()
            # print(msg)
            task = 0

            if msg is not None:
                if len(msg) == 2:
                    _, experiment = msg
                    if experiment['_name'] == 'compensation_helper' and experiment['use_trap_x_y_ax_default']:
                        v= ''
                        if experiment['rf_set'] == self._rf_low:
                            rf = 'low'
                        elif experiment['rf_set'] == self._rf_high:
                            rf = 'high'
                        if rf != '':
                            self.results = pd.concat([self.results,
                                                      pd.DataFrame([{'task': experiment['_task'],
                                                                     'rep': experiment['_repetition'],
                                                                     'run': experiment['_run'], 'head': experiment,
                                                                     'rf': rf, 'pos_x_y_ax': [None, None, None],
                                                                     'pos_x_y_ax_no_ROI': [None, None, None],
                                                                     'timestamp': None}])],
                                                     ignore_index=True).reset_index(drop=True)
                            task = experiment['_task']
                            if self._verbose_output:
                                print_error('stray_field_check.py: Got experiment trigger for rf {0}.'.format(rf), 'weak')
                    else:
                        self._init_vars()
                elif len(msg) == 3:
                    msgstr, head, _ = msg
                    task = head['_task']
                    row = ((self.results['task'] == head['_task']) & (self.results['rep'] == head['_repetition'])
                           & (self.results['run'] == head['_run']))
                    if row.any():
                        pos_x_y_ax = self.results.loc[row, 'pos_x_y_ax'].values[0]
                        pos_x_y_ax_no_ROI = self.results.loc[row, 'pos_x_y_ax_no_ROI'].values[0]
                        index = None
                        if msgstr == 'axcolsumcut.axgaussx':
                            index = 0
                        elif msgstr == 'ax_yrowsumcut.axgaussy' and not self._use_radba_for_y:
                            index = 1
                        elif msgstr == 'radba_yrowsumcut.radba_gaussy' and self._use_radba_for_y:
                            index = 1
                        elif msgstr == 'radba_xcolsumcut.radba_gaussx_skewed':
                            index = 2
                        if index is not None:
                            if index == 2:  # only use x, since y can be done by both cameras
                                old = self.last_2_radba_images[1]
                                new = datetime.now()
                                self.last_2_radba_images = [old, new]

                                # if the current radba image is close to the last one, there should have
                                # been enough time for the axial image
                                radba_diff = np.diff(self.last_2_radba_images)[0].total_seconds()
                                ax_diff = (new - self.last_2_axial_images[1]).total_seconds()
                                if radba_diff < 30 and ax_diff > 120 and not self.axial_images_missing:
                                    self.axial_images_missing = True
                                    self._alert('Axial camera images missing!')
                                # else:
                                #     self._props.set('/Cameras/auto_force_ip', False)
                            elif index == 0:
                                old = self.last_2_axial_images[1]
                                new = datetime.now()
                                self.last_2_axial_images = [old, new]
                                self.axial_images_missing = False
                            if self._verbose_output:
                                print_error('stray_field_check.py: Got camera image for {0}.'.format(msgstr), 'weak')
                            if self._ion_present(head, msgstr, self.results.loc[row, 'rf'].values):
                                pos_x_y_ax[index] = head['pos']
                                self.results.loc[row, 'pos_x_y_ax'] = pd.array([pos_x_y_ax])
                                self.results.loc[row, 'timestamp'] = head['timestamp']
                            else:
                                print_error('stray_field_check.py: No ion present in image for {0}. Ignoring fit'
                                            ' ...'.format(msgstr), 'warning')
                        else:
                            if msgstr == 'axba_allcolsumcut.axba_all_gauss_x':
                                index = 0
                            elif msgstr == 'radba_allrowsumcut.radba_all_gauss_y':
                                index = 1
                            elif msgstr == 'radba_allcolsumcut.radba_all_gauss_x':
                                index = 2
                            if index is not None:
                                if self._ion_present(head, msgstr, self.results.loc[row, 'rf'].values, no_roi=True):
                                    pos_x_y_ax_no_ROI[index] = head['pos']
                                    self.results.loc[row, 'pos_x_y_ax_no_ROI'] = pd.array([pos_x_y_ax_no_ROI])
                                    self.results.loc[row, 'timestamp'] = head['timestamp']
                    else:  # No clear distinction possible, reset.
                        self._init_vars()
                        if self._verbose_output:
                            print_error('stray_field_check.py: No experiment found for image.', 'warning')
                else:
                    print_error('stray_field_check.py: Got invalid message length {0}'.format(len(msg)), 'error')

                self.results = self.results[self.results['task'] >=
                                            task - max(20, self._binning * 10)].reset_index(drop=True)

                # once all data received, compute stray-field driven position shifts and send:
                self._evaluate_and_send()

    def _evaluate_and_send(self):
        try:
            self.results['complete'] = [(np.array(self.results.loc[k, 'pos_x_y_ax']) != None).all()
                                        for k in range(len(self.results))]
        except Exception as e:
            print_error('stray_field_check: Error while updating the column \'complete\':\n\n{0}\n\n{1}'.format(e, self.results), 'error')
            for k in range(len(self.results)):
                print(k)
                print(self.results.loc[k, 'pos_x_y_ax'])
                print(np.array(self.results.loc[k, 'pos_x_y_ax']))
                print(np.array(self.results.loc[k, 'pos_x_y_ax']) != None)
                print((np.array(self.results.loc[k, 'pos_x_y_ax']) != None).all())

        self.results['complete'] = [(np.array(self.results.loc[k, 'pos_x_y_ax']) != None).all()
                                    for k in range(len(self.results))]
        if self._verbose_output:
            print_error('\n\nstray_field_check.py: Results so far\n{0}.'.format(self.results), 'info')
            print_error('\nstray_field_check.py: pos_x_y_ax so far\n{0}.'.format(self.results['pos_x_y_ax']), 'info')
            print_error('\nstray_field_check.py: pos_x_y_ax_no_ROI so far\n{0}.'.format(self.results['pos_x_y_ax_no_ROI']), 'info')

        data_low = self.results[(self.results['complete']) * (self.results['rf'] == 'low')]
        data_high = self.results[(self.results['complete']) * (self.results['rf'] == 'high')]
        pos_ax = 0.
        if len(data_low) >= self._binning and len(data_high) >= self._binning:
            dpos = np.array([0., 0., 0.])
            for enum, vals in enumerate([data_low['pos_x_y_ax'].values, data_high['pos_x_y_ax'].values]):
                pos = np.array([0., 0., 0.])
                num = 0.
                for v in vals:
                    pos += np.array(v)
                    num += 1
                pos /= num
                dpos += (1 if enum == 0 else -1) * pos
                pos_ax += (1 if enum == 0 else 0) * pos

            pos_ax = pos_ax[2]

            head = self.results[self.results['complete']]['head'].values[0]
            head['timestamp'] = max(self.results[self.results['complete']]['timestamp'].values)

            stray_fields = [None, None, None]
            for enum, ax in enumerate(['x', 'y', 'ax']):
                head['position_shift.{0}'.format(ax)] = dpos[enum]
                stray_fields[enum] = dpos[enum] * self._conversion_factors[enum]
                head['stray_field.{0}'.format(ax)] = stray_fields[enum]
            self.dataq.send(head)
            if self._verbose_output:
                print_error('stray_field_check.py: Sending \n{0}.'.format(head), 'success')
            self.results = self.results.drop(self.results[self.results['complete']].index).reset_index(drop=True)

            msg_mm = 'stray-field compensation:'
            send_mm = 0

            if self._auto_correct:
                for enum, ax in enumerate(['x', 'y', 'ax']):
                    V_defaults = self._props.get('/Experiments/Defaults/trap_' + ax)
                    V_new = V_defaults + dpos[enum] / self.dm[enum]
                    if enum == 2 and self._fix_axial_pos_by_absolute_position:
                        V_new = V_defaults - (pos_ax - self._pos_ax_set) / self._slopes_radba_gaussx_low_high[0]
                    if (not self._set_upper_limit_auto_correction or
                            np.abs(stray_fields[enum]) < self._upper_limit_auto_correction_V_per_m):
                        if self._verbose_output:
                            print_error('stray_field_check.py - _evaluate_and_send(): Changing trap_{0} from '
                                        '{1} V to {2} V.'.format(ax, V_defaults, V_new), 'weak')
                        self._props.set('/Experiments/Defaults/trap_' + ax, V_new)

                        offset = self.trap_offsets[enum]
                        if ax == 'x':
                            self.trap_x1, self.trap_x2 = Trap_Shim.shim(shim_name='trap_' + ax, setval=V_new,
                                                                        offset=offset)
                        elif ax == 'y':
                            self.trap_top, self.trap_bottom = Trap_Shim.shim(shim_name='trap_' + ax, setval=V_new,
                                                                             offset=offset)
                        elif ax == 'ax':
                            self.trap_ax_pc, self.trap_ax_wall = Trap_Shim.shim(shim_name='trap_' + ax, setval=V_new,
                                                                             offset=offset)

                        msg = ('stray_field_check.py: Changing \033[1mtrap_{0}\033[22m to {3} by'
                               ' \033[1m{1} mV\033[22m to compensate for a stray field of \033[1m{2} mV/m\033[22m.'
                               .format(ax, np.round((V_defaults - V_new) * 1e3, 1),
                                       np.round(1e3 * stray_fields[enum], 1), np.round(V_new, 4)))
                        print_error(msg, 'weak')
                        send_mm += 1
                        msg_mm += '\n{0}:\tE = {1} mV/m, dV = {2} mV, V_new = {3} V'.format(
                            ax, np.round(1e3 * stray_fields[enum]),
                            np.round((V_defaults - V_new) * 1e3, 1), np.round(V_new, 4))
                if send_mm == 3:
                    post_to_mattermost(msg=msg_mm, channel_name="pytweezer-experiment-status")

            self.last_successful_compensation = datetime.now()
            self.error = False
        elif self.error:
            pos_x_y_ax_no_ROI = [None, None, None]
            counts = [0, 0, 0]
            for vals in self.results['pos_x_y_ax_no_ROI'].values:
                for enum, val in enumerate(vals):
                    if val is not None:
                        counts[enum] += 1
                        if pos_x_y_ax_no_ROI[enum] is None:
                            pos_x_y_ax_no_ROI[enum] = 0.
                        pos_x_y_ax_no_ROI[enum] += val

            for enum, ax in enumerate(['x', 'y', 'ax']):
                if counts[enum] >= 2:
                    pos = pos_x_y_ax_no_ROI[enum] / counts[enum]
                    V_defaults = self._props.get('/Experiments/Defaults/trap_' + ax)
                    V_new = V_defaults - (pos - self._pos_x_y_ax_rough[enum]) / self._slopes_rf_high_x_y_ax_rough[enum]
                    if np.abs(V_new - V_defaults) < 1:  # V
                        print_error('stray_field_check.py - _evaluate_and_send(): Changing trap_{0} from '
                                    '{1} V to {2} V without ions in the ROI.'.format(ax, V_defaults, V_new),
                                    'error')
                        self._props.set('/Experiments/Defaults/trap_' + ax, V_new)
                    else:
                        print_error('stray_field_check.py - _evaluate_and_send(): Won\'t change trap_{0} from '
                                    '{1} V to {2} V without ions in the ROI.'.format(ax, V_defaults, V_new),
                                    'error')

    def _ion_present(self, head, msgstr, rf, no_roi=False):
        pos = head['pos']
        sigma = head['sigma']
        A0 = head['A0']
        converged = head['converged']

        filtered_out = False

        if no_roi:
            if rf == 'high':
                filtered_out = not self._check_bool(converged, 'converged ({0}, rf high)'.format(msgstr))
            else:
                return False
        else:
            if rf == 'low':
                if 'radba' in msgstr and self._check_A0_rf_low_ax and not filtered_out:
                    filtered_out = not self._check_limits(A0, self._A0_min_max_rf_low_ax, 'A0 (ax, rf low)')
                if 'radba' in msgstr and self._check_pos_rf_low_ax and not filtered_out:
                    filtered_out = not self._check_limits(pos, self._pos_min_max_rf_low_ax, 'pos (ax, rf low)')
                if 'radba' in msgstr and self._check_converged_rf_low_ax and not filtered_out:
                    filtered_out = not self._check_bool(converged, 'converged (ax, rf low)')
                if 'radba' in msgstr and self._check_sigma_rf_low_ax and not filtered_out:
                    filtered_out = not self._check_limits(sigma, self._sigma_min_max_rf_low_ax, 'sigma (ax, rf low)')
                if 'axgaussx' in msgstr and self._check_A0_rf_low_x and not filtered_out:
                    filtered_out = not self._check_limits(A0, self._A0_min_max_rf_low_x, 'A0 (x, rf low)')
                if 'axgaussx' in msgstr and self._check_pos_rf_low_x and not filtered_out:
                    filtered_out = not self._check_limits(pos, self._pos_min_max_rf_low_x, 'pos (x, rf low)')
                if 'axgaussx' in msgstr and self._check_converged_rf_low_x and not filtered_out:
                    filtered_out = not self._check_bool(converged, 'converged (x, rf low)')
                if 'axgaussx' in msgstr and self._check_sigma_rf_low_x and not filtered_out:
                    filtered_out = not self._check_limits(sigma, self._sigma_min_max_rf_low_x, 'sigma (x, rf low)')
                if 'axgaussy' in msgstr and self._check_A0_rf_low_y and not filtered_out:
                    filtered_out = not self._check_limits(A0, self._A0_min_max_rf_low_y, 'A0 (y, rf low)')
                if 'axgaussy' in msgstr and self._check_pos_rf_low_y and not filtered_out:
                    filtered_out = not self._check_limits(pos, self._pos_min_max_rf_low_y, 'pos (y, rf low)')
                if 'axgaussy' in msgstr and self._check_converged_rf_low_y and not filtered_out:
                    filtered_out = not self._check_bool(converged, 'converged (y, rf low)')
                if 'axgaussy' in msgstr and self._check_sigma_rf_low_y and not filtered_out:
                    filtered_out = not self._check_limits(sigma, self._sigma_min_max_rf_low_y, 'sigma (y, rf low)')
            elif rf == 'high':
                if 'radba' in msgstr and self._check_A0_rf_high_ax and not filtered_out:
                    filtered_out = not self._check_limits(A0, self._A0_min_max_rf_high_ax, 'A0 (ax, rf high)')
                if 'radba' in msgstr and self._check_pos_rf_high_ax and not filtered_out:
                    filtered_out = not self._check_limits(pos, self._pos_min_max_rf_high_ax, 'pos (ax, rf high)')
                if 'radba' in msgstr and self._check_converged_rf_high_ax and not filtered_out:
                    filtered_out = not self._check_bool(converged, 'converged (ax, rf high)')
                if 'radba' in msgstr and self._check_sigma_rf_high_ax and not filtered_out:
                    filtered_out = not self._check_limits(sigma, self._sigma_min_max_rf_high_ax, 'sigma (ax, rf high)')
                if 'axgaussx' in msgstr and self._check_A0_rf_high_x and not filtered_out:
                    filtered_out = not self._check_limits(A0, self._A0_min_max_rf_high_x, 'A0 (x, rf high)')
                if 'axgaussx' in msgstr and self._check_pos_rf_high_x and not filtered_out:
                    filtered_out = not self._check_limits(pos, self._pos_min_max_rf_high_x, 'pos (x, rf high)')
                if 'axgaussx' in msgstr and self._check_converged_rf_high_x and not filtered_out:
                    filtered_out = not self._check_bool(converged, 'converged (x, rf high)')
                if 'axgaussx' in msgstr and self._check_sigma_rf_high_x and not filtered_out:
                    filtered_out = not self._check_limits(sigma, self._sigma_min_max_rf_high_x, 'sigma (x, rf high)')
                if 'axgaussy' in msgstr and self._check_A0_rf_high_y and not filtered_out:
                    filtered_out = not self._check_limits(A0, self._A0_min_max_rf_high_y, 'A0 (y, rf high)')
                if 'axgaussy' in msgstr and self._check_pos_rf_high_y and not filtered_out:
                    filtered_out = not self._check_limits(pos, self._pos_min_max_rf_high_y, 'pos (y, rf high)')
                if 'axgaussy' in msgstr and self._check_converged_rf_high_y and not filtered_out:
                    filtered_out = not self._check_bool(converged, 'converged (y, rf high)')
                if 'axgaussy' in msgstr and self._check_sigma_rf_high_y and not filtered_out:
                    filtered_out = not self._check_limits(sigma, self._sigma_min_max_rf_high_y, 'sigma (y, rf high)')

        if filtered_out and not no_roi:
            now = datetime.now()
            if (now - self.last_3_failed_compensations[-1]).total_seconds() > 50:  # in case several images within one compensation are missing ions
                if self.last_successful_compensation < self.last_3_failed_compensations[0]:
                    self._alert('Automatic stray-field compensation not possible!')
                    self.error = True
                self.last_3_failed_compensations[:-1] = self.last_3_failed_compensations[1:]
                self.last_3_failed_compensations[-1] = datetime.now()  # not for missing axial images

        return not filtered_out

    def _check_limits(self, value, bounds, msg):
        inside = max(bounds) >= value >= min(bounds)

        if not inside:  #  and self._verbose_output:
            print_error('stray_field_check.py: Filtered out image by {0}: {1} >= {2} >= {3}.'.format(msg, max(bounds), value, min(bounds)), 'warning')

        return inside

    def _check_bool(self, value, msg):
        inside = value

        if self._verbose_output and not inside:
            print_error('stray_field_check.py: Filtered out image by {0}: {1}.'.format(msg, value), 'warning')

        return inside

    def _alert(self, msg):
        print_error('StrayFieldCheck: {0}'.format(msg), 'error')
        msg_mm = msg
        if 'images missing!' in msg:
            msg_mm = ('Alert: {0}\nPause the ExperimentQueue,\nterminate the camera server,\n'
                      'force IP by SpinView,\nstart first 3 cameras by camerahub,\nstart ExperimentQ.').format(msg)
            self.last_2_axial_images = [datetime.now() - timedelta(seconds=2), datetime.now()]
            self._props.set('/Cameras/auto_force_ip', True)
        elif 'compensation not possible!' in msg:
            msg_mm = ('Alert: {0}\nPause the ExperimentQueue,\ncheck the Ba lasers,\n'
                      'compensate for stray-fields manually,\nstart ExperimentQ.').format(msg)
            self.last_successful_compensation = datetime.now()
        post_to_mattermost(msg=msg_mm, channel_name="lab-status")


def main_run(name):
    slc = StrayFieldCheck(name)
    slc.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs=1, help='name of this program instance')
    args = parser.parse_args()
    name = args.name[0]
    main_run(name)
