import time
from pytweezer.servers import DataClient
from pytweezer.servers import Properties, PropertyAttribute
from pytweezer.analysis.print_messages import print_error
import pandas as pd
import numpy as np
import argparse

from pytweezer.servers.mattermost_interface import post_to_mattermost


class AblationLoadingEfficiency:
    _datastreams = PropertyAttribute('datastreams', ['Experiment.start', 'Ion_Countbloblist_'])
    _binning = PropertyAttribute('binning', 100)
    _loading_threshold_efficiency_p = PropertyAttribute('_loading_threshold_efficiency_p', 20)
    _multiple_ions_threshold_p = PropertyAttribute('_multiple_ions_threshold_p', 5)
    _ablation_increase_fraction = PropertyAttribute('_ablation_increase_fraction', 0.33)
    _ablation_pulses_min_max = PropertyAttribute('_ablation_pulses_min_max', [1, 50])
    _verbose_output = PropertyAttribute('_verbose_output', False)

    def __init__(self, name):
        self._props = Properties(name)
        self.dataq = DataClient(name.split('/')[-1])
        self.dataq.subscribe(self._datastreams)
        self.name = name

        self.loading_experiments = ['00_loading_scheme', '01_optical_trapping']
        self.last_round_successful = True  # if False, don't send alerts at failure

        self.reset()

    def reset(self):
        self.count = 0
        self.results = pd.DataFrame(columns=['task', 'rep', 'run', 'experiment', 'N_bright', 'timestamp'])
        self.n_ablation_pulses = self._props.get('n_ablation_pulses', defaultvalue=15)

    def run(self):
        update_sent = False
        while True:
            msg = self.dataq.recv()
            if msg is not None:
                if len(msg) == 2:
                    _, experiment = msg
                    if experiment['_expName'] in self.loading_experiments:
                        new_exp = pd.DataFrame([{'task': experiment['_task'], 'rep': experiment['_repetition'],
                                                 'run': experiment['_run'], 'experiment': experiment['_expName'],
                                                 'N_bright': None, 'timestamp': None}])
                        self.results = pd.concat([self.results, new_exp],
                                                 ignore_index=True).reset_index(drop=True)
                        if self._verbose_output:
                            print_error('ablation_loading_efficiency.py: Got N_bright, results so far: '
                                        '{0}.'.format(self.results), 'weak')
                elif len(msg) == 3:
                    msgstr, head, _ = msg
                    row = ((self.results['task'] == head['_task']) & (self.results['rep'] == head['_repetition'])
                           & (self.results['run'] == head['_run']))
                    if row.any() and 'N_bright' in head:
                        self.results.loc[row, 'timestamp'] = head['timestamp']
                        self.results.loc[row, 'N_bright'] = head['N_bright']
                        self.count += 1
                        update_sent = False
                        if self._verbose_output:
                            print_error('ablation_loading_efficiency.py: Got N_bright, results so far: '
                                        '{0}.'.format(self.results), 'weak')
                    else:  # No clear distinction possible, reset.
                        if self._verbose_output:
                            print_error('ablation_loading_efficiency.py: No experiment found for image.',
                                        'warning')
                else:
                    print_error('ablation_loading_efficiency.py: Got invalid message length {0}'.format(len(msg)),
                                'error')

            if self.count >= self._binning:
                self._evaluate_and_send()
                update_sent = False
            elif self.count > 0 and self.count % 25 == 0 and not update_sent:
                print_error('ablation_loading_efficiency.py: Analyzing {0} / {1} loading '
                            'attempts.'.format(self.count, self._binning), 'weak')
                update_sent = True

            time.sleep(0.1)  # sec

    def _evaluate_and_send(self):
        self.results.dropna(inplace=True)

        # only count from the first occurence of loading:
        first_loading_exp = self.results[self.results['experiment'] == '00_loading_scheme'].iloc[0].name
        self.results = self.results.loc[first_loading_exp:, :].reset_index()

        # Filter out experiments between optical trapping with N=1 and loading:
        loading_cycle = True
        loading_indices = []

        for enum, row in enumerate(self.results.iterrows()):
            if not loading_cycle and row[1]['experiment'] == '00_loading_scheme':
                loading_cycle = True
            if loading_cycle:
                loading_indices = np.append(loading_indices, enum)
            if loading_cycle and row[1]['experiment'] == '01_optical_trapping' and row[1]['N_bright'] == 1:
                loading_cycle = False

        self.results = self.results.loc[loading_indices, :].reset_index()

        # eval data:
        opt_trapping_results = {}  # {0: 13, 1: 23, 2: 0, 3: 11} would mean max_ions = 3,
        # 13 % of the loading attempts resulted in 0 ions, 23 % in 1 ion, ..., 11 % in 3 ions or more [max_ions]
        loading_results = {}
        max_ions = 2  # must be >= 2

        for N_bright in range(max_ions + 1):
            opt_trapping_attempts = max(1, len(self.results[self.results['experiment'] == '01_optical_trapping']))
            loading_attempts = max(1, len(self.results[self.results['experiment'] == '00_loading_scheme']))

            opt_trapping_results[N_bright] = len(self.results[(self.results['experiment'] == '01_optical_trapping') *
                                                 (self.results['N_bright'] == N_bright if N_bright < max_ions
                                                  else self.results['N_bright'] >= N_bright)])
            loading_results[N_bright] = len(self.results[(self.results['experiment'] == '00_loading_scheme') *
                                                         (self.results['N_bright'] == N_bright if N_bright < max_ions
                                                          else self.results['N_bright'] >= N_bright)])
            opt_trapping_results[N_bright] = np.round(100. * opt_trapping_results[N_bright] / opt_trapping_attempts, 2)
            loading_results[N_bright] = np.round(100. * loading_results[N_bright] / loading_attempts, 2)

        print_error('ablation_loading_efficiency.py: Ablation loading resulted in {0} ions.'.format(loading_results),
                    'weak')
        print_error('ablation_loading_efficiency.py: Optical ion trapping resulted in {0}'
                    ' ions.'.format(opt_trapping_results), 'weak')

        n_pulses_old = self._props.get('n_ablation_pulses', defaultvalue=self.n_ablation_pulses)
        increase_ablation_pulses = 0
        too_many_ions_loaded = np.array([loading_results[dc] for dc in range(2, max_ions + 1)]).sum()
        if too_many_ions_loaded >= self._multiple_ions_threshold_p:
            increase_ablation_pulses = -1  # loaded too often multiple ions
            print_error('ablation_loading_efficiency.py: Loaded too many ions - happened {0} % of the attempts, '
                        'allowed {1} %.'.format(too_many_ions_loaded, self._multiple_ions_threshold_p),
                        'weak')
        elif loading_results[0] >= 100 - self._loading_threshold_efficiency_p:
            increase_ablation_pulses = 1  # loaded too often nothing
            print_error('ablation_loading_efficiency.py: Loaded ions too rarely - happened {0} % of the attempts, '
                        'allowed {1} %.'.format(loading_results[0], 100 - self._loading_threshold_efficiency_p),
                        'weak')
        #if opt_trapping_results[0] >= self._multiple_ions_threshold_p:
        #    increase_ablation_pulses = -1  # couldn't optically trap (multiple ions cooling in?)
        sign = np.sign(increase_ablation_pulses)
        increase_ablation_pulses = int(np.ceil(np.abs(self._ablation_increase_fraction * increase_ablation_pulses
                                               * n_pulses_old)) * sign)

        head = {'timestamp': max(self.results['timestamp'])}
        for N_bright in range(max_ions + 1):
            head['loading_{0}_ions_p'.format(int(N_bright))] = loading_results[N_bright]
            head['opt_trap_{0}_ions_p'.format(int(N_bright))] = opt_trapping_results[N_bright]
        head['increase_ablation_pulses'] = increase_ablation_pulses

        self.n_ablation_pulses = n_pulses_old + increase_ablation_pulses
        self.n_ablation_pulses = max(self.n_ablation_pulses, self._ablation_pulses_min_max[0])
        self.n_ablation_pulses = min(self.n_ablation_pulses, self._ablation_pulses_min_max[1])
        self._props.set('n_ablation_pulses', self.n_ablation_pulses)
        print_error('ablation_loading_efficiency.py: Adjusting the pulse number from {0} to {1} by'
                    ' {2}.'.format(n_pulses_old, self.n_ablation_pulses, increase_ablation_pulses), 'weak')

        self.dataq.send(head)
        success = send_to_mattermost(head, n_pulses_old, self.n_ablation_pulses, max_ions,
                                     send_alerts=self.last_round_successful)
        self.last_round_successful = success
        self.reset()


def send_to_mattermost(head, n_pulses_old, n_pulses_new, max_ions, send_alerts=True):
    msg_mm = 'ablation_loading_efficiency:'
    msg_mm += '\nAdjusting the anblation pulse number from {0} to {1},'.format(n_pulses_old, n_pulses_new)
    nothing_loaded = True
    for n_ions in range(max_ions + 1):
        if n_ions > 0 and head['loading_{0}_ions_p'.format(n_ions)] > 0:
            nothing_loaded = False
        msg_mm += ('\nLoaded {3}{0} ion{2} {1} % times'
                   .format(n_ions, np.round(head['loading_{0}_ions_p'.format(n_ions)], 2),
                           's' if n_ions != 1 else '', '>= ' if n_ions == max_ions else ''))
    for n_ions in range(max_ions + 1):
        msg_mm += ('\nOptically trapped {3}{0} ion{2} {1} % times'
                   .format(n_ions, np.round(head['opt_trap_{0}_ions_p'.format(n_ions)], 2),
                           's' if n_ions != 1 else '', '>= ' if n_ions == max_ions else ''))
    post_to_mattermost(msg=msg_mm, channel_name="pytweezer-experiment-status")

    if nothing_loaded and send_alerts:
        msg_mm = ('Alert: Ablation loading not working!\nPause the ExperimentQueue,\ncheck the Ba lasers,\n'
                  'only increase ablation power if necessary - USE power meter\n'
                  'compensate for stray-fields manually,\nstart ExperimentQ.')
        post_to_mattermost(msg=msg_mm, channel_name="lab-status")

    return not nothing_loaded


def main_run(name):
    slc = AblationLoadingEfficiency(name)
    slc.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs=1, help='name of this program instance')
    args = parser.parse_args()
    name = args.name[0]
    main_run(name)
