import pathlib
import subprocess

from PyQt5 import QtCore
from PyQt5.QtCore import QDateTime, QThread, pyqtSignal
import numpy as np
import time
import threading

from pytweezer.analysis.print_messages import print_error
from pytweezer.experiment.experiment import get_experiment
from pytweezer.experiment.motmaster_client import MotMasterClient
from pytweezer.logging_utils import get_logger
from pytweezer.servers.model_sync import SyncedScheduleModel
from pytweezer.servers.properties import Properties


logger = get_logger(__name__)


class QueueThread(QThread):
    """Simple QThread subclass to run the queue worker."""
    update_ui = pyqtSignal()
    delete_item = pyqtSignal(int)
    table_update = pyqtSignal()
    
    def __init__(self, queue_fn):
        super().__init__()
        self.queue_fn = queue_fn
    
    def run(self):
        self.queue_fn()


class ExperimentManager:
    """
    Contains the functions for running experiments from the ExperimentQ.
    The queue_fn is constantly running inside a worker thread checking the synced
    schedule model for experiments to run.
    Experiments consist of:
        *   measurement: one or more repetitions of

            * sequence (or scan):  multiple runs with varied parameters

                *   run:  one single run

    Before and after each measurement, start_ and end_measurement are called.
    Before and after each sequence, start_ and end_sequence are called.
    By default, and a measurement is a single repetition of a sequence with a single run.
    """
    def __init__(self):
        self._props = Properties("Manager")
        self._results_h5_dir = ""
        self.queue_thread = None
        self.queue_running = False
        self._queue_condition = threading.Condition()
        self.paused = False
        self.running = False  # tells the queue whether an experiment is currently running
        self.taskNr = 0
        self._measurement_start_time = None
        self._measurement_results = {}
        self.motmaster_client = MotMasterClient()
        # Create own instance of synced model for standalone operation
        self._model = SyncedScheduleModel()
        self.start_queue()

    @property
    def model(self):
        """Returns the synced schedule model (standalone instance)."""
        return self._model

    @property
    def exp_dict(self):
        """Returns the underlying experiment dictionary from the synced model."""
        return self.model.backing_store

    def run(self, taskNr, progress_callback=None):
        """
        Handles the running of experiment tasks
        """
        if taskNr not in self.exp_dict.keys():  # in case the task is deleted before it can be run
            print_error('Task {} no longer exists'.format(taskNr), 'error')
            return
        task_dict = self.exp_dict[taskNr]
        self.taskNr = taskNr
        self.name = task_dict['expName'] + ' ' + task_dict['label']
        # self.experiment = task_dict['experiment']
        filepath = task_dict['filepath']
        experiment_cls = get_experiment(filepath, task_dict['expName'])
        self.experiment = experiment_cls(self._props, motmaster_client=self.motmaster_client)
        self.arg_dict = task_dict['args']
        self.n_runs = task_dict['nRuns']
        self.scan_pars = task_dict['scanpars']
        self.scan_values = np.array(task_dict['scanvals'])
        self.n_reps = task_dict['nReps']
        self.scan_sequence = task_dict['scansequence']
        self.terminated = task_dict['terminated']
        self.rep = task_dict['repetition']
        self.run_nr = task_dict['run']
        self.experiment.expName = task_dict['expName']
        self._props.set('ExperimentQ/current_scan', self.scan_pars)

        self.start_measurement()  # pass argdict to the experiment
        """Repeat the experiment nRep times."""
        while self.rep <= self.n_reps and self.running:
            self.set_rep_nr(self.rep)
            self._sequence_run()  # exp run in here. also handles scans
            self.rep += 1
            if self.terminated:
                self.terminate_experiment()
                return
        if taskNr not in self.exp_dict.keys():  # in case the task is deleted while running
            print_error('Task {} no longer exists'.format(taskNr), 'error')
            return
        self.end_measurement()

    def _sequence_run(self):
        """Initialize runloop and start experiments."""
        # self.experiment.start_sequence()
        self._runloop()
        # self.experiment.end_sequence()

    def _runloop(self):
        """
        Runs the experiment (or experiment scan).
        Checks for experiment pause or termination before each run
        """
        if self.taskNr not in self.exp_dict.keys():  # in case the task is deleted while running
            print_error('Task {} no longer exists'.format(self.taskNr), 'error')
            return
        while self.run_nr < self.n_runs and self.running:
            self.terminated = self.exp_dict[self.taskNr]['terminated']
            while self._is_task_paused(self.taskNr) and not self.terminated:
                # Reflect pause in manager state for legacy code paths.
                self.paused = True
                self._wait_for_queue_event(0.2)
                self.terminated = self.exp_dict[self.taskNr]['terminated']
            self.paused = False
            if self.terminated:
                return  # pass back to the main loop to terminate the task
            if self.n_runs > 1:
                self._update_task_status(self.taskNr, 'Scanning')
            else:
                self._update_task_status(self.taskNr, 'Running')

            for i, parname in enumerate(self.scan_pars):  # modify value(s) for scan
                if parname != '--NONE--':
                    self.set_dict(parname, self.scan_values[self.scan_sequence[self.run_nr], i])

            self.set_run_nr(self.run_nr)
            # time.sleep(0.5)
            self.experiment._start_run(self.taskNr)
            self._collect_run_results(self.rep, self.run_nr)
            self.run_nr += 1
        self.run_nr = 0

    def pause(self):
        """Toggle pause by editing current task status in the synced model."""
        if not self.running or self.taskNr not in self.exp_dict:
            return
        if self._is_task_paused(self.taskNr):
            self._update_task_status(self.taskNr, 'Queued')
            self.paused = False
        else:
            self._update_task_status(self.taskNr, 'Paused')
            self.paused = True
        self._notify_queue_event()

    def _is_task_paused(self, taskNr):
        """Return True if the given task is marked paused in the shared model."""
        if taskNr not in self.exp_dict:
            return False
        return self.exp_dict[taskNr].get('status') == 'Paused'

    def start_queue(self):
        """Starts the experiment queue worker in its own thread."""
        logger.info('Queue worker starting')
        self.queue_running = True
        self.queue_thread = QueueThread(self.queue_fn)
        self.queue_thread.update_ui.connect(self.update_table)
        self.queue_thread.delete_item.connect(self.delete_item)
        self.queue_thread.start()
        
    def delete_item(self, k):
        del self.model[k]

    def finished(self):
        logger.info('Queue worker finished')
        self.queue_running = False
        self._notify_queue_event()

    def _notify_queue_event(self):
        """Wake the queue worker when queue state changes."""
        with self._queue_condition:
            self._queue_condition.notify_all()

    def _wait_for_queue_event(self, timeout_s=0.5):
        """Wait for queue-relevant state changes with timeout fallback."""
        with self._queue_condition:
            self._queue_condition.wait(timeout=timeout_s)


    def queue_fn(self, progress_callback=None):
        """
        Experiment queue runs continuously checking the experiment dictionary for new tasks.
        If the dictionary is empty or there is an experiment running, it does nothing.
        If the are tasks, is looks through them in order of 1) priority and 2) task number.
        If a task is marked as 'waiting', it moves on to the next task. Otherwise, it runs the task.
        """
        while self.queue_running:
            i = 0
            while self.queue_running and (self.exp_dict == {} or self.running):
                self._wait_for_queue_event(0.5)
            if not self.queue_running:
                break
            while i < len(self.exp_dict.keys()):
                try:
                    try:
                        nextTaskNr = self.model.row_to_key[i] 
                    except Exception as e:
                        print_error('ExperimentManager: IndexError in queue_fn, likely due to task deletion during iteration', 'error')
                        logger.exception('Exception while selecting next task in queue loop')
                        break
                    nextTask = self.exp_dict[nextTaskNr]
                    # Only check due_check for tasks that are sleeping/waiting;
                    # never run failed/done/terminating tasks.
                    should_run = (
                        nextTask['status'] not in ('Sleeping', 'Failed', 'Waiting', 'Done', 'Terminating')
                        or (nextTask['status'] in ('Sleeping', 'Waiting') and self.due_check(nextTask))
                    )
                    if should_run:
                        self.run(nextTaskNr)
                        i = 0
                    else:
                        i += 1
                    if i == len(self.exp_dict.keys()):
                        # If all tasks are waiting/sleeping, avoid a tight poll loop.
                        self._wait_for_queue_event(0.5)
                except Exception as e:
                    print_error('Task {} failed with error: {}'.format(nextTaskNr, e), 'error')
                    self.running = False
                    self._update_task_status(nextTaskNr, 'Failed')
                    i += 1

    @staticmethod
    def due_check(task_dict):
        if 'dueDateTime' not in task_dict.keys():
            return True
        else:
            currentDateTime = QDateTime.currentDateTime()
            dueDateTime = QDateTime.fromString(task_dict['dueDateTime'])
            if currentDateTime >= dueDateTime:
                return True
            else:
                return False

    def update_table(self):
        """Sometimes we need to tell the GUI to update the table display."""
        self.model.layoutChanged.emit()

    def _update_task_status(self, taskNr, status):
        """
        Updates task status through the synced model to ensure proper synchronization.
        """
        if taskNr in self.exp_dict:
            self.exp_dict[taskNr]['status'] = status
            # Use RPC to notify the server of the status change
            self.model._sync.apply_operation(
                {
                    "command": "set_data",
                    "row_or_key": taskNr,
                    "field": "status",
                    "value": status,
                }
            )
            self.update_table()
            self._notify_queue_event()

    def terminate_experiment(self):
        """Graceful termination of the experiment. Allows current run to finish."""
        logger.info('Terminating experiment')
        self._update_task_status(self.taskNr, 'Terminating')
        self.running = False
        self._notify_queue_event()
        time.sleep(0.1)  # just to give the table time to display. Can be shortened or taken out.
        self.delete_item(self.taskNr)

    def start_measurement(self):
        """Sets arguments, calls experimental start_measurement function."""
        self.experiment.build()
        # Manager owns HDF5 persistence at measurement level.
        self.experiment._save_results_h5 = False
        # Keep runtime identifiers on the experiment instance in sync with queue state.
        self.experiment._task = self.taskNr
        self.experiment._run = int(self.run_nr)
        self.experiment._rep = int(self.rep)
        self._measurement_start_time = time.time()
        self._measurement_results = {}
        for arg in self.arg_dict.keys():
            self.set_dict(arg, self.arg_dict[arg])
        # print('\n\033[1mExperimentQ: task {} ({}) starting\n\033[0m'.format(self.taskNr, self.name))
        print_error('ExperimentQ: task {} ({}) starting'.format(self.taskNr, self.name), 'bold')
        # print('_task:', self._props.get('/Experiments/_task', 0))
        self.running = True
        if self.paused:
            self._update_task_status(self.taskNr, 'Paused')
        # self.experiment.start_measurement()

    def end_measurement(self):
        """Run the end_measurement function, clean up the table"""
        self.experiment.cleanup()
        # print('\n\033[1mExperimentQ: task {} ({}) completed\n\033[0m'.format(self.taskNr, self.name))
        self._save_measurement_h5()
        print_error('ExperimentQ: task {} ({}) completed'.format(self.taskNr, self.name), 'bold')
        self._update_task_status(self.taskNr, 'Done')
        # time.sleep(0.1)  # just to give the table time to display. Can be shortened or taken out.
        del self.model[self.taskNr]
        self.running = False
        self._notify_queue_event()
        
        
    def _get_results_h5_dir(self) -> pathlib.Path:
        if isinstance(self._results_h5_dir, str) and self._results_h5_dir.strip():
            out_dir = pathlib.Path(self._results_h5_dir)
        else:
            base = self._props.get('/Servers/Imagepath', str(pathlib.Path.cwd() / 'Data'))
            out_dir = pathlib.Path(base) / 'measurements_h5'
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    @staticmethod
    def _sanitize_h5_key(key: str) -> str:
        return str(key).replace('/', '_').replace(' ', '_')

    def _write_h5_value(self, group, key: str, value):
        key = self._sanitize_h5_key(key)

        if isinstance(value, dict):
            sub = group.create_group(key)
            for k, v in value.items():
                self._write_h5_value(sub, str(k), v)
            return

        if isinstance(value, np.ndarray):
            group.create_dataset(key, data=value)
            return

        if isinstance(value, (list, tuple)):
            arr = np.asarray(value)
            if arr.dtype == object:
                group.create_dataset(key, data=np.asarray([str(v) for v in value], dtype='S'))
            else:
                group.create_dataset(key, data=arr)
            return

        if isinstance(value, (np.integer, int, np.floating, float, np.bool_, bool)):
            group.attrs[key] = value
            return

        if isinstance(value, str):
            group.attrs[key] = value
            return

        if value is None:
            group.attrs[key] = 'None'
            return

        group.attrs[key] = str(value)

    def _collect_run_results(self, rep: int, run: int):
        """Capture run result-channel payloads for end-of-measurement saving."""
        rep_map = self._measurement_results.setdefault(int(rep), {})
        run_map = rep_map.setdefault(int(run), {})
        for channel_name, channel in self.experiment.result_channels.items():
            run_map[channel_name] = channel.data

    def _save_measurement_h5(self):
        """
        saves: 
        - metadata: dict of metadata at the start of the measurement. one for whole file
            - file name (from task dict)
            - experiment name (from task dict)
            - start time (add to queue loop)
            - git info (from method)
            - arguments and mm_params (if present) (from task dict)
            - scan parameters and values (from task dict)
            - end time (add to queue loop)
        - results: dict of results taken from experiment.result_channels. one dataset per repetition, with name given by the channel name. Can be scalars or arrays. collected as the measurement is running as saved once at the end
        """
        try:
            import h5py
        except Exception as error:
            print_error(f"Could not import h5py for measurement save: {error}", 'error')
            return

        task_dict = self.exp_dict.get(self.taskNr, {})
        exp_name = task_dict.get('expName', getattr(self.experiment, 'name', 'Experiment'))
        filename = f"task{self.taskNr}_{exp_name}.h5"
        path = self._get_results_h5_dir() / filename
        end_time = time.time()

        try:
            with h5py.File(path, 'w') as f:
                f.attrs['task'] = int(self.taskNr)

                g_task_info = f.create_group('task_info')
                g_task_info.attrs['experiment_name'] = exp_name
                g_task_info.attrs['label'] = task_dict.get('label', '')
                g_task_info.attrs['filepath'] = task_dict.get('filepath', '')
                g_task_info.attrs['start_time'] = float(self._measurement_start_time or end_time)
                g_task_info.attrs['end_time'] = float(end_time)
                g_task_info.attrs['duration_s'] = float(end_time - (self._measurement_start_time or end_time))
                g_task_info.attrs['n_runs'] = int(task_dict.get('nRuns', self.n_runs))
                g_task_info.attrs['n_reps'] = int(task_dict.get('nReps', self.n_reps))

                self._write_h5_value(g_task_info, 'arguments', task_dict.get('args', {}))
                mm_params = {name: arg.get() for name, arg in self.experiment.mm_args.items()}
                self._write_h5_value(g_task_info, 'mm_params', mm_params)

                scan_info = {
                    'scan_parameters': list(task_dict.get('scanpars', [])),
                    'scan_sequence': list(task_dict.get('scansequence', [])),
                    'scan_values': np.asarray(task_dict.get('scanvals', [])),
                }
                self._write_h5_value(g_task_info, 'scan_info', scan_info)

                g_git = g_task_info.create_group('git')
                for k, v in self._get_git_repo_info().items():
                    self._write_h5_value(g_git, k, v)

                g_results = f.create_group('results')
                for rep in sorted(self._measurement_results.keys()):
                    g_rep = g_results.create_group(f"rep_{rep:04d}")
                    run_map = self._measurement_results[rep]

                    channel_values = {}
                    for run in sorted(run_map.keys()):
                        for channel_name, value in run_map[run].items():
                            channel_values.setdefault(channel_name, []).append(value)

                    for channel_name, values in channel_values.items():
                        key = self._sanitize_h5_key(channel_name)
                        try:
                            arr = np.asarray(values)
                            if arr.dtype == object:
                                g_rep.create_dataset(key, data=np.asarray([str(v) for v in values], dtype='S'))
                            else:
                                g_rep.create_dataset(key, data=arr)
                        except Exception:
                            g_rep.create_dataset(key, data=np.asarray([str(v) for v in values], dtype='S'))

            print_error(f"Saved measurement HDF5: {path}", 'success')
        except Exception as error:
            print_error(f"Failed to save measurement HDF5 at {path}: {error}", 'error')
    
    @staticmethod
    def _run_git(repo_root: pathlib.Path, args: list[str]) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                return ""
            return completed.stdout.strip()
        except Exception:
            return ""

    def _get_git_repo_info(self) -> dict:
        info = {
            "available": False,
            "repo_root": "",
            "branch": "",
            "commit": "",
            "commit_short": "",
            "dirty": False,
            "remote_origin": "",
        }

        try:
            start_path = pathlib.Path(__file__).resolve().parent
            top = self._run_git(start_path, ["rev-parse", "--show-toplevel"])
            if not top:
                return info

            repo_root = pathlib.Path(top)
            info["available"] = True
            info["repo_root"] = str(repo_root)
            info["branch"] = self._run_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
            info["commit"] = self._run_git(repo_root, ["rev-parse", "HEAD"])
            info["commit_short"] = self._run_git(repo_root, ["rev-parse", "--short", "HEAD"])
            info["remote_origin"] = self._run_git(repo_root, ["config", "--get", "remote.origin.url"])

            status = self._run_git(repo_root, ["status", "--porcelain"])
            info["dirty"] = bool(status)
        except Exception:
            return info

        return info


    def set_dict(self, name, value):
        """Sets experiment arguments."""
        logger.debug("set_dict called: %s=%r", name, value)
        self.experiment.set_argument_value(name, value)

    def set_run_nr(self, value):
        """Sets the experiment run number and updates the table."""
        self.experiment._run = value       
        self.exp_dict[self.taskNr]['run'] = value
        # Sync through RPC to ensure GUI sees the update
        self.model._sync.apply_operation(
            {
                "command": "set_data",
                "row_or_key": self.taskNr,
                "field": "run",
                "value": value,
            }
        )
        self.update_table()

    def set_rep_nr(self, value):
        """Sets the experiment repetition number and updates the table."""
        self.experiment._rep = value
        self.exp_dict[self.taskNr]['repetition'] = value
        # Sync through RPC to ensure GUI sees the update
        self.model._sync.apply_operation(
            {
                "command": "set_data",
                "row_or_key": self.taskNr,
                "field": "repetition",
                "value": value,
            }
        )
        self.update_table()

def main():
    """Run the experiment manager standalone."""
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    
    # Create and run the experiment manager
    manager = ExperimentManager()
    
    logger.info("Experiment Manager started and listening to synced schedule model")
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
