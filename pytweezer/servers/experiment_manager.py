from PyQt5 import QtCore
from PyQt5.QtCore import QDateTime, QThread, pyqtSignal
import numpy as np
import time

from pytweezer.analysis.print_messages import print_error
from pytweezer.experiment.experiment import get_experiment
from pytweezer.experiment.motmaster_client import MotMasterClient
from pytweezer.servers.model_sync import SyncedScheduleModel
from pytweezer.servers.properties import Properties


class QueueThread(QThread):
    """Simple QThread subclass to run the queue worker."""
    update_ui = pyqtSignal()
    deleteItem = pyqtSignal(int)
    tableUpdate = pyqtSignal()
    
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
        self.queueThread = None
        # Create own instance of synced model for standalone operation
        self._model = SyncedScheduleModel()
        self.start_queue()
        self.paused = False
        self.running = False  # tells the queue whether an experiment is currently running
        self.taskNr = 0
        self.motmaster_client = MotMasterClient()

    @property
    def model(self):
        """Returns the synced schedule model (standalone instance)."""
        return self._model

    @property
    def expDict(self):
        """Returns the underlying experiment dictionary from the synced model."""
        return self.model.backing_store

    def run(self, taskNr, progress_callback=None):
        """
        Handles the running of experiment tasks
        """
        if taskNr not in self.expDict.keys():  # in case the task is deleted before it can be run
            print_error('Task {} no longer exists'.format(taskNr), 'error')
            return
        task_dict = self.expDict[taskNr]
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
        if taskNr not in self.expDict.keys():  # in case the task is deleted while running
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
        if self.taskNr not in self.expDict.keys():  # in case the task is deleted while running
            print_error('Task {} no longer exists'.format(self.taskNr), 'error')
            return
        while self.run_nr < self.n_runs and self.running:
            self.terminated = self.expDict[self.taskNr]['terminated']
            while self.paused and not self.terminated:
                time.sleep(0.01)
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
            self.run_nr += 1
        self.run_nr = 0

    def pause(self):
        """Pauses or unpauses the experiment."""
        if self.running:
            if self.paused:
                self._update_task_status(self.taskNr, 'Running')
            else:
                self._update_task_status(self.taskNr, 'Paused')
        self.paused = not self.paused

    def start_queue(self):
        """Starts the experiment queue worker in its own thread."""
        print('queue worker starting')
        self.queueRunning = True
        self.queueThread = QueueThread(self.queue_fn)
        self.queueThread.update_ui.connect(self.update_table)
        self.queueThread.deleteItem.connect(self.delete_item)
        self.queueThread.start()
        
    def delete_item(self, k):
        del self.model[k]

    def finished(self):
        print('queue worker finished')
        self.queueRunning = False


    def queue_fn(self, progress_callback=None):
        """
        Experiment queue runs continuously checking the experiment dictionary for new tasks.
        If the dictionary is empty or there is an experiment running, it does nothing.
        If the are tasks, is looks through them in order of 1) priority and 2) task number.
        If a task is marked as 'waiting', it moves on to the next task. Otherwise, it runs the task.
        """
        while True:
            i = 0
            while self.expDict == {} or self.running:
                time.sleep(0.01)
            while i < len(self.expDict.keys()):
                try:
                    try:
                        nextTaskNr = self.model.row_to_key[i] 
                    except Exception as e:
                        print_error('ExperimentManager: IndexError in queue_fn, likely due to task deletion during iteration', 'error')
                        print(e)
                        break
                    nextTask = self.expDict[nextTaskNr]
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
                    if i == len(self.expDict.keys()):
                        time.sleep(0.5)  # if all experiments are waiting, this stops the program locking up
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
        if taskNr in self.expDict:
            self.expDict[taskNr]['status'] = status
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

    def terminate_experiment(self):
        """Graceful termination of the experiment. Allows current run to finish."""
        print('Terminating experiment')
        self._update_task_status(self.taskNr, 'Terminating')
        self.running = False
        time.sleep(0.1)  # just to give the table time to display. Can be shortened or taken out.
        self.delete_item(self.taskNr)

    def start_measurement(self):
        """Sets arguments, calls experimental start_measurement function."""
        self.experiment.build()
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
        print_error('ExperimentQ: task {} ({}) completed'.format(self.taskNr, self.name), 'bold')
        self._update_task_status(self.taskNr, 'Done')
        # time.sleep(0.1)  # just to give the table time to display. Can be shortened or taken out.
        del self.model[self.taskNr]
        self.running = False

    def set_dict(self, name, value):
        """Sets experiment arguments."""
        print("set_dict called")
        print(f"name: {name}")
        print(f"value: {value}")
        # self.experiment.__dict__[name] = value
        setattr(self.experiment, name, value)

    def set_run_nr(self, value):
        """Sets the experiment run number and updates the table."""
        self.experiment._run = value
        self.expDict[self.taskNr]['run'] = value
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
        self.experiment._repetition = value
        self.expDict[self.taskNr]['repetition'] = value
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
    
    print(f"Experiment Manager started and listening to synced schedule model")
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
