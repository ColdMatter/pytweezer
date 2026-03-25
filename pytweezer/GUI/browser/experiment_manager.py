import numpy as np
import time
import importlib.util
import inspect

from pytweezer.GUI.browser.browser_workers import Worker
from PyQt5.QtCore import QDateTime
from pytweezer.analysis.print_messages import print_error
from pytweezer.experiment.experiment import Experiment


class ExperimentManager:
    """
    Contains the functions for running experiments from the ExperimentQ.
    The queue_fn is constantly running inside a worker thread checking the Q's expDict for
    experiments to run.
    Experiments consist of:
        *   measurement: one or more repetitions of

            * sequence (or scan):  multiple runs with varied parameters

                *   run:  one single run

    Before and after each measurement, start_ and end_measurement are called.
    Before and after each sequence, start_ and end_sequence are called.
    By default, and a measurement is a single repetition of a sequence with a single run.
    """
    def __init__(self, browser=None):
        self.browser = browser
        self._props = browser._props
        self.queue = browser.queue
        self.expDict = self.queue.expDict
        self.threadPool = browser.threadPool
        self.start_queue()
        self.paused = False
        self.running = False  # tells the queue whether an experiment is currently running
        self.taskNr = 0

    def _load_experiment_from_task(self, task_dict):
        filepath = task_dict.get('filepath')
        if not filepath:
            raise ValueError('Task has no filepath field')

        spec = importlib.util.spec_from_file_location(filepath, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            mro = inspect.getmro(obj)
            if len(mro) > 1 and inspect.getmro(obj)[1].__name__ == 'Experiment':
                experiment = obj(self.browser.props, self.browser.motmaster_interface)
                experiment.build()
                return experiment

        raise ValueError(f'No Experiment subclass found in {filepath}')

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
        self.experiment = self._load_experiment_from_task(task_dict)
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
        self.browser.props.set('ExperimentQ/current_scan', self.scan_pars)

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
                self.queue.set_task_field(self.taskNr, 'status', 'Scanning')
            else:
                self.queue.set_task_field(self.taskNr, 'status', 'Running')

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
                self.queue.set_task_field(self.taskNr, 'status', 'Running')
            else:
                self.queue.set_task_field(self.taskNr, 'status', 'Paused')
        self.paused = not self.paused
        self.browser.props.set('/ExperimentQ/paused', self.paused)
        self.queueWorker.signals.tableUpdate.emit()

    def start_queue(self):
        """Starts the experiment queue worker in the experiment thread."""
        print('queue worker starting')
        self.queueRunning = True
        self.queueWorker = Worker(self.queue_fn)
        self.queueWorker.signals.update_ui.connect(self.update_table)
        self.queueWorker.signals.deleteItem.connect(self.queue.delete_item)
        self.queueWorker.signals.finished.connect(self.finished)
        self.threadPool.start(self.queueWorker)

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
            try:
                while self.expDict == {} or self.running:
                    time.sleep(0.01)
                while i < len(self.expDict.keys()):
                    nextTaskNr = self.queue.tableModel.row_to_key[i]  # list of task numbers in run order
                    nextTask = self.expDict[nextTaskNr]
                    if nextTask['status'] not in ('Sleeping', 'Failed', 'Waiting') or self.due_check(nextTask):
                        self.run(nextTaskNr)
                        i = 0
                    else:
                        i += 1
                    if i == len(self.expDict.keys()):
                        time.sleep(0.5)  # if all experiments are waiting, this stops the program locking up
            except Exception as e:
                print_error('Task {} failed with error: {}'.format(nextTaskNr, e), 'error')
                self.running = False
                self.queue.set_task_field(nextTaskNr, 'status', 'Failed')
                self.queueWorker.signals.tableUpdate.emit()
                i += 1
                raise

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
        self.queue.tableModel.layoutChanged.emit()

    def terminate_experiment(self):
        """Graceful termination of the experiment. Allows current run to finish."""
        print('Terminating experiment')
        self.queue.set_task_field(self.taskNr, 'status', 'Terminating')
        self.running = False
        self.queueWorker.signals.tableUpdate.emit()
        time.sleep(0.1)  # just to give the table time to display. Can be shortened or taken out.
        # del self.queue.tableModel[self.taskNr]
        self.queue.lock = True
        self.queueWorker.signals.deleteItem.emit(self.taskNr)
        while self.queue.lock:
            time.sleep(0.01)

    def start_measurement(self):
        """Sets arguments, calls experimental start_measurement function."""
        for arg in self.arg_dict.keys():
            self.set_dict(arg, self.arg_dict[arg])
        # print('\n\033[1mExperimentQ: task {} ({}) starting\n\033[0m'.format(self.taskNr, self.name))
        print_error('ExperimentQ: task {} ({}) starting'.format(self.taskNr, self.name), 'bold')
        # print('_task:', self._props.get('/Experiments/_task', 0))
        self.running = True
        if self.paused:
            self.queue.set_task_field(self.taskNr, 'status', 'Paused')
            self.queueWorker.signals.update_ui.emit()
        # self.experiment.start_measurement()

    def end_measurement(self):
        """Run the end_measurement function, clean up the table"""
        self.experiment.cleanup()
        # print('\n\033[1mExperimentQ: task {} ({}) completed\n\033[0m'.format(self.taskNr, self.name))
        print_error('ExperimentQ: task {} ({}) completed'.format(self.taskNr, self.name), 'bold')
        self.queue.set_task_field(self.taskNr, 'status', 'Done')
        time.sleep(0.1)  # just to give the table time to display. Can be shortened or taken out.
        self.queue.delete_task(self.taskNr)
        self.running = False

    def set_dict(self, name, value):
        """Sets experiment arguments."""
        print("set_dict called")
        print(f"name: {name}")
        print(f"value: {value}")
        self.experiment.__dict__[name] = value

    def set_run_nr(self, value):
        """Sets the experiment run number and updates the table."""
        self.experiment._run = value
        self.queue.set_task_field(self.taskNr, 'run', value)
        self.queueWorker.signals.update_ui.emit()

    def set_rep_nr(self, value):
        """Sets the experiment repetition number and updates the table."""
        self.experiment._repetition = value
        self.queue.set_task_field(self.taskNr, 'repetition', value)
        self.queueWorker.signals.update_ui.emit()

    def terminate_all(self):
        for task_nr in list(self.queue.tableModel.row_to_key):
            self.queue.set_task_field(task_nr, 'terminated', True)
            self.queue.set_task_field(task_nr, 'status', 'Termination Pending')

    def restart_queue(self):
        if not self.queueRunning:
            self.start_queue()
