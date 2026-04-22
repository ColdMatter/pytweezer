from pytweezer.experiment.experiment import Experiment

class RBBasicTest(Experiment):
    motmaster_script = "RbTweezerBasic2025"
    def build(self):
        super().build()

    def run(self):
        print(f"sending params {self._experiment_params} to MotMaster and starting experiment...")
        self._motmaster_client.start_experiment(parameters=self._experiment_params)