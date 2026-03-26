from pytweezer.experiment.experiment import Experiment

class RBBasicTest(Experiment):
    motmaster_script = "RbTweezerBasic2025"
    def build(self):
        super().build()

    def run(self):
        self._motmaster_client.start_experiment()