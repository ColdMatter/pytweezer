from concurrent.futures import ProcessPoolExecutor, TimeoutError
from multiprocessing import get_context
from pytweezer.experiment.experiment import Experiment


# 1) Top-level pure function (must be picklable).
def _rearrangement_kernel(input_data, config):
    # Hypothetical CPU-heavy logic:
    # - no Qt objects
    # - no hardware clients
    # - no self / instance state
    output = {"moves": [], "score": 0.0}
    # ... heavy compute ...
    return output


class RearrangementExperiment(Experiment):
    _pool = None  # shared per-process pool for this class

    def __init__(self):
        super().__init__()

    @classmethod
    def _get_pool(cls):
        if cls._pool is None:
            # Spawn is safest with Qt/threaded apps.
            cls._pool = ProcessPoolExecutor(
                max_workers=1,
                mp_context=get_context("spawn"),
            )
        return cls._pool

    @classmethod
    def shutdown_pool(cls):
        if cls._pool is not None:
            cls._pool.shutdown(wait=True, cancel_futures=False)
            cls._pool = None

    def run(self):
        result = self.do_rearrangement()
        # Use result in normal experiment flow
        # self.result_channels["..."].data = result
        return result

    def do_rearrangement(self):
        # 2) Build serializable inputs from experiment state.
        input_data = {
            # e.g. "image": self.latest_image.tolist(),
            # or lightweight arrays/params
        }
        config = {
            # e.g. "max_iter": 2000
        }

        future = self._get_pool().submit(_rearrangement_kernel, input_data, config)

        try:
            # Optional timeout to avoid hanging forever.
            return future.result(timeout=10.0)
        except TimeoutError:
            future.cancel()
            raise RuntimeError("Rearrangement timed out")