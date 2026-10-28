"""The generic simulated-device mechanism (pytweezer/servers/simulated_device.py)
and the simulated backends built with it."""

from functools import wraps

import pytest
from sipyco import pyon

from pytweezer.drivers.imagemX2 import ImagEMX2Camera, SimulatedImagEMX2Camera
from pytweezer.drivers.orca import ORCACamera, SimulatedORCACamera
from pytweezer.drivers.slm import SLM, SimulatedSLM
from pytweezer.servers import device_server
from pytweezer.servers.simulated_device import public_methods, simulate


def _guard(func):
    """A functools.wraps decorator, like imagemX2's requires_camera."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    return wrapper


class RealBase:
    def base_method(self):
        raise RuntimeError("real base hardware call")


class Real(RealBase):
    def __init__(self):
        raise RuntimeError("real __init__ must never be called")

    def acquire(self):
        raise RuntimeError("real hardware call")

    @_guard
    def guarded(self, x):
        raise RuntimeError("real hardware call")

    def get_config(self):
        raise RuntimeError("real hardware call")

    def _private(self):
        raise RuntimeError("real private call")

    @staticmethod
    def a_staticmethod():
        raise RuntimeError("real static call")


def test_public_methods_enumerates_the_public_surface_without_instantiating():
    assert set(public_methods(Real)) == {
        "acquire",
        "guarded",
        "get_config",
        "a_staticmethod",
        "base_method",
    }
    assert set(public_methods(Real, exclude={"acquire", "guarded"})) == {
        "get_config",
        "a_staticmethod",
        "base_method",
    }


def test_simulate_stubs_missing_public_methods_and_keeps_hand_written_ones():
    @simulate(Real)
    class Sim:
        def __init__(self):
            pass

        def acquire(self):
            return "hand-written"

    sim = Sim()
    assert sim.acquire() == "hand-written"
    assert sim.get_config() is None
    assert sim.base_method() is None
    assert sim.guarded(42, keyword="ok") is None
    assert not hasattr(Sim, "_private")
    assert Sim._simulates is Real


def test_simulate_defaults_and_exclude():
    @simulate(Real, exclude={"acquire"}, defaults={"get_config": dict, "guarded": 7})
    class Sim:
        def __init__(self):
            pass

    sim = Sim()
    assert not hasattr(Sim, "acquire")
    assert sim.guarded() == 7
    first = sim.get_config()
    first["mutated"] = True
    # A callable default must give a fresh mutable per call.
    assert sim.get_config() == {}


@pytest.mark.parametrize(
    "real, simulated",
    [
        (ImagEMX2Camera, SimulatedImagEMX2Camera),
        (ORCACamera, SimulatedORCACamera),
        (SLM, SimulatedSLM),
    ],
)
def test_simulated_backend_covers_real_interface(real, simulated):
    assert not set(public_methods(real)) - set(dir(simulated))


SIM_CONFS = {
    # No hand-written sim class: exercises the auto-generated stand-in.
    "motmaster": {
        "class": "pytweezer.drivers.motmaster:MotMasterInterface",
        "simulate": True,
    },
    "imagemx2": {
        "sim_class": "pytweezer.drivers.imagemX2:SimulatedImagEMX2Camera",
        "simulate": True,
    },
    "ids1054b": {
        "sim_class": "pytweezer.drivers.ids1054b:SimulatedIDS1054B",
        "simulate": True,
    },
}


@pytest.mark.parametrize("driver", sorted(SIM_CONFS))
def test_simulated_backend_methods_are_pyon_safe(driver):
    """Catch a non-serialisable stub return here rather than at the first live RPC."""
    spec = device_server.build_spec(driver, conf=SIM_CONFS[driver])
    target = spec.target
    try:
        for name in public_methods(type(target)):
            try:
                result = getattr(target, name)()
            except TypeError:  # needs arguments we can't guess
                continue
            except RuntimeError:  # state-machine guard, e.g. "camera is not running"
                continue
            pyon.encode({"ret": result})
    finally:
        if spec.teardown is not None:
            spec.teardown()
