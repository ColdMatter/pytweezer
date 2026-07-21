"""Unit tests for the generic simulated-device mechanism
(pytweezer/servers/simulated_device.py), exercised in isolation against small
fake "real" classes so nothing here depends on hardware backends.
"""

from functools import wraps

import pytest

from pytweezer.servers.simulated_device import public_methods, simulate


# --------------------------------------------------------------------------- #
# Fixtures: tiny stand-in "real" backend classes
# --------------------------------------------------------------------------- #

def _guard(func):
    """A functools.wraps-based decorator, mirroring imagemX2's requires_camera,
    to prove signature/name introspection sees through it."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    return wrapper


class RealBase:
    def base_method(self):
        raise RuntimeError("real base hardware call")


class Real(RealBase):
    def __init__(self):
        # A real backend's __init__ touches hardware; simulate() must never run it.
        raise RuntimeError("real __init__ must never be called by simulate()")

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


# --------------------------------------------------------------------------- #
# public_methods
# --------------------------------------------------------------------------- #

def test_public_methods_enumerates_public_and_inherited():
    names = set(public_methods(Real))
    assert "acquire" in names
    assert "get_config" in names
    assert "guarded" in names  # seen through functools.wraps
    assert "a_staticmethod" in names
    assert "base_method" in names  # inherited from RealBase


def test_public_methods_excludes_dunders_and_privates():
    names = set(public_methods(Real))
    assert "_private" not in names
    assert not any(n.startswith("_") for n in names)


def test_public_methods_honors_exclude():
    names = set(public_methods(Real, exclude={"acquire", "guarded"}))
    assert "acquire" not in names
    assert "guarded" not in names
    assert "get_config" in names


def test_public_methods_does_not_instantiate_real_class():
    # Real.__init__ raises; enumerating methods must not trip it.
    public_methods(Real)  # would raise if it instantiated


# --------------------------------------------------------------------------- #
# simulate
# --------------------------------------------------------------------------- #

def test_simulate_autostubs_missing_methods():
    @simulate(Real)
    class Sim:
        def __init__(self):
            pass

    sim = Sim()
    # Not hand-written -> auto-stub, returns None, never touches "hardware".
    assert sim.acquire() is None
    assert sim.get_config() is None
    assert sim.base_method() is None


def test_simulate_preserves_hand_written_methods():
    @simulate(Real)
    class Sim:
        def __init__(self):
            pass

        def acquire(self):
            return "real-sim-behavior"

    assert Sim().acquire() == "real-sim-behavior"


def test_simulate_stub_accepts_arbitrary_args():
    @simulate(Real)
    class Sim:
        def __init__(self):
            pass

    # guarded(self, x) on the real class takes an arg; the stub must swallow it.
    assert Sim().guarded(42, keyword="ok") is None


def test_simulate_scalar_default_return():
    @simulate(Real, defaults={"get_config": 7})
    class Sim:
        def __init__(self):
            pass

    assert Sim().get_config() == 7


def test_simulate_callable_default_produces_fresh_mutable_each_call():
    @simulate(Real, defaults={"get_config": dict})
    class Sim:
        def __init__(self):
            pass

    sim = Sim()
    first = sim.get_config()
    assert first == {}
    first["mutated"] = True
    # A second call must not see the mutation from the first.
    assert sim.get_config() == {}


def test_simulate_stamps_simulates_attribute():
    @simulate(Real)
    class Sim:
        def __init__(self):
            pass

    assert Sim._simulates is Real


def test_simulate_does_not_add_private_methods():
    @simulate(Real)
    class Sim:
        def __init__(self):
            pass

    assert not hasattr(Sim, "_private")


def test_simulate_exclude_leaves_method_unstubbed():
    @simulate(Real, exclude={"acquire"})
    class Sim:
        def __init__(self):
            pass

    assert not hasattr(Sim, "acquire")


# --------------------------------------------------------------------------- #
# Interface parity of the real simulated backends
# --------------------------------------------------------------------------- #

def test_simulated_imagemx2_covers_real_interface():
    from pytweezer.drivers.imagemX2 import ImagEMX2Camera, SimulatedImagEMX2Camera

    missing = set(public_methods(ImagEMX2Camera)) - set(dir(SimulatedImagEMX2Camera))
    assert not missing, f"SimulatedImagEMX2Camera missing methods: {missing}"
