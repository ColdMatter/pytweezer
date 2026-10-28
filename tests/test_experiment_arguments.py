import json

import pytest

from pytweezer.experiment import (
    Bool,
    Choice,
    Device,
    Experiment,
    Integer,
    Number,
    Text,
)
from pytweezer.experiment.arguments import coerce_arguments


class Base(Experiment):
    detuning = Number(-12e6, unit="MHz", scale=1e6, min=-50e6, max=0)
    shots = Integer(10, min=1)
    camera = Device("Rb ThorCam")


class Child(Base):
    shots = Integer(5, min=1)
    imaging = Bool(True)
    mode = Choice(["absorption", "fluorescence"])
    note = Text()

    def __init__(self, args=None):
        raise AssertionError("introspection must not instantiate")


class Plain(Base):
    def run_point(self):
        pass


def test_values_are_coerced_to_the_declared_type():
    assert Base.detuning.coerce(-1) == -1.0
    assert type(Base.shots.coerce(3.0)) is int
    assert Child.imaging.coerce(1) is True
    assert Child.mode.default == "absorption"


@pytest.mark.parametrize(
    "argument, value, match",
    [
        (Base.detuning, 1e6, "above the maximum"),
        (Base.detuning, "1", "not a number"),
        (Base.detuning, True, "not a number"),
        (Base.detuning, float("nan"), "not finite"),
        (Base.shots, 2.5, "whole number"),
        (Base.shots, 0, "minimum"),
        (Child.imaging, "yes", "not a bool"),
        (Child.mode, "other", "not one of"),
        (Child.note, 3, "not a string"),
    ],
)
def test_invalid_values_are_rejected(argument, value, match):
    with pytest.raises(ValueError, match=match):
        argument.coerce(value)


def test_bad_default_is_rejected_at_declaration():
    with pytest.raises(ValueError):
        Number(5, max=1)


def test_schema_is_read_from_the_class_without_instantiating():
    schema = Child.schema()
    json.dumps(schema)
    assert list(schema["arguments"]) == ["detuning", "shots", "imaging", "mode", "note"]
    assert schema["arguments"]["shots"]["default"] == 5
    assert schema["arguments"]["detuning"] == {
        "kind": "number",
        "default": -12e6,
        "tooltip": "",
        "group": "",
        "unit": "MHz",
        "scale": 1e6,
        "min": -50e6,
        "max": 0,
        "step": None,
        "ndecimals": None,
    }
    assert schema["devices"] == {"camera": {"device": "Rb ThorCam", "timeout": None}}


def test_coerce_arguments_fills_defaults_and_rejects_unknown_names():
    assert coerce_arguments(Base, {"shots": 2.0}) == {"detuning": -12e6, "shots": 2}
    with pytest.raises(ValueError, match="no argument"):
        coerce_arguments(Base, {"shot": 2})


def test_instance_attributes_hold_effective_values():
    experiment = Plain({"shots": 4})
    assert experiment.shots == 4
    assert Plain.shots.default == 10
    assert experiment.argument_values() == {"detuning": -12e6, "shots": 4}


def test_device_is_resolved_lazily_once_and_closed(fake_devices):
    opened = fake_devices
    experiment = Plain()
    assert opened == []
    assert experiment.camera is experiment.camera
    assert experiment.device("Rb ThorCam") is experiment.camera
    extra = experiment.device("Rb ThorCam", fresh=True)
    assert extra is not experiment.camera
    assert len(opened) == 2

    experiment.close_devices()
    assert all(client.closed for client in opened)
    assert "camera" not in experiment.__dict__
