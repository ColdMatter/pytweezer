device_db = {
    "dummy_camera": {
        "module": "pytweezer.experiment.dummy_drivers",
        "class": "DummyCamera",
        "parameters": {
            "mode": "test",
            "n_frames": 3
        }
    },
    "dummy_synth": {
        "module": "pytweezer.experiment.dummy_drivers",
        "class": "DummySynth",
        "parameters": {
            "frequency": 1e9,
        }
    },
}