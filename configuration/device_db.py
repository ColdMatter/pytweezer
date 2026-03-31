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
    "imagemx2_camera": {
        "module": "pytweezer.drivers.imagemX2",
        "class": "ImagEMX2CameraClient",
        "parameters": {
            "server_name": "ImagEM X2 Camera",
            "timeout": 5.0
        }
    },
}