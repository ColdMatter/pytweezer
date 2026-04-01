HOST_DICT = {
    "beast": "10.59.3.2",
    "localhost": "127.0.0.1"
}

port_iterator = iter(range(7278, 99999))
get_next_port = lambda: int(next(port_iterator))

HOST = HOST_DICT["beast"]

SIMULATING = True

CONFIG = {
    "GUI": {
        "Browser": {
            "active": True,
            "script": "../pytweezer/GUI/tweezer_browser.py"
        },
        "StreamMonitor": {
            "active": False,
            "script": "../pytweezer/GUI/streammonitor.py"
        },
        "H5 Manager": {
            "active": False,
            "script": "../pytweezer/GUI/h5storage.py"
        },
        "Property_Editor": {
            "active": False,
            "script": "../pytweezer/GUI/property_editor.py"
        },
        "Live Plot": {
            "active": False,
            "script": "../pytweezer/GUI/viewers/live_plot.py"
        },
        "Analysis Manager UI": {
            "active": True,
            "script": "../pytweezer/GUI/analysismanager.py"
        }

    },
    "Servers": {
        "Analysis Manager": {
            "active": True,
            "script": "../pytweezer/servers/analysis_manager.py",
            "host": HOST,
            "port": get_next_port()
        },
        "Model Sync": {
            "active": True,
            "script": "../pytweezer/servers/model_sync.py",
            "host": HOST,
            "command_port": get_next_port(),
            "pub_port": get_next_port(),
        },
        "MotMaster Server": {
            "active": True,
            "script": "../pytweezer/experiment/motmaster_server.py",
            "host": HOST,
            "port": get_next_port(),
            "simulate": SIMULATING
        },
        "Imagehub": {
            "active": True,
            "host": HOST,
            "pub_port": get_next_port(),
            "sub_port": get_next_port(),
            "script": "../pytweezer/servers/xsub_xpub.py", 
        },
        "Commandhub": {
            "active": True,
            "host": HOST,
            "pub_port": get_next_port(),
            "sub_port": get_next_port(),
            "script": "../pytweezer/servers/xsub_xpub.py",
        },
        "Datahub": {
            "active": True,
            "host": HOST,
            "pub_port": get_next_port(),
            "sub_port": get_next_port(),
            "script": "../pytweezer/servers/xsub_xpub.py",
        },
        "Propertyhub": {
            "active": True,
            "host": HOST,
            "pub_port": get_next_port(),
            "sub_port": get_next_port(),
            "script": "../pytweezer/servers/xsub_xpub.py",
        },
        "Messagehub": {
            "active": True,
            "host": HOST,
            "pub_port": get_next_port(),
            "sub_port": get_next_port(),
            "stream_name": "Global Messages",
            "script": "../pytweezer/servers/xsub_xpub.py",
        },
        "Propertylogger": {
            "active": True,
            "script": "../pytweezer/servers/propertylogger.py",
            "host": HOST,
            "port": get_next_port()
        },
        "Datalogger": {
            "active": True,
            "script": "../pytweezer/servers/datalogger.py"
        },
        "Imagelogger": {
            "active": True,
            "script": "../pytweezer/servers/imagelogger.py"
        },
        "Experiment Manager": {
            "active": True,
            "script": "../pytweezer/servers/experiment_manager.py"
        },
        "ImagEM X2 Camera": {
            "active": True,
            "script": "../pytweezer/servers/imagemx2_server.py",
            "host": HOST,
            "port": get_next_port(),
            "simulate": SIMULATING,
            "stream_name": "imagemx2",
            "timeout": 5.0,
            "tooltip": "Persistent camera process; experiments should use ImagEMX2CameraClient"
        },
        "Elephant": {
            "active": False,
            "script": "../pytweezer/GUI/mighty.py"
        }
    },
    "Viewer": {
        "DummyViewer": {
            "active": False,
            "script": "../pytweezer/GUI/viewers/image_group.py"
        },
        "TweezerViewer": {
            "active": True,
            "script": "../pytweezer/GUI/viewers/tweezer_image_monitor.py"
        }
    }
}