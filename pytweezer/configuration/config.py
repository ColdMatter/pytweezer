HOSTS = {
    "beast": "10.59.3.1",
    "rb_mm_pc": "10.59.3.2",
    "caf_mm_pc": "10.59.3.5",
    "localhost": "127.0.0.1",
    
}

port_iterator = iter(range(7278, 99999))
get_next_port = lambda: int(next(port_iterator))

SIMULATING = False
LOCAL = True
SERVER_HOST = HOSTS["beast"] if (not SIMULATING and not LOCAL) else HOSTS["localhost"]


CONFIG = {
    "Servers": {
        "Analysis Manager": {
            "active": True,
            "script": "../pytweezer/servers/analysis_manager.py",
            "host": SERVER_HOST,
            "port": get_next_port()
        },
        "Experiment Manager": {
            "active": True,
            "script": "../pytweezer/servers/experiment_manager.py",
            "host": SERVER_HOST
        },
        "Model Sync": {
            "active": True,
            "script": "../pytweezer/servers/model_sync.py",
            "host": SERVER_HOST,
            "command_port": get_next_port(),
            "pub_port": get_next_port(),
        },
        "Imagehub": {
            "active": True,
            "host": SERVER_HOST,
            "pub_port": get_next_port(),
            "sub_port": get_next_port(),
            "script": "../pytweezer/servers/xsub_xpub.py", 
        },
        "Commandhub": {
            "active": True,
            "host": SERVER_HOST,
            "pub_port": get_next_port(),
            "sub_port": get_next_port(),
            "script": "../pytweezer/servers/xsub_xpub.py",
        },
        "Datahub": {
            "active": True,
            "host": SERVER_HOST,
            "pub_port": get_next_port(),
            "sub_port": get_next_port(),
            "script": "../pytweezer/servers/xsub_xpub.py",
        },
        "Propertyhub": {
            "active": True,
            "host": SERVER_HOST,
            "pub_port": get_next_port(),
            "sub_port": get_next_port(),
            "script": "../pytweezer/servers/xsub_xpub.py",
        },
        "Messagehub": {
            "active": True,
            "host": SERVER_HOST,
            "pub_port": get_next_port(),
            "sub_port": get_next_port(),
            "stream_name": "Global Messages",
            "script": "../pytweezer/servers/xsub_xpub.py",
        },
        "Propertylogger": {
            "active": True,
            "script": "../pytweezer/servers/propertylogger.py",
            "host": SERVER_HOST,
            "port": get_next_port()
        },
        "Datalogger": {
            "active": True,
            "script": "../pytweezer/servers/datalogger.py",
            "host": SERVER_HOST,
        },
        "Imagelogger": {
            "active": True,
            "script": "../pytweezer/servers/imagelogger.py",
            "host": SERVER_HOST,
        }
    },
    "Devices": {
         "Rb MotMaster Server": {
            "active": True,
            "script": "../pytweezer/experiment/motmaster_server.py",
            "config_file": "rb_mm_config.json",
            "host": HOSTS["rb_mm_pc"],
            "port": get_next_port(),
            "simulate": SIMULATING
        },
        "CaF MotMaster Server": {
            "active": True,
            "script": "../pytweezer/experiment/motmaster_server.py",
            "config_file": "caf_mm_config.json",
            "host": HOSTS["caf_mm_pc"],
            "port": get_next_port(),
            "simulate": SIMULATING
        },
        "Rb HamCam": {
            "active": True,
            "script": "../pytweezer/servers/imagemx2_server.py",
            "host": SERVER_HOST,
            "port": get_next_port(),
            "simulate": SIMULATING,
            "stream_name": "rb_hamcam",
            "timeout": 5.0,
        },
        "CaF HamCam": {
            "active": True,
            "script": "../pytweezer/servers/imagemx2_server.py",
            "host": HOSTS["caf_mm_pc"],
            "port": get_next_port(),
            "simulate": SIMULATING,
            "stream_name": "caf_hamcam",
            "timeout": 5.0,
        },
        "Blackfly Camera": {
            "active": False,
            "script": "../pytweezer/drivers/bfly2.py",
            "host": SERVER_HOST,
            "port": get_next_port(),
            "simulate": SIMULATING,
            "stream_name": "bfly",
            "timeout": 5.0,
        },
    },
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