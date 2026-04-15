HOSTS = {
    "beast": "10.59.3.1",
    "rb_mm_pc": "10.59.3.2",
    "caf_mm_pc": "10.59.3.5",
    "localhost": "127.0.0.1",
    
}

port_iterator = iter(range(7278, 99999))
get_next_port = lambda: int(next(port_iterator))

SIMULATING = True
SERVER_HOST = HOSTS["beast"] if not SIMULATING else HOSTS["localhost"]


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
            "host": SERVER_HOST,
            "port": get_next_port()
        },
        "Model Sync": {
            "active": True,
            "script": "../pytweezer/servers/model_sync.py",
            "host": SERVER_HOST,
            "command_port": get_next_port(),
            "pub_port": get_next_port(),
        },
        "Rb MotMaster Server": {
            "active": True,
            "script": "../pytweezer/experiment/motmaster_server.py",
            "host": HOSTS["rb_mm_pc"],
            "port": get_next_port(),
            "simulate": SIMULATING
        },
        "CaF MotMaster Server": {
            "active": True,
            "script": "../pytweezer/experiment/motmaster_server.py",
            "host": HOSTS["caf_mm_pc"],
            "port": get_next_port(),
            "simulate": SIMULATING
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
            "host": SERVER_HOST,
            "port": get_next_port(),
            "simulate": SIMULATING,
            "stream_name": "imagemx2",
            "timeout": 5.0,
            "tooltip": "Persistent camera process; experiments should use ImagEMX2CameraClient"
        },
        "Blackfly Camera": {
            "active": True,
            "script": "../pytweezer/drivers/bfly2.py",
            "host": SERVER_HOST,
            "port": get_next_port(),
            "simulate": SIMULATING,
            "stream_name": "bfly",
            "timeout": 5.0,
            "tooltip": "Persistent camera process; experiments should use BlackflyCameraClient"
        },
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