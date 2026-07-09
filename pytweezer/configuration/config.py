import os

HOSTS = {
    "ph-beast": "10.59.3.1",
    "IC-CZC4287H3W": "10.59.3.2", # rb pc
    "ph-bonesaw": "10.59.3.5",
    "localhost": "127.0.0.1",

}

port_iterator = iter(range(7278, 99999))
get_next_port = lambda: int(next(port_iterator))

SIMULATING = False # set to True to run in simulation mode (no real devices, no real cameras, etc.)
LOCAL = False
SERVER_HOST = HOSTS["ph-beast"] if (not SIMULATING and not LOCAL) else HOSTS["localhost"]

# Self-hosted InfluxDB 2.x connection. Every value can be overridden by an
# environment variable so the token need not be hardcoded in a real deployment;
# the defaults let a fresh checkout "just work" against a local InfluxDB.
# See docs/influx_logging.md for the one-command self-hosted setup.
INFLUXDB = {
    "url": os.environ.get("INFLUXDB_URL", f"http://{SERVER_HOST}:8086"),
    "token": os.environ.get("INFLUXDB_TOKEN", "pytweezer-token"),
    "org": os.environ.get("INFLUXDB_ORG", "pytweezer"),
    "bucket": os.environ.get("INFLUXDB_BUCKET", "devices"),
}


CONFIG = {
    "Servers": {
        "Analysis Manager": {
            "active": True,
            "script": "../pytweezer/servers/analysis_manager.py",
            "host": SERVER_HOST,
            "port": get_next_port()
        },
        "Device Status": {
            "active": True,
            "script": "../pytweezer/servers/device_status.py",
            "host": SERVER_HOST,
            "pub_port": get_next_port(),
            "poll_interval": 2.0,
        },
        # "Experiment Manager": {
        #     "active": True,
        #     "script": "../pytweezer/servers/experiment_manager.py",
        #     "host": SERVER_HOST
        # },
        # "Model Sync": {
        #     "active": True,
        #     "script": "../pytweezer/servers/model_sync.py",
        #     "host": SERVER_HOST,
        #     "command_port": get_next_port(),
        #     "pub_port": get_next_port(),
        # },
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
         "Rb MotMaster": {
            "active": True,
            "script": "../pytweezer/servers/device_server.py",
            "driver": "motmaster",
            "config_file": "rb_mm_config.json",
            "host": HOSTS["IC-CZC4287H3W"],
            "port": get_next_port(),
            "simulate": SIMULATING
        },
        "CaF MotMaster": {
            "active": True,
            "script": "../pytweezer/servers/device_server.py",
            "driver": "motmaster",
            "config_file": "caf_mm_config.json",
            "host": HOSTS["ph-bonesaw"],
            "port": get_next_port(),
            "simulate": SIMULATING
        },
        "Rb HamCam": {
            "active": True,
            "script": "../pytweezer/servers/device_server.py",
            "driver": "imagemx2",
            "host": SERVER_HOST,
            "port": get_next_port(),
            "simulate": SIMULATING,
            "stream_name": "rb_hamcam",
            "timeout": 5.0,
            "image_dir": "C:\\Users\\cafmot\\Documents\\TempCameraImages\\Driver"
        },
        "CaF HamCam": {
            "active": True,
            "script": "../pytweezer/servers/device_server.py",
            "driver": "imagemx2",
            "host": HOSTS["ph-bonesaw"],
            "port": get_next_port(),
            "simulate": SIMULATING,
            "stream_name": "caf_hamcam",
            "timeout": 5.0,
            "image_dir": "C:\\Users\\cafmot\\Documents\\TempCameraImages\\Driver"
        },
        "Blackfly": {
            "active": False,
            "script": "../pytweezer/servers/device_server.py",
            "driver": "blackfly",
            "host": SERVER_HOST,
            "port": get_next_port(),
            "simulate": SIMULATING,
            "stream_name": "bfly",
            "timeout": 5.0,
        },
    },
    # Background InfluxDB loggers. Each entry runs pytweezer/servers/logger_server.py,
    # which builds the Logger subclass named by "logger" and polls it on "interval".
    # This is opt-in: nothing is pushed to InfluxDB unless a logger (or explicit
    # InfluxWriter/log() call) writes it. See docs/influx_logging.md.
    "Loggers": {
        "NI ADC Logger": {
            "active": False,
            "script": "../pytweezer/servers/logger_server.py",
            "logger": "ni_adc",
            "host": SERVER_HOST,
            "interval": 1.0,
            "simulate": SIMULATING,
            "channels": ["Dev1/ai0", "Dev1/ai1"],
            "measurement": "ni_adc",
            "tags": {"system": "Rb"},
        },
    },
    "GUI": {
        # "Browser": {
        #     "active": True,
        #     "script": "../pytweezer/GUI/tweezer_browser.py"
        # },
        "StreamMonitor": {
            "active": True,
            "script": "../pytweezer/GUI/streammonitor.py"
        },
        "Applet Launcher": {
            "active": True,
            "script": "../pytweezer/GUI/applet_launcher.py"
        },
        # "H5 Manager": {
        #     "active": False,
        #     "script": "../pytweezer/GUI/h5storage.py"
        # },
        "Property_Editor": {
            "active": False,
            "script": "../pytweezer/GUI/property_editor.py"
        },
        # "Live Plot": {
        #     "active": False,
        #     "script": "../pytweezer/GUI/viewers/live_plot.py"
        # },
        "Analysis Manager UI": {
            "active": True,
            "script": "../pytweezer/GUI/analysismanager.py"
        }
    },
}