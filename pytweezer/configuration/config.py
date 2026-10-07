import os
import socket

HOSTS = {
    "PH-BEAST": "10.59.3.1",
    "IC-CZC4287H3W": "10.59.3.2",  # rb pc
    "ph-bonesaw": "10.59.3.5",
    "localhost": "127.0.0.1",
}
#: The PC that runs the shared servers (hubs, managers) for real.
SERVER_PC = "PH-BEAST"

port_iterator = iter(range(7278, 99999))
get_next_port = lambda: int(next(port_iterator))

SIMULATING = False  # set to True to run in simulation mode (no real devices, no real cameras, etc.)
LOCAL = False

#: Set to "server" by ``pytweezer-server`` and inherited by every process it starts.
SERVER_ROLE_ENV = "PYTWEEZER_ROLE"
#: A server session anywhere but SERVER_PC must not bind the lab's addresses or
#: drive real hardware, so it simulates whatever SIMULATING says.
SIMULATION_FORCED = (
    os.environ.get(SERVER_ROLE_ENV) == "server"
    and socket.gethostname().lower() != SERVER_PC.lower()
)
SIMULATING = SIMULATING or SIMULATION_FORCED
SERVER_HOST = HOSTS[SERVER_PC] if (not SIMULATING and not LOCAL) else HOSTS["localhost"]

# Postgres + TimescaleDB holding monitor readings and experiment runs. Set
# PYTWEEZER_DB_DSN to keep a real password out of the repository.
DATABASE = {
    "dsn": os.environ.get(
        "PYTWEEZER_DB_DSN",
        f"postgresql://pytweezer:pytweezer@{SERVER_HOST}:5432/pytweezer",
    ),
}


CONFIG = {
    "Servers": {
        "Analysis Manager": {
            "active": True,
            "script": "../pytweezer/servers/analysis_manager.py",
            "host": SERVER_HOST,
            "port": get_next_port(),
        },
        "Device Status": {
            "active": True,
            "script": "../pytweezer/servers/device_status.py",
            "host": SERVER_HOST,
            "pub_port": get_next_port(),
            "poll_interval": 2.0,
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
            "port": get_next_port(),
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
        },
    },
    # Each entry names
    # its backend class directly as "class" (a "module.path:ClassName" string), plus
    # an optional "sim_class" used when "simulate" is True (if omitted, a no-op
    # stand-in is generated from "class") and an optional "teardown" method run at
    # shutdown. Any remaining keys whose names match the backend's
    # __init__ parameters are passed to it. Device names must be unique across the
    # whole category, including composite sub-devices, because get_device()
    # addresses them all by name.
    "Devices": {
        "Rb MotMaster": {
            "active": True,
            "class": "pytweezer.drivers.motmaster:MotMasterInterface",
            "teardown": "disconnect",
            "config_file": "rb_mm_config.json",
            "host": HOSTS["IC-CZC4287H3W"],
            "port": get_next_port(),
            "simulate": SIMULATING,
        },
        "CaF MotMaster": {
            "active": True,
            "class": "pytweezer.drivers.motmaster:MotMasterInterface",
            "teardown": "disconnect",
            "config_file": "caf_mm_config.json",
            "host": HOSTS["ph-bonesaw"],
            "port": get_next_port(),
            "simulate": SIMULATING,
        },
        "CaF HamCam": {
            "active": True,
            "class": "pytweezer.drivers.imagemX2:ImagEMX2Camera",
            "sim_class": "pytweezer.drivers.imagemX2:SimulatedImagEMX2Camera",
            "host": HOSTS["ph-bonesaw"],
            "port": get_next_port(),
            "simulate": SIMULATING,
            "stream_name": "caf_hamcam",
            "timeout": 5.0,
            "image_dir": "C:\\Users\\cafmot\\Documents\\TempCameraImages\\Driver",
        },
        "Rb ThorCam": {
            "active": True,
            "class": "pytweezer.drivers.thorcam:ThorCam",
            "sim_class": "pytweezer.drivers.thorcam:SimulatedThorLabsCamera",
            "host": SERVER_HOST,
            "port": get_next_port(),
            "simulate": SIMULATING,
            "stream_name": "rb_thorcam",
            "timeout": 5.0,
            "image_dir": "C:\\Users\\cafmot\\Documents\\TempCameraImages\\Driver",
        },
        "Tweezer Monitor ThorCam": {
            "active": True,
            "class": "pytweezer.drivers.tweezermonitorcam:ThorCam",
            "sim_class": "pytweezer.drivers.tweezermonitorcam:SimulatedThorLabsCamera",
            "host": SERVER_HOST,
            "port": get_next_port(),
            "simulate": SIMULATING,
            "stream_name": "rb_thorcam2",
            "timeout": 5.0,
            "image_dir": "C:\\Users\\cafmot\\Documents\\TempCameraImages\\Driver",
        },
        # Atom-rearrangement rig: a rearrangement camera and the Blink SLM in one
        # process, with the rearrangement coordinator streaming GPU-computed phase
        # frames straight to slm.update_mask() (no socket). Needs cupy/lap + a CUDA
        # GPU on this host to arm; status()/test() work without them. The SLM is
        # addressable on its own as get_device("Rb SLM").
        "Rb Rearrangement Rig": {
            "active": True,
            "host": SERVER_HOST,
            "port": get_next_port(),
            "simulate": SIMULATING,
            "devices": {
                "Rb HamCam": {
                    "class": "pytweezer.drivers.imagemX2:ImagEMX2Camera",
                    "sim_class": "pytweezer.drivers.imagemX2:SimulatedImagEMX2Camera",
                    "role": "camera",
                    "stream_name": "rb_hamcam",
                    "timeout": 20.0,
                    "image_dir": "C:\\Users\\cafmot\\Documents\\TempCameraImages\\Driver",
                },
                "Rb SLM": {
                    "class": "pytweezer.drivers.slm:SLM",
                    "sim_class": "pytweezer.drivers.slm:SimulatedSLM",
                    "teardown": "close",
                    "role": "slm",
                    # sdk_dll / lut_file / board_number default to the lab's Blink Plus
                    # install (see pytweezer/drivers/slm.py); override here if needed.
                },
            },
            "coordinator": "pytweezer.coordinators.rearrangement:Rearrangement",
        },
    },
    # Background database loggers. Each entry runs pytweezer/servers/logger_server.py,
    # which builds the Logger subclass named by "logger" and polls it on "interval".
    # This is opt-in: nothing reaches the database unless a logger (or explicit
    # DBWriter/log() call) writes it.
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
            "script": "../pytweezer/GUI/streammonitor.py",
        },
        "Applet Launcher": {
            "active": True,
            "script": "../pytweezer/GUI/applet_launcher.py",
        },
        # "H5 Manager": {
        #     "active": False,
        #     "script": "../pytweezer/GUI/h5storage.py"
        # },
        "Property_Editor": {
            "active": False,
            "script": "../pytweezer/GUI/property_editor.py",
        },
        # "Live Plot": {
        #     "active": False,
        #     "script": "../pytweezer/GUI/viewers/live_plot.py"
        # },
        "Analysis Manager UI": {
            "active": True,
            "script": "../pytweezer/GUI/analysismanager.py",
        },
    },
}

# Added after the literal so its ports are allocated last: get_next_port() hands
# out ports in declaration order, so an entry inside "Servers" would shift every
# device's port.
CONFIG["Servers"]["Experiment Manager"] = {
    "active": True,
    "script": "../pytweezer/servers/experiment_manager.py",
    "host": SERVER_HOST,
    "port": get_next_port(),
    "pub_port": get_next_port(),
    # Experiments get in-process simulated devices; data goes to <data_root>/simulated.
    "simulate": SIMULATING,
    # Measurement files and the queue state; PYTWEEZER_DATA_DIR overrides it.
    # None means <repo>/data.
    "data_root": None,
    # Seconds a worker keeps running without reaching the manager.
    "orphan_timeout": 30.0,
}


def get_config():
    """Return the configuration dict.

    The single accessor for ``CONFIG`` used across the servers, drivers and GUI.
    Tests monkeypatch this (per importing module) to inject a fake config.
    """
    return CONFIG
