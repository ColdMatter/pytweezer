import importlib
import os
import socket

#: The lab's PCs. ``address`` is both the IP that ``"host"`` entries use and the
#: SSH address; ``repo`` is the pytweezer checkout (None if not managed);
#: ``launcher`` is the GUI that runs on the PC, if any.
MACHINES = {
    "PH-BEAST": {
        "address": "10.59.3.1",
        "user": "tweez",
        "repo": r"C:\Users\tweez\ColdMatter\pytweezer",
        "os": "windows",
        "launcher": "server",
    },
    "IC-CZC221CCHS": {  # rb pc
        "address": "10.59.3.2",
        "user": "cafmot",
        "repo": r"C:\Users\CaFMOT\ColdMatter\pytweezer",
        "os": "windows",
        "launcher": "client",
    },
    "IC-CZC4287H3W": {
        "address": "10.59.3.3",
        "user": "cafmot",
        "repo": None,
        "os": "windows",
        "launcher": "client",
    },
    "ph-bonesaw": {
        "address": "10.59.3.5",
        "user": "cafmot",
        "repo": r"C:\Users\cafmot\pytweezer",
        "os": "windows",
        "launcher": "client",
    },
    "twalker-laptop": {
        "address": "100.101.42.103",
        "user": "twalker",
        "repo": "/home/twalker/RbCaF/pytweezer",
        "os": "linux",
        "launcher": None,
    },
}
LOCALHOST = "127.0.0.1"


def machine_name(address):
    """Return the ``MACHINES`` name with this address, or None if there is none."""
    return next(
        (name for name, machine in MACHINES.items() if machine["address"] == address),
        None,
    )


#: The PC that runs the shared servers (hubs, managers) for real.
SERVER_PC = "PH-BEAST"

#: Just below the range ``get_next_port()`` hands out, so no CONFIG port takes it.
GUI_LOCK_PORT = 7277

port_iterator = iter(range(7278, 99999))
get_next_port = lambda: int(next(port_iterator))


#: A simulating process has simulated devices, hubs on this PC and "sim."-prefixed
#: stream names, so nothing it does reaches the lab. It simulates if
#: ``PYTWEEZER_SIMULATION=1`` (the tests set it; set it to simulate on the server
#: PC), or if it is part of a ``pytweezer-server`` session (``PYTWEEZER_ROLE=server``)
#: on any PC but SERVER_PC. Processes inherit both from whoever started them.
ON_SERVER_PC = socket.gethostname().lower() == SERVER_PC.lower()
SIMULATING = os.environ.get("PYTWEEZER_SIMULATION") == "1" or (
    os.environ.get("PYTWEEZER_ROLE") == "server" and not ON_SERVER_PC
)
SERVER_HOST = LOCALHOST if SIMULATING else MACHINES[SERVER_PC]["address"]

# Postgres + TimescaleDB holding monitor readings and experiment runs. Set
# PYTWEEZER_DB_DSN to keep a real password out of the repository.
DATABASE = {
    "dsn": os.environ.get(
        "PYTWEEZER_DB_DSN",
        f"postgresql://pytweezer:pytweezer@{SERVER_HOST}:5432/pytweezer",
    ),
}

# Grafana on the server PC, for the GUI's "Open in Grafana" buttons.
GRAFANA = {
    "url": os.environ.get("PYTWEEZER_GRAFANA_URL", f"http://{SERVER_HOST}:3000"),
}


CONFIG = {
    "Servers": {
        "Analysis Manager": {
            "run_at_startup": True,
            "script": "../pytweezer/servers/analysis_manager.py",
            "host": SERVER_HOST,
            "port": get_next_port(),
        },
        "Device Status": {
            "run_at_startup": True,
            "script": "../pytweezer/servers/device_status.py",
            "host": SERVER_HOST,
            "pub_port": get_next_port(),
            "poll_interval": 2.0,
        },
        "Imagehub": {
            "run_at_startup": True,
            "host": SERVER_HOST,
            "pub_port": get_next_port(),
            "sub_port": get_next_port(),
            "script": "../pytweezer/servers/xsub_xpub.py",
        },
        "Commandhub": {
            "run_at_startup": True,
            "host": SERVER_HOST,
            "pub_port": get_next_port(),
            "sub_port": get_next_port(),
            "script": "../pytweezer/servers/xsub_xpub.py",
        },
        "Datahub": {
            "run_at_startup": True,
            "host": SERVER_HOST,
            "pub_port": get_next_port(),
            "sub_port": get_next_port(),
            "script": "../pytweezer/servers/xsub_xpub.py",
        },
        # The shared properties tree: "port" publishes it, "rpc_port" takes edits.
        "Properties": {
            "run_at_startup": True,
            "script": "../pytweezer/servers/property_server.py",
            "host": SERVER_HOST,
            "port": get_next_port(),
            "rpc_port": get_next_port(),
        },
        "Messagehub": {
            "run_at_startup": True,
            "host": SERVER_HOST,
            "pub_port": get_next_port(),
            "sub_port": get_next_port(),
            "stream_name": "Global Messages",
            "script": "../pytweezer/servers/xsub_xpub.py",
        },
        "Datalogger": {
            "run_at_startup": True,
            "script": "../pytweezer/servers/datalogger.py",
            "host": SERVER_HOST,
        },
        "Imagelogger": {
            "run_at_startup": True,
            "script": "../pytweezer/servers/imagelogger.py",
            "host": SERVER_HOST,
        },
        "Experiment Manager": {
            "run_at_startup": True,
            "script": "../pytweezer/servers/experiment_manager.py",
            "host": SERVER_HOST,
            "port": get_next_port(),
            # Publishes the queue and the running task's points (sipyco
            # sync_struct).
            "sync_port": get_next_port(),
            # Serves the measurement files to other PCs (sipyco RPC).
            "results_port": get_next_port(),
            # Experiments get in-process simulated devices; data goes to
            # <data_root>/simulated.
            "simulate": SIMULATING,
            # Measurement files and the queue state; PYTWEEZER_DATA_DIR overrides it.
            # None means <repo>/data.
            "data_root": None,
            # Each finished measurement file is copied here, at the same path
            # (a folder on the router's USB drive, which it shares as "g" -
            # see the labbook's Lab network overview). None means no copy.
            "copy_root": r"\\10.59.0.1\g\pytweezer-data",
            # Seconds a worker keeps running without reaching the manager.
            "orphan_timeout": 30.0,
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
            "run_at_startup": True,
            "class": "pytweezer.drivers.motmaster:MotMasterInterface",
            "sim_class": "pytweezer.drivers.motmaster:SimulatedMotMasterInterface",
            "teardown": "disconnect",
            "config_file": "rb_mm_config.json",
            "host": MACHINES["IC-CZC221CCHS"]["address"],
            "port": get_next_port(),
            "simulate": SIMULATING,
        },
        "CaF MotMaster": {
            "run_at_startup": True,
            "class": "pytweezer.drivers.motmaster:MotMasterInterface",
            "sim_class": "pytweezer.drivers.motmaster:SimulatedMotMasterInterface",
            "teardown": "disconnect",
            "config_file": "caf_mm_config.json",
            "linked_partner": "Rb MotMaster",
            "host": MACHINES["ph-bonesaw"]["address"],
            "port": get_next_port(),
            "simulate": SIMULATING,
        },
        "ORCA": {
            "run_at_startup": False,
            "class": "pytweezer.drivers.orca:ORCACamera",
            "sim_class": "pytweezer.drivers.orca:SimulatedORCACamera",
            "host": MACHINES["ph-bonesaw"]["address"],
            "port": get_next_port(),
            "simulate": SIMULATING,
            "serial_number": "280187",
            "stream_name": "orca",
            "object_pixel_m": None,
            "external_exposure": True,
            "timeout": 5.0,
            "image_dir": "C:\\Users\\cafmot\\Documents\\TempCameraImages\\Driver",
        },
        "Rb ThorCam": {
            "run_at_startup": False,
            "class": "pytweezer.drivers.thorcam:ThorCam",
            "sim_class": "pytweezer.drivers.thorcam:SimulatedThorLabsCamera",
            "host": MACHINES["PH-BEAST"]["address"],
            "port": get_next_port(),
            "simulate": SIMULATING,
            "stream_name": "rb_thorcam",
            "object_pixel_m": None,
            "timeout": 5.0,
            "image_dir": "C:\\Users\\cafmot\\Documents\\TempCameraImages\\Driver",
        },
        "Tweezer Monitor ThorCam": {
            "run_at_startup": False,
            "class": "pytweezer.drivers.tweezermonitorcam:ThorCam",
            "sim_class": "pytweezer.drivers.tweezermonitorcam:SimulatedThorLabsCamera",
            "host": MACHINES["PH-BEAST"]["address"],
            "port": get_next_port(),
            "simulate": SIMULATING,
            # A Zelux CS165MU whose firmware (v0.9.6) has no external trigger.
            "has_external_trigger": False,
            "stream_name": "rb_thorcam2",
            "timeout": 5.0,
            "image_dir": "C:\\Users\\cafmot\\Documents\\TempCameraImages\\Driver",
        },
        # Atom-rearrangement rig: a rearrangement camera and the Blink SLM in one
        # process, with the rearrangement coordinator streaming GPU-computed phase
        # frames straight to slm.update_mask() (no socket). Needs cupy/lap + a CUDA
        # GPU on this host to arm; status()/test() work without them. The SLM is
        # addressable on its own as get_device("Rb SLM").
        # CaFMOT Cam is here only because DCAM works from one process at a time on
        # this PC: while one process has it initialised, every other process's
        # dcamapi_init fails with NOCAMERA. FIXME: move it out once a shared
        # device process exists (docs/superpowers/specs/2026-11-16-shared-device-process-design.md).
        "Rb Rearrangement Rig": {
            "run_at_startup": True,
            "host": MACHINES["PH-BEAST"]["address"],
            "port": get_next_port(),
            "simulate": SIMULATING,
            "devices": {
                "Rb HamCam": {
                    "class": "pytweezer.drivers.imagemX2:ImagEMX2Camera",
                    "sim_class": "pytweezer.drivers.imagemX2:SimulatedTweezerArrayCamera",
                    "teardown": "close",
                    "role": "camera",
                    "serial_number": "000407",
                    "stream_name": "rb_hamcam",
                    "object_pixel_m": None,
                    "external_exposure": True,
                    "timeout": 20.0,
                    "temperature_measurement": "rb_hamcam_temperature",
                    "temperature_interval_s": 30.0,
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
                # ORCA-Fusion BT (C15440-20UP); the ORCA-R2 driver runs it unchanged.
                "CaFMOT Cam": {
                    "class": "pytweezer.drivers.orca:ORCACamera",
                    "sim_class": "pytweezer.drivers.orca:SimulatedORCACamera",
                    "teardown": "close",
                    "serial_number": "503326",
                    "stream_name": "cafmot_cam",
                    "object_pixel_m": None,
                    "external_exposure": True,
                    "global_reset": True,
                    "timeout": 5.0,
                    "temperature_measurement": "cafmot_cam_temperature",
                    "temperature_interval_s": 30.0,
                    # Air-cooled to -8 °C; blocked fans took it past 60 °C.
                    "max_temperature_c": 0.0,
                    "image_dir": "C:\\Users\\cafmot\\Documents\\TempCameraImages\\Driver",
                },
            },
            "coordinator": "pytweezer.coordinators.rearrangement:Rearrangement",
        },
        "IDS-1054B Scope": {
            "run_at_startup": False,
            "class": "pytweezer.drivers.ids1054b:IDS1054B",
            "sim_class": "pytweezer.drivers.ids1054b:SimulatedIDS1054B",
            "teardown": "close",
            "address": "TCPIP::10.59.3.50::3000::SOCKET",
            "host": MACHINES["PH-BEAST"]["address"],
            "port": get_next_port(),
            "simulate": SIMULATING,
        },
        "CaF ThorCam": {
            "run_at_startup": True,
            "class": "pytweezer.drivers.thorcam:ThorCam",
            "sim_class": "pytweezer.drivers.thorcam:SimulatedThorLabsCamera",
            "teardown": "close",
            "host": MACHINES["ph-bonesaw"]["address"],
            "port": get_next_port(),
            "simulate": SIMULATING,
            "serial_number": None,
            "trigger_mode": "bulb",
            "trigger_polarity": "rise",
            "binning": 16,
            "gain_db": 12.0,
            "stream_name": "caf_thorcam",
            "object_pixel_m": None,
            "timeout": 5.0,
            "image_dir": "C:/Users/CaFMOT/OneDrive - Imperial College London/caftweezers/ThorCamImages",
        },
        # The NI Cards tab: channels by MOTMaster alias, from each PC's hardware file.
        "CaF NI Cards": {
            "run_at_startup": False,
            "class": "pytweezer.drivers.ni_cards:NICards",
            "sim_class": "pytweezer.drivers.ni_cards:SimulatedNICards",
            "host": MACHINES["ph-bonesaw"]["address"],
            "port": get_next_port(),
            "simulate": SIMULATING,
            "hardware_file": r"C:\ControlPrograms\EDMSuite\DAQ\MoleculeMOTHardware.cs",
            "reserved": {
                "Wavemeter Lock": {
                    "channels": [
                        "v00Lock",
                        "v10Lock",
                        "v21Lock",
                        "v32Lock",
                        "bXLock",
                        "bXBeastLock",
                    ]
                },
                "Hardware Control": {"cards": ["tclBoard2"]},
            },
            # MOTMaster's safe-state channels (DAQMxAnalogSafeState).
            "protected": ["MOTCoilsCurrent", "transferCoils", "TweezerCoils"],
            "motmaster": "CaF MotMaster",
            "teardown": "close",
        },
        "Rb NI Cards": {
            "run_at_startup": False,
            "class": "pytweezer.drivers.ni_cards:NICards",
            "sim_class": "pytweezer.drivers.ni_cards:SimulatedNICards",
            "host": MACHINES["IC-CZC221CCHS"]["address"],
            "port": get_next_port(),
            "simulate": SIMULATING,
            "hardware_file": r"C:\ControlPrograms\EDMSuite\DAQ\RbTweezerHardware.cs",
            "reserved": {"Wavemeter Lock": {"channels": ["tRbD1Lock"]}},
            # Coil currents.
            "protected": ["tMOTcc", "tMagneticTrapcc"],
            "motmaster": "Rb MotMaster",
            "teardown": "close",
        },
    },
    # Background database loggers. Like devices, each entry names its Logger
    # subclass as "class" (a "module.path:ClassName" string); every entry runs
    # pytweezer/servers/logger_server.py, which builds that class and polls it on
    # "interval".
    # This is opt-in: nothing reaches the database unless a logger (or explicit
    # DBWriter/log() call) writes it.
    "Loggers": {
        "NI ADC Logger": {
            "run_at_startup": False,
            "class": "pytweezer.loggers.ni_adc_logger:NIADCLogger",
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
        #     "run_at_startup": True,
        #     "script": "../pytweezer/GUI/tweezer_browser.py"
        # },
        "StreamMonitor": {
            "run_at_startup": True,
            "script": "../pytweezer/GUI/streammonitor.py",
        },
        "Applet Launcher": {
            "run_at_startup": True,
            "script": "../pytweezer/GUI/applet_launcher.py",
        },
        # "H5 Manager": {
        #     "run_at_startup": False,
        #     "script": "../pytweezer/GUI/h5storage.py"
        # },
        "Property_Editor": {
            "run_at_startup": False,
            "script": "../pytweezer/GUI/property_editor.py",
        },
        # "Live Plot": {
        #     "run_at_startup": False,
        #     "script": "../pytweezer/GUI/viewers/live_plot.py"
        # },
        "Analysis Manager UI": {
            "run_at_startup": True,
            "script": "../pytweezer/GUI/analysismanager.py",
        },
    },
}


def load_class(path):
    """Import and return the object named by a ``"module.path:ClassName"`` string."""
    module_name, _, attr = path.partition(":")
    if not attr:
        raise ValueError(f"Class path {path!r} must be 'module.path:ClassName'")
    return getattr(importlib.import_module(module_name), attr)


def get_config():
    """Return the configuration dict.

    The single accessor for ``CONFIG`` used across the servers, drivers and GUI.
    Tests monkeypatch this (per importing module) to inject a fake config.
    """
    return CONFIG


def this_pc_address():
    """Return this PC's ``MACHINES`` address, or None if it is not listed.

    Windows reports hostnames in whatever case it likes, so the match ignores case.
    """
    hostname = socket.gethostname().lower()
    return next(
        (
            machine["address"]
            for name, machine in MACHINES.items()
            if name.lower() == hostname
        ),
        None,
    )
