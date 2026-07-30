HOSTS = {
    "ph-beast": "10.59.3.1",
    "ic-czc221cchs": "10.59.3.2", # rb pc
    "ph-bonesaw": "10.59.3.5",
    "localhost": "127.0.0.1",
    
}

port_iterator = iter(range(7278, 99999))
get_next_port = lambda: int(next(port_iterator))

SIMULATING = False
LOCAL = False
SERVER_HOST = HOSTS["ph-beast"] if (not SIMULATING and not LOCAL) else HOSTS["localhost"]


CONFIG = {
    "Servers": {
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
    },
    "Devices": {
         "Rb MotMaster Server": {
            "active": True,
            "script": "../pytweezer/experiment/motmaster_server.py",
            "config_file": "rb_mm_config.json",
            "host": HOSTS["ic-czc221cchs"],
            "port": get_next_port(),
            "simulate": SIMULATING
        },
        "CaF MotMaster Server": {
            "active": True,
            "script": "../pytweezer/experiment/motmaster_server.py",
            "config_file": "caf_mm_config.json",
            "host": HOSTS["ph-bonesaw"],
            "port": get_next_port(),
            "simulate": SIMULATING
        },
        "Rb HamCam": {
            "active": True,
            "script": "../pytweezer/drivers/imagemx2.py",
            "host": SERVER_HOST,
            "port": get_next_port(),
            "simulate": SIMULATING,
            "stream_name": "rb_hamcam",
            "timeout": 5.0,
            "image_dir": "C:\\Users\\cafmot\\Documents\\TempCameraImages\\Driver"
        },
        "CaF HamCam": {
            "active": True,
            "script": "../pytweezer/drivers/imagemx2.py",
            "host": HOSTS["ph-bonesaw"],
            "port": get_next_port(),
            "simulate": SIMULATING,
            "stream_name": "caf_hamcam",
            "timeout": 5.0,
            "image_dir": "C:\\Users\\cafmot\\Documents\\TempCameraImages\\Driver"
        },
    },
}