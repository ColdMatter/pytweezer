from pytweezer.experiment.motmaster_client import MotMasterClient

from pytweezer.drivers.imagemX2 import ImagEMX2Camera
from pytweezer.drivers.slm import SLM
from pytweezer.drivers.thorcam import ThorCamera

print("Testing connections to devices...\n")

successful_devices = []
failed_devices = []


def test_device(name: str, create_device):
    try:
        device = create_device()
        device.close()
        successful_devices.append(name)
    except Exception as e:
        failed_devices.append((name, e))
        print(f"Failed to connect to {name}:", e)


test_device("MotMaster client", MotMasterClient)
test_device("ImagEM X2 camera", ImagEMX2Camera)
test_device("SLM", SLM)
test_device("MotCam ThorCamera", lambda: ThorCamera(id="motcam"))
test_device("TweezerCam ThorCamera", lambda: ThorCamera(id="tweezercam"))

if failed_devices:
    print("\nFailed to connect to the following devices:")
    for name, error in failed_devices:
        print(f"- {name}: {error}")
else:
    print("\nSuccessfully connected to all devices.")
