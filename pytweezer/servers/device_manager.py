from sipyco.pc_rpc import RPCClient

class DeviceMananger:

    def __init__(self):
        self.devices: dict

    def get_device(self) -> RPCClient:
        pass