from edge_impulse.client import EdgeImpulseClient


class DeviceManager:
    """
    Remote device management
    """

    def __init__(self, config):
        self.client = EdgeImpulseClient(config)
        self.base_url = config.studio_url

    def list_devices(self):
        return self.client.get(f"{self.base_url}/devices")

    def trigger_data_acquisition(self, device_id: str):
        return self.client.post(
            f"{self.base_url}/devices/{device_id}/capture",
            {}
        )