import numpy as np


class MockTemMicroscopeClient:
    """Mock TEM microscope client for development"""

    def __init__(self):
        self.service = self.Service()
        self.vacuum = self.Vacuum()
        self.optics = self.Optics()
        self.detectors = self.Detectors()
        self.acquisition = self.Acquisition()
        self.auto_functions = self.AutoFunctions()

    def connect(self, ip_address):
        print(f"Mock: Connected to microscope at {ip_address}")

    class Service:
        class System:
            name = "Mock TEM"
            serial_number = "MOCK-001"
            version = "1.0.0"

        system = System()

    class Vacuum:
        state = "READY"

    class Optics:
        is_beam_blanked = False

    class Detectors:
        camera_detectors = ["BM_CETA"]

        def get_camera_detector(self, cam):
            return type("Detector", (), {"is_operational": True})()

    class Acquisition:
        def acquire_camera_image(
            self, detector_type, frame_size, exposure_time, trail_mode
        ):
            # Create a mock image as a numpy array
            mock_image = (
                np.random.rand(frame_size, frame_size) * 255
            )  # Random grayscale image
            return mock_image.astype(np.uint8)  # Convert to 8-bit unsigned integer

        def acquire_stem_data(self, frame_size, exposure_time, trail_mode):
            mock_image = (
                np.random.rand(frame_size, frame_size) * 255
            )  # Random grayscale image
            return mock_image.astype(np.uint8)  # Convert to 8-bit unsigned integer

        def acquire_stem_image(self, frame_size, exposure_time, trail_mode):
            mock_image = (
                np.random.rand(frame_size, frame_size) * 255
            )  # Random grayscale image
            return mock_image.astype(np.uint8)  # Convert to 8-bit unsigned integer

    class AutoFunctions:
        def run_beam_tilt_auto_focus(self, settings):
            return type("FocusResult", (), {"focus_value": 1.0})()

        def run_objective_auto_stigmator(self, settings):
            return type("StigResult", (), {"stigmator_values": {"x": 0.0, "y": 0.0}})()


# Mock settings class
class RunBeamTiltAutoFocusSettings:
    def __init__(self, camera_type):
        self.camera_type = camera_type


class CameraType:
    BM_CETA = "BM_CETA"
