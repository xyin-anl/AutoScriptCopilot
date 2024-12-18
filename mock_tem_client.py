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
            self, detector_type, frame_size, exposure_time, trial_mode
        ):
            # Create a mock image as a numpy array
            mock_image = (
                #np.random.rand(frame_size, frame_size) * 255
                _generate_periodic_atomic_structure(frame_size, exposure_time, trial_mode)

            )  # Random grayscale image
            return mock_image.astype(np.uint8)  # Convert to 8-bit unsigned integer

        def acquire_stem_data(self, frame_size, exposure_time, trial_mode):
            mock_image = (
                #np.random.rand(frame_size, frame_size) * 255
                _generate_periodic_atomic_structure(frame_size, exposure_time, trial_mode)

            )  # Random grayscale image
            return mock_image.astype(np.uint8)  # Convert to 8-bit unsigned integer

        def acquire_stem_image(self, frame_size, exposure_time, trial_mode):
            mock_image = (
                
                _generate_periodic_atomic_structure(frame_size, exposure_time, trial_mode)
                #np.random.rand(frame_size, frame_size) * 255
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

def _generate_periodic_atomic_structure(frame_size, exposure_time, trial_mode):
    """Generate a mock image simulating periodic atomic structures with consistent Gaussian widths."""

    # Parameters for atomic structure
    lattice_constant = frame_size // 10  # Distance between atoms
    atom_width = np.random.uniform(0.5, 5.0)  # Randomly generate width between 0.5 and 3
    
    # Create meshgrid for the entire image
    y, x = np.meshgrid(np.arange(frame_size), np.arange(frame_size))
    
    # Calculate number of atoms needed in each dimension
    n_atoms = int(frame_size // lattice_constant + 2)  # Add extra atoms for edges and ensure integer
    
    # Create grid of atom positions
    atom_positions_x = np.linspace(-lattice_constant, frame_size + lattice_constant, n_atoms)
    atom_positions_y = np.linspace(-lattice_constant, frame_size + lattice_constant, n_atoms)
    
    # Add random offsets and displacements to positions
    random_offset_x = np.random.uniform(-lattice_constant/4, lattice_constant/4)
    random_offset_y = np.random.uniform(-lattice_constant/4, lattice_constant/4)

    atom_positions_x = atom_positions_x + random_offset_x + np.random.normal(0, lattice_constant/100, n_atoms)
    atom_positions_y = atom_positions_y + random_offset_y + np.random.normal(0, lattice_constant/100, n_atoms)
    
    # Reshape coordinates for broadcasting
    x_coords = x[:, :, np.newaxis, np.newaxis]
    y_coords = y[:, :, np.newaxis, np.newaxis]
    atom_x = atom_positions_x[np.newaxis, np.newaxis, :, np.newaxis]
    atom_y = atom_positions_y[np.newaxis, np.newaxis, np.newaxis, :]
    
    # Calculate all Gaussians at once
    gaussians = np.exp(-((x_coords - atom_x)**2 + (y_coords - atom_y)**2) / (2 * atom_width**2))
    image = np.sum(gaussians, axis=(2, 3))

    # Normalize and convert to uint8
    image = ((image - image.min()) / (image.max() - image.min()) * 255)
    return image.astype(np.uint8)