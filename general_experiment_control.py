import os, time, json
import cv2, tifffile
import numpy as np
from typing import List, Dict
from nodeology.state import State
from nodeology.node import Node, as_node
from nodeology.workflow import Workflow
from langgraph.graph import END

from microscope_manager import MicroscopeManager

# Actual microscope client may be used in production
# from autoscript_tem_microscope_client import TemMicroscopeClient
# from autoscript_tem_microscope_client.enumerations import *
# from autoscript_tem_microscope_client.structures import *

# Use mock microscope client for development
from mock_tem_client import (
    MockTemMicroscopeClient,
    CameraType,
    RunBeamTiltAutoFocusSettings,
)


# Define the state class
class TEMState(State):
    """State for TEM experiment workflow"""

    # Microscope connection (store ID or reference instead of client object)
    microscope_id: str  # Store ID or connection reference
    microscope_info: Dict[str, str]

    # System status
    vacuum_state: str
    beam_state: bool
    detector_status: Dict[str, bool]

    # Optical settings
    magnification: float
    focus: float
    stigmator_values: Dict[str, float]

    # Acquisition settings
    exposure_time: float
    frame_size: int
    detector_type: str

    # Results
    current_image: str  # Path to the image file instead of np.ndarray
    image_quality_metrics: Dict[str, float]
    optimization_history: List[Dict]
    validation_response: Dict
    updated_parameters: Dict


# Define validation node
validate_setup = Node(
    prompt_template="""Analyze microscope setup:
Vacuum State: {vacuum_state}
Beam State: {beam_state}
Detector Status: {detector_status}

Validate against required conditions:
1. Vacuum must be in READY state
2. Beam must be active and properly controlled
3. Selected detector must be operational

Output validation results as JSON:
{{
    "is_valid": bool,
    "issues": [str],
    "recommendations": [str]
}}""",
    sink="validation_response",
    sink_format="json",
    sink_transform=lambda x: json.loads(x),
)


# Function to initialize microscope
@as_node(
    sink=[
        "microscope_id",
        "microscope_info",
        "vacuum_state",
        "beam_state",
        "detector_status",
    ]
)
def initialize_microscope(ip_address: str = "localhost") -> tuple:
    """Initialize microscope connection and get basic status"""
    try:
        microscope = MockTemMicroscopeClient()
        microscope.connect(ip_address)

        # Store microscope in manager and get reference ID
        microscope_id = MicroscopeManager.get_instance().set_microscope(microscope)

        # Get microscope info
        info = {
            "name": microscope.service.system.name,
            "serial": microscope.service.system.serial_number,
            "version": microscope.service.system.version,
        }

        # Get system status
        vacuum = microscope.vacuum.state
        beam = not microscope.optics.is_beam_blanked

        # Get detector status
        detectors = {}
        for cam in microscope.detectors.camera_detectors:
            detector = microscope.detectors.get_camera_detector(cam)
            detectors[cam] = detector.is_operational

        return microscope_id, info, vacuum, beam, detectors

    except Exception as e:
        raise RuntimeError(f"Failed to initialize microscope: {str(e)}")


@as_node(sink=["current_image"])
def acquire_image(
    microscope_id: str,
    detector_type: str,
    frame_size: int,
    exposure_time: float,
) -> str:  # Return str instead of np.ndarray
    """Acquire image and save to file"""
    try:
        microscope = MicroscopeManager.get_instance().get_microscope(microscope_id)
        image = microscope.acquisition.acquire_camera_image(
            detector_type, frame_size, exposure_time
        )

        # Convert to numpy array if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Save image to file with timestamp or unique identifier
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        tiff_image_path = os.path.join("data", "images", f"tem_image_{timestamp}.tiff")
        png_image_path = os.path.join("data", "images", f"tem_image_{timestamp}.png")

        # Ensure directory exists
        os.makedirs(os.path.dirname(tiff_image_path), exist_ok=True)
        os.makedirs(os.path.dirname(png_image_path), exist_ok=True)

        # Save image (using appropriate format/library depending on your needs)
        tifffile.imwrite(tiff_image_path, image)
        cv2.imwrite(png_image_path, image)

        return png_image_path

    except Exception as e:
        raise RuntimeError(f"Image acquisition failed: {str(e)}")


@as_node(sink=["focus", "stigmator_values"])
def auto_align(microscope_id: str) -> tuple:
    """Run automatic alignment procedures"""
    try:
        microscope = MicroscopeManager.get_instance().get_microscope(microscope_id)

        # Configure auto functions
        settings = RunBeamTiltAutoFocusSettings(CameraType.BM_CETA)

        # Run auto focus
        focus_result = microscope.auto_functions.run_beam_tilt_auto_focus(settings)

        # Run stigmator correction
        stig_result = microscope.auto_functions.run_objective_auto_stigmator(settings)

        return focus_result.focus_value, stig_result.stigmator_values

    except Exception as e:
        raise RuntimeError(f"Auto-alignment failed: {str(e)}")


quality_assessor = Node(
    prompt_template="""Analyze the TEM image quality:

Current Settings:
- Magnification: {magnification}
- Focus: {focus}
- Exposure: {exposure_time}

Evaluate image for:
1. Focus quality (check Fresnel fringes, edge sharpness)
2. Astigmatism correction
3. Brightness and contrast
4. Signal-to-noise ratio
5. Stage drift effects
6. Beam damage indicators

Output analysis as JSON:
{{
    "quality_score": float (0-10),
    "issues": [
        {{
            "type": str,
            "severity": str,
            "description": str
        }}
    ],
    "improvements": [
        {{
            "parameter": str,
            "suggestion": str,
            "reasoning": str
        }}
    ]
}}""",
    sink="image_quality_metrics",
    sink_format="json",
    image_keys=["current_image"],
    sink_transform=lambda x: json.loads(x),
)


parameter_optimizer = Node(
    prompt_template="""Review current imaging performance:
Quality Metrics: {image_quality_metrics}
Optimization History: {optimization_history}

Current Parameters:
- Magnification: {magnification}
- Focus: {focus}
- Exposure Time: {exposure_time}

Suggest parameter adjustments to improve image quality.
Consider:
1. Previous optimization attempts
2. Quality improvement trends
3. Physical limits of the system

Output as JSON:
{{
    "parameter_updates": {{
        "magnification": float,
        "focus": float,
        "exposure_time": float
    }},
    "reasoning": str,
    "stop_optimization": bool
}}""",
    sink="updated_parameters",
    sink_format="json",
    sink_transform=lambda x: json.loads(x),
)


# Create the workflow class
class TEMWorkflow(Workflow):
    """Workflow for automated TEM imaging"""

    def create_workflow(self):
        # Add nodes
        self.add_node("initialize", initialize_microscope)
        self.add_node("validate", validate_setup)
        self.add_node("acquire", acquire_image)
        self.add_node("align", auto_align)
        self.add_node("assess", quality_assessor)
        self.add_node("optimize", parameter_optimizer)

        # Add edges with conditional logic
        self.add_flow("initialize", "validate")
        self.add_conditional_flow(
            "validate",
            lambda state: state["validation_response"]["is_valid"],
            then="acquire",
            otherwise=END,
        )

        self.add_flow("acquire", "assess")

        self.add_conditional_flow(
            "assess",
            lambda state: state["image_quality_metrics"]["quality_score"] < 8.0,
            then="align",
            otherwise="optimize",
        )

        self.add_flow("align", "acquire")

        self.add_conditional_flow(
            "optimize",
            lambda state: not state["updated_parameters"]["stop_optimization"],
            then="acquire",
            otherwise=END,
        )

        # Set entry point
        self.set_entry("initialize")

        # Compile workflow with intervention points
        self.compile(interrupt_before=["optimize"], checkpointer="memory")


if __name__ == "__main__":
    # Initialize workflow
    workflow = TEMWorkflow(
        state_defs=TEMState, llm_name="gpt-4o", vlm_name="gpt-4o", debug_mode=True
    )

    # # Optional: Export workflow flowchart and template
    # workflow.to_yaml("tem_workflow.yaml")
    # workflow.graph.get_graph().draw_mermaid_png(output_file_path="tem_workflow.png")

    # Define initial state
    initial_state = {
        # Set initial imaging parameters
        "magnification": 20000,
        "exposure_time": 0.1,
        "frame_size": 2048,
        "detector_type": "BM_CETA",
        # Initialize empty tracking lists
        "optimization_history": [],
        "image_quality_metrics": {"quality_score": 0},
    }

    # Run workflow
    result = workflow.run(initial_state)
