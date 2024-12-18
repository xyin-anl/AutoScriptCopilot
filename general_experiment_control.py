import os, time, json
import cv2, tifffile
import numpy as np
from typing import List, Dict
from nodeology.state import State
from nodeology.node import Node, as_node
from nodeology.workflow import Workflow
from nodeology.client import R2R_Client, PPLX_Client
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
    trial_mode: bool
    direct_mode: bool

    # Knowledge
    recommender_knowledge: str
    data_analysis_tool: str
    trial_mode_detector_type: str

    # Results
    current_image: str  # Path to the image file instead of np.ndarray
    image_quality_metrics: Dict[str, float]
    optimization_history: List[Dict]
    validation_response: Dict
    updated_parameters: Dict
    data_analysis_results: Dict


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


parameter_recommender = Node(
    prompt_template="""Recommender knowledge:
{recommender_knowledge}

Current microscope state and setup:
- Vacuum State: {vacuum_state}
- Beam State: {beam_state}
- Detector Type: {detector_type}

Recommend optimal imaging parameters for the current setup.
Consider:
1. Detector capabilities and limitations
2. Sample preservation requirements
3. Image quality requirements

Output as JSON:
{{
    "parameter_values": {{
        "magnification": float,
        "focus": float,
        "exposure_time": float,
        "frame_size": int
    }},
    "reasoning": str
}}""",
    sink="updated_parameters",
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
    trial_mode: bool = False,
) -> str:  # Return str instead of np.ndarray
    """Acquire image and save to file"""
    try:
        microscope = MicroscopeManager.get_instance().get_microscope(microscope_id)
        if detector_type == "empad":
            image = microscope.acquisition.acquire_stem_data(
                frame_size, exposure_time, trial_mode
            )
        elif detector_type in ["haadf"]:
            image = microscope.acquisition.acquire_stem_image(
                frame_size, exposure_time, trial_mode
            )
        else:
            image = microscope.acquisition.acquire_camera_image(
                detector_type, frame_size, exposure_time, trial_mode
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


# The quality accessment process itself could be a complex workflow that involves multiple steps and techniques
# This is a simplified version for demonstration purposes using only vision language model
quality_assessor = Node(
    prompt_template="""Analyze the TEM image quality:

Current Settings:
- Magnification: {magnification}
- Focus: {focus}
- Stigmator Values: {stigmator_values}
- Frame Size: {frame_size}
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
- Stigmator Values: {stigmator_values}
- Exposure Time: {exposure_time}
- Frame Size: {frame_size}

Suggest parameter adjustments to improve image quality.
Consider:
1. Previous optimization attempts
2. Quality improvement trends
3. Physical limits of the system

Output as JSON:
{{
    "parameter_values": {{
        "magnification": float,
        "focus": float,
        "exposure_time": float,
        "frame_size": int
    }},
    "reasoning": str,
    "stop_optimization": bool
}}""",
    sink="updated_parameters",
    sink_format="json",
    sink_transform=lambda x: json.loads(x),
)


@as_node(
    sink=[
        "magnification",
        "focus",
        "exposure_time",
        "frame_size",
        "trial_mode",
        "direct_mode",
        "detector_type",
    ]
)
def confirm_parameters(updated_parameters: Dict) -> tuple:
    """Display recommended parameters and get user confirmation/adjustments"""
    print("\nRecommended parameters:")
    for param, value in updated_parameters.get("parameter_values", {}).items():
        print(f"- {param}: {value}")

    if "reasoning" in updated_parameters:
        print(f"\nReasoning: {updated_parameters['reasoning']}")

    while True:
        choice = input("\nDo you want to: [a]ccept, [m]odify, or [c]ancel? ").lower()

        if choice == "a":
            params = updated_parameters["parameter_values"]
            print("Do you want to use trial mode? [y/n]")
            trial_mode = input().lower() == "y"
            print("Do you want to use direct mode? [y/n]")
            direct_mode = input().lower() == "y"
            return (
                params.get("magnification"),
                params.get("focus"),
                params.get("exposure_time"),
                params.get("frame_size"),
                trial_mode,
                direct_mode,
                (
                    params.get("trial_mode_detector_type")
                    if trial_mode
                    else params.get("detector_type")
                ),
            )
        elif choice == "m":
            modified_params = updated_parameters.copy()
            params = modified_params["parameter_values"]
            print("\nEnter new values (press Enter to keep current value):")

            for param in ["magnification", "focus", "exposure_time", "frame_size"]:
                current = params.get(param)
                new_value = input(f"{param} ({current}): ").strip()
                if new_value:
                    try:
                        params[param] = float(new_value)
                    except ValueError:
                        print(f"Invalid value for {param}, keeping original")

            print("Do you want to use trial mode? [y/n]")
            trial_mode = input().lower() == "y"
            print("Do you want to use direct mode? [y/n]")
            direct_mode = input().lower() == "y"

            return (
                params.get("magnification"),
                params.get("focus"),
                params.get("exposure_time"),
                params.get("frame_size"),
                trial_mode,
                direct_mode,
                (
                    params.get("trial_mode_detector_type")
                    if trial_mode
                    else params.get("detector_type")
                ),
            )
        elif choice == "c":
            params = updated_parameters["parameter_values"]
            if "stop_optimization" in params:
                params["stop_optimization"] = True
            print("Do you want to use trial mode? [y/n]")
            trial_mode = input().lower() == "y"
            print("Do you want to use direct mode? [y/n]")
            direct_mode = input().lower() == "y"
            return (
                params.get("magnification"),
                params.get("focus"),
                params.get("exposure_time"),
                params.get("frame_size"),
                trial_mode,
                direct_mode,
                (
                    params.get("trial_mode_detector_type")
                    if trial_mode
                    else params.get("detector_type")
                ),
            )


@as_node(sink=["data_analysis_results"])
def data_analysis(data_analysis_tool: str, data_analysis_results: Dict) -> Dict:
    """Run data analysis using specified tool"""
    try:
        # Use the specified data analysis tool to analyze the results
        # This is a placeholder for actual data analysis logic
        # Replace this with the actual implementation of the data analysis tool
        return {"data_analysis_results": data_analysis_results}
    except Exception as e:
        raise RuntimeError(f"Data analysis failed: {str(e)}")


# Create the workflow class
class TEMWorkflow(Workflow):
    """Workflow for automated TEM imaging"""

    def create_workflow(self):
        # Add nodes
        self.add_node("initialize", initialize_microscope)
        self.add_node("validate", validate_setup)
        self.add_node("recommend", parameter_recommender)
        self.add_node("acquire", acquire_image)
        self.add_node("align", auto_align)
        self.add_node("assess", quality_assessor)
        self.add_node("optimize", parameter_optimizer)
        self.add_node("confirm", confirm_parameters)
        self.add_node("analyze", data_analysis)
        # Add edges with conditional logic
        self.add_flow("initialize", "validate")
        self.add_conditional_flow(
            "validate",
            lambda state: state["validation_response"]["is_valid"],
            then="recommend",
            otherwise=END,
        )
        self.add_flow("recommend", "confirm")

        self.add_flow("confirm", "acquire")

        self.add_conditional_flow(
            "acquire",
            lambda state: not state["direct_mode"],
            then="assess",
            otherwise="analyze",
        )

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
            then="confirm",
            otherwise="analyze",
        )

        self.add_flow("analyze", END)

        # Set entry point
        self.set_entry("initialize")

        # Compile workflow with intervention points
        self.compile(checkpointer="memory")


if __name__ == "__main__":
    # Initialize workflow
    workflow = TEMWorkflow(
        state_defs=TEMState, llm_name="gpt-4o", vlm_name="gpt-4o", debug_mode=True
    )

    # Optional: Export workflow to yaml template and flowchart
    workflow.to_yaml("tem_workflow.yaml")
    # workflow.graph.get_graph().draw_mermaid_png(output_file_path="tem_workflow.png")

    # Provide expert knowledge for recommender or retrieve from knowledge base or search the internet
    expert_knowledge = """Knowledge about microscope settings and parameters..."""
    # # Define query
    # query = "How will the parameters (magnification, focus, exposure time, frame size, detector type) affect the STEM image quality?"
    # # Retrieve from knowledge base
    # rag_client = R2R_Client(model_name="gpt-4o", search_strategy="hybrid", rag_strategy="hyde")
    # expert_knowledge = rag_client([{"role":"user", "content":query}])
    # # Retrieve from internet
    # web_client = PPLX_Client(model_name="lama-3.1-sonar-large-128k-online")
    # expert_knowledge = web_client([{"role":"user", "content":query}])

    # Define initial state
    initial_state = {
        # Set initial imaging parameters
        "magnification": 20000,
        "focus": 0.0,
        "exposure_time": 0.1,
        "frame_size": 2048,
        "detector_type": "empad",
        "trial_mode_detector_type": "haadf",
        "recommender_knowledge": expert_knowledge,
    }

    # Run workflow
    result = workflow.run(initial_state)
