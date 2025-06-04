import json
from datetime import datetime
from pathlib import Path

import papermill as pm
from openhexa.sdk import current_run, pipeline, workspace, parameter


# ====================== NEW: PIPELINE PARAMETERS ======================
@pipeline("dhis2_reporting_rate")
@parameter(
    "config_file_name",
    name="Configuration File",
    help="Name of the JSON configuration file (must be in /configuration/ folder)",
    type=str,
    default="SNT_config_COD.json",
    required=True,
)
@parameter(
    "reporting_rate_threshold",
    name="Reporting Rate Threshold",
    help="Threshold for considering reporting rate 'good' (0-1)",
    type=float,
    default=0.8,
    required=True,
)
def dhis2_reporting_rate(config_file_name: str, reporting_rate_threshold: float):
    """Pipeline for calculating DHIS2 reporting rates with configurable parameters."""
    # ====================== END NEW PARAMETERS ======================

    # ====================== DEBUG BLOCK ======================
    current_run.log_info("🚀 STARTING DEBUG OUTPUT")
    current_run.log_info(f"\n⚙️ PIPELINE PARAMETERS:")
    current_run.log_info(f"Config file: {config_file_name}")
    current_run.log_info(f"Reporting threshold: {reporting_rate_threshold}")

    # 1. Path verification
    current_run.log_info(f"\n📁 PATH VERIFICATION:")
    current_run.log_info(f"workspace.files_path = {workspace.files_path}")
    current_run.log_info(f"Current working directory = {Path.cwd()}")

    # 2. Config file check (UPDATED to use parameter)
    config_path = Path(workspace.files_path) / "configuration" / config_file_name
    current_run.log_info(f"\n🔧 CONFIG FILE:")
    current_run.log_info(f"Config path = {config_path}")
    current_run.log_info(f"Config exists = {config_path.exists()}")
    # ==================== END DEBUG BLOCK ====================

    try:
        # Set paths
        root_path = Path(workspace.files_path)
        pipeline_path = root_path / "pipelines/dhis2_reporting_rate"
        papermill_output_path = pipeline_path / "papermill_outputs"
        papermill_output_path.mkdir(parents=True, exist_ok=True)

        # Load configuration
        config = load_configuration(config_path)

        # Notebook details
        nb_name = "01_reporting_rate"
        nb_path = pipeline_path / "code" / f"{nb_name}.ipynb"

        # Execute notebook (UPDATED to use parameter)
        current_run.log_info("Starting notebook execution...")
        pm.execute_notebook(
            input_path=str(nb_path),
            output_path=str(
                papermill_output_path / f"OUTPUT_{nb_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"
            ),
            parameters={
                "ROOT_PATH": str(root_path),
                "CONFIG_FILE_NAME": config_file_name,  # UPDATED: Using parameter
                "REPORTING_RATE_THRESHOLD": reporting_rate_threshold,  # UPDATED: Using parameter
            },
            kernel_name="ir",
            request_save_on_cell_execute=False,
        )
        current_run.log_info("✅ Pipeline completed successfully!")

    except Exception as e:
        current_run.log_error(f"❌ Pipeline failed: {str(e)}")
        raise


def load_configuration(config_path: Path) -> dict:
    """Load configuration with fallback values."""
    try:
        with config_path.open() as f:
            config = json.load(f)

        # Ensure required structure exists
        config.setdefault("SNT_CONFIG", {})
        snt_config = config["SNT_CONFIG"]

        # Set fallback values if missing
        snt_config.setdefault("COUNTRY_CODE", "COD")
        snt_config.setdefault("DHIS2_ADMINISTRATION_1", "PROVINCE")
        snt_config.setdefault("DHIS2_ADMINISTRATION_2", "ZONE")

        return config

    except Exception as e:
        current_run.log_warning(f"⚠️ Using fallback config - {str(e)}")
        return {
            "SNT_CONFIG": {
                "COUNTRY_CODE": "COD",
                "DHIS2_ADMINISTRATION_1": "PROVINCE",
                "DHIS2_ADMINISTRATION_2": "ZONE",
            }
        }


if __name__ == "__main__":
    # For local testing only
    dhis2_reporting_rate(config_file_name="SNT_config_COD.json", reporting_rate_threshold=0.8)
