
from datetime import datetime
from pathlib import Path
import papermill as pm  # Tool that executes notebooks with parameters
from openhexa.sdk import current_run, pipeline, workspace, parameter

@pipeline("DHIS2 Outliers Removal and Imputation") 

# @parameter(
#     "config_file_name",
#     name="Configuration File Name",
#     help="Name of the JSON configuration file (must be in /configuration/ folder)",
#     type=str,
#     default="SNT_config.json",
#     required=True
# )
@parameter(
    "outlier_method", 
    name="Method used for outlier detection",
    help="Outliers have been detected in upstream pipeline 'Outliers Detection' using different methods.",
    choices=["mean3sd", "median3mad", "iqr1-5", "magic_glasses_partial", "magic_glasses_complete"], 
    type=str,
    # default="threshold",
    required=True
)

def run_notebook(outlier_method: str):
    """
    Runs a notebook with specified parameters.
    
    HOW TO CUSTOMIZE:
    1. Change pipeline name above
    2. Add/remove parameters above
    3. Set notebook location below
    4. Configure parameters passed to notebook
    """
    
    try:
        # Format: workspace_files/pipelines/[pipeline_name]/code/[notebook_name].ipynb
        notebook_name = "01_outliers_removal_imputation"  # <<< CHANGE THIS to your notebook's name
        pipeline_folder_name = "dhis2_outliers_removal_imputation" # <<< CHANGE THIS to your pipeline's FOLDER name
        input_notebook_path = Path(workspace.files_path) / "pipelines" / pipeline_folder_name / "code" / f"{notebook_name}.ipynb"
        
        papermill_outputs_dir = Path(workspace.files_path) / "pipelines" / pipeline_folder_name / "papermill_outputs"
        papermill_outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a descriptive output filename with timestamp
        executed_notebook_path = papermill_outputs_dir / f"EXECUTED_{notebook_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"
        
        notebook_parameters = {
            # "CONFIG_FILE_NAME": config_file_name  
            "OUTLIER_METHOD": outlier_method,      
            "ROOT_PATH": str(Path(workspace.files_path)),  # Standard workspace path
        }
        
        # Execute the notebook (usually no changes needed here)
        current_run.log_info(f"⚙️ Running notebook: {input_notebook_path}")
        pm.execute_notebook(
            input_path=str(input_notebook_path),
            output_path=str(executed_notebook_path),
            parameters=notebook_parameters,
            kernel_name="ir",  
            request_save_on_cell_execute=False
        )
        
        current_run.log_info(f"✅ Success! Executed notebook saved to: {executed_notebook_path}")
        
    except Exception as e:
        current_run.log_error(f"❌ Notebook execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_notebook()