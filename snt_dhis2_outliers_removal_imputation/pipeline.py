from datetime import datetime
from pathlib import Path

# Import the specific run_notebook function from snt_pipeline_utils.py
from snt_lib.snt_pipeline_utils import (
    run_notebook,
    run_report_notebook,
    generate_html_report,
    validate_config,
    load_configuration_snt,
    add_files_to_dataset
)  

from openhexa.sdk import current_run, pipeline, workspace, parameter


@pipeline(name="snt-dhis2-outliers-removal-imputation") 
@parameter(
    "outlier_method", 
    name="Method used for outlier detection",
    help="Outliers have been detected in upstream pipeline 'Outliers Detection' using different methods.",
    choices=["mean3sd", "median3mad", "iqr1-5", "magic_glasses_partial", "magic_glasses_complete"], 
    type=str,
    required=True
)
@parameter(
    "run_report_only",
    name="Run reporting only",
    help="This will only execute the reporting notebook",
    type=bool,
    default=False,
    required=False,
)
def run_pipeline_task(outlier_method: str,
                      run_report_only: bool):  
    """Orchestration function. Calls other functions within the pipeline."""
    try:
        current_run.log_info("Starting SNT DHIS2 Outliers Removal and Imputation Pipeline...")

        # Define paths and notebook names
        notebook_name = "SNT_dhis2_outliers_removal_imputation" 
        report_notebook_name = "SNT_dhis2_outliers_removal_imputation_report"
        folder_name = "dhis2_outliers_removal_imputation" 
        root_path = Path(workspace.files_path)
        pipeline_path = root_path / "pipelines" / folder_name
        data_path = root_path / "data" / folder_name
        current_run.log_info(f"Pipeline path: {pipeline_path}")
        current_run.log_info(f"Data path: {data_path}")

        # Load configuration
        config_path = root_path / "configuration" / "SNT_config.json"
        snt_config = load_configuration_snt(config_path=config_path)
        validate_config(snt_config)
        country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]

        if not run_report_only:
            input_notebook_path = pipeline_path / "code" / f"{notebook_name}.ipynb"  
            papermill_outputs_dir = pipeline_path / "papermill_outputs"  

            # Run the notebook  
            run_notebook(
                nb_path=input_notebook_path,
                out_nb_path=papermill_outputs_dir,
                kernel_name="ir",
                parameters={
                "OUTLIER_METHOD": outlier_method, 
                "ROOT_PATH": str(Path(workspace.files_path)), 
                }
            )
            # Add files to Dataset
            add_files_to_dataset(
                dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_OUTLIERS_REMOVAL_IMPUTATION"],
                country_code=country_code,
                file_paths=[
                    *[p for p in (data_path.glob(f"{country_code}_routine_outliers-*_*.parquet"))],
                    *[p for p in (data_path.glob(f"{country_code}_routine_outliers-*_*.csv"))]
                ],
            )
            
        else:
            current_run.log_info(
                "🦘 Skipping outliers removal and imputation calculations, running only the reporting notebook."
            )

        run_report_notebook(
            nb_file=pipeline_path / "reporting" / f"{report_notebook_name}.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
        )
        
        current_run.log_info("Pipeline finished!")

    except Exception as e:
        current_run.log_error(f"❌ Notebook execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_pipeline_task()