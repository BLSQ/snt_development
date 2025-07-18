# from datetime import datetime
from pathlib import Path

from snt_lib.snt_pipeline_utils import (
    run_notebook,
    run_report_notebook,
    # generate_html_report,
    validate_config,
    load_configuration_snt,
    add_files_to_dataset
)  

from openhexa.sdk import current_run, pipeline, workspace, parameter

# A.4 DHIS2 Outliers Detection
@pipeline(name="SNT DHIS2 Outliers Detection") 
@parameter(
    "deviation_mean",
    name="Number of SD around the mean",
    help="Number of standard deviations around the mean (deault is 3)",
    type=int,
    default=3,
    required=False
)
@parameter(
    "deviation_median", 
    name="Number of MAD around the median",
    help="Number of MAD around the median (default is 3)",
    type=int,
    default=3,
    required=False
)
@parameter(
    "deviation_iqr", 
    name="IQR multiplier",
    help="IQR multiplier (default is 1.5)",
    type=float,
    default=1.5,
    required=False
)
@parameter(
    "run_mg_partial", 
    name="Run Magic Glasses Partial method (up to MAD10)",
    help="Identifies outliers based on MAD15 and removes them, then identifies outliers based on MAD10",
    type=bool,
    default=False,
    required=False
)
@parameter(
    "run_mg_complete", 
    name="Run Magic Glasses Complete method (up to seasonal3)",
    help="Picks up from Magic Glasses Partial, and then applies sequentially seasonal 5 and seasonal 3",
    type=bool,
    default=False,
    required=False
)
@parameter(
    "run_report_only",
    name="Run reporting only",
    help="This will only execute the reporting notebook",
    type=bool,
    default=False,
    required=False,
)
def run_pipeline_task(deviation_mean: int,
                      deviation_median: int,
                      deviation_iqr: float,
                      run_mg_partial: bool,
                      run_mg_complete: bool,
                      run_report_only: bool):  
    """Orchestration function. Calls other functions within the pipeline."""
    try:
        current_run.log_info("Starting SNT DHIS2 Outliers Detection Pipeline...")

        # Define paths and notebook names
        notebook_name = "SNT_dhis2_outliers_detection" 
        report_notebook_name = "SNT_dhis2_outliers_detection_report"
        folder_name = "dhis2_outliers_detection" 
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
                    "ROOT_PATH": str(Path(workspace.files_path)),
                    "DEVIATION_MEAN": deviation_mean,
                    "DEVIATION_MEDIAN": deviation_median,
                    "DEVIATION_IQR": deviation_iqr,
                    "RUN_MAGIC_GLASSES_PARTIAL": run_mg_partial,
                    "RUN_MAGIC_GLASSES_COMPLETE": run_mg_complete
                }
            )
            # Add files to Dataset
            add_files_to_dataset(
                dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_OUTLIERS_DETECTION"],
                country_code=country_code,
                file_paths=[
                    *[p for p in (data_path.glob(f"{country_code}_flagged_outliers_allmethods.parquet"))],
                    *[p for p in (data_path.glob(f"{country_code}_outlier_*.parquet"))]
                ],
            )
            
        else:
            current_run.log_info(
                "🦘 Skipping Outliers Detection calculations, running only the reporting notebook."
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