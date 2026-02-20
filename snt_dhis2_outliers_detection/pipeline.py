from pathlib import Path

from openhexa.sdk import current_run, parameter, pipeline, workspace
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    load_configuration_snt,
    pull_scripts_from_repository,
    run_notebook,
    run_report_notebook,
    validate_config,
    save_pipeline_parameters,
)


@pipeline("snt_dhis2_outliers_detection")
@parameter(
    "deviation_mean",
    name="Number of SD around the mean",
    help="Number of standard deviations around the mean (deault is 3)",
    type=int,
    default=3,
    required=False,
)
@parameter(
    "deviation_median",
    name="Number of MAD around the median",
    help="Number of MAD around the median (default is 3)",
    type=int,
    default=3,
    required=False,
)
@parameter(
    "deviation_iqr",
    name="IQR multiplier",
    help="IQR multiplier (default is 1.5)",
    type=float,
    default=1.5,
    required=False,
)
@parameter(
    "run_mg_partial",
    name="Run magic glasses partial method (up to MAD10)",
    help="Identifies outliers based on MAD15 and removes them, then identifies outliers based on MAD10",
    type=bool,
    default=False,
    required=False,
)
@parameter(
    "run_mg_complete",
    name="Run magic glasses complete method (up to seasonal3)",
    help="Picks up from magic glasses partial, and then applies sequentially seasonal 5 and seasonal 3",
    type=bool,
    default=False,
    required=False,
)
@parameter(
    "run_report_only",
    name="Run reporting only",
    help="This will only execute the reporting notebook",
    type=bool,
    default=False,
    required=False,
)
@parameter(
    "pull_scripts",
    name="Pull Scripts",
    help="Pull the latest scripts from the repository",
    type=bool,
    default=False,
    required=False,
)
def run_pipeline_task(
    deviation_mean: int,
    deviation_median: int,
    deviation_iqr: float,
    run_mg_partial: bool,
    run_mg_complete: bool,
    run_report_only: bool,
    pull_scripts: bool,
):
    """Outliers detection pipeline for SNT DHIS2 data."""
    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_dhis2_outliers_detection",
            report_scripts=["snt_dhis2_outliers_detection_report.ipynb"],
            code_scripts=["snt_dhis2_outliers_detection.ipynb"],
        )

    try:
        current_run.log_info("Starting SNT DHIS2 outliers detection pipeline...")

        # Define paths
        root_path = Path(workspace.files_path)
        pipeline_path = root_path / "pipelines" / "snt_dhis2_outliers_detection"
        data_path = root_path / "data" / "dhis2" / "outliers_detection"
        data_path.mkdir(parents=True, exist_ok=True)  # Ensure data path exists
        current_run.log_info(f"Pipeline path: {pipeline_path}")
        current_run.log_info(f"Data path: {data_path}")

        # Load configuration
        config_path = root_path / "configuration" / "SNT_config.json"
        snt_config = load_configuration_snt(config_path=config_path)
        validate_config(snt_config)
        country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]

        if not run_report_only:
            input_params = {
                "ROOT_PATH": Path(workspace.files_path).as_posix(),
                "DEVIATION_MEAN": deviation_mean,
                "DEVIATION_MEDIAN": deviation_median,
                "DEVIATION_IQR": deviation_iqr,
                "RUN_MAGIC_GLASSES_PARTIAL": run_mg_partial,
                "RUN_MAGIC_GLASSES_COMPLETE": run_mg_complete,
            }
            parameters_file = save_pipeline_parameters(
                pipeline_name="snt_dhis2_outliers_detection",
                parameters=input_params,
                output_path=data_path,
                country_code=country_code,
            )

            run_notebook(
                nb_path=pipeline_path / "code" / "snt_dhis2_outliers_detection.ipynb",
                out_nb_path=pipeline_path / "papermill_outputs",
                kernel_name="ir",
                parameters=input_params,
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
                country_code=country_code,
            )

            # Add files to Dataset
            add_files_to_dataset(
                dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_OUTLIERS_DETECTION"],
                country_code=country_code,
                file_paths=[
                    *[p for p in (data_path.glob(f"{country_code}_flagged_outliers_allmethods.parquet"))],
                    *[p for p in (data_path.glob(f"{country_code}_outlier_*.parquet"))],
                    parameters_file,
                ],
            )

        else:
            current_run.log_info(
                "🦘 Skipping Outliers Detection calculations, running only the reporting notebook."
            )

        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_dhis2_outliers_detection_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            country_code=country_code,
        )

        current_run.log_info("Pipeline finished!")

    except Exception as e:
        current_run.log_error(f"Notebook execution failed: {e}")
        raise


if __name__ == "__main__":
    run_pipeline_task()
