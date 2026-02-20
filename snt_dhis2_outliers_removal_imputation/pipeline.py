from pathlib import Path

from openhexa.sdk import current_run, parameter, pipeline, workspace
from snt_lib.snt_pipeline_utils import (
    pull_scripts_from_repository,
    add_files_to_dataset,
    load_configuration_snt,
    run_notebook,
    run_report_notebook,
    validate_config,
    save_pipeline_parameters,
)


@pipeline("snt-dhis2-outliers-removal-imputation")
@parameter(
    "outlier_method",
    name="Method used for outlier detection",
    help="Outliers have been detected in upstream pipeline 'Outliers Detection' using different methods.",
    choices=["mean3sd", "median3mad", "iqr1-5", "magic_glasses_partial", "magic_glasses_complete"],
    type=str,
    default="mean3sd",
    required=True,
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
def run_pipeline_task(outlier_method: str, run_report_only: bool, pull_scripts: bool):
    """Orchestration function. Calls other functions within the pipeline."""
    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_dhis2_outliers_removal_imputation",
            report_scripts=["snt_dhis2_outliers_removal_imputation_report.ipynb"],
            code_scripts=["snt_dhis2_outliers_removal_imputation.ipynb"],
        )
    try:
        current_run.log_info("Starting SNT DHIS2 outliers removal and imputation pipeline...")

        # Define paths and notebook names
        root_path = Path(workspace.files_path)
        pipeline_path = root_path / "pipelines" / "snt_dhis2_outliers_removal_imputation"
        data_path = root_path / "data" / "dhis2" / "outliers_removal_imputation"
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
                "OUTLIER_METHOD": outlier_method,
                "ROOT_PATH": root_path.as_posix(),
            }
            parameters_file = save_pipeline_parameters(
                pipeline_name="snt_dhis2_outliers_removal_imputation",
                parameters=input_params,
                output_path=data_path,
                country_code=country_code,
            )

            run_notebook(
                nb_path=pipeline_path / "code" / "snt_dhis2_outliers_removal_imputation.ipynb",
                out_nb_path=pipeline_path / "papermill_outputs",
                kernel_name="ir",
                parameters=input_params,
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
                country_code=country_code,
            )

            add_files_to_dataset(
                dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_OUTLIERS_REMOVAL_IMPUTATION"],
                country_code=country_code,
                file_paths=[
                    *[p for p in (data_path.glob(f"{country_code}_routine_outliers-*_*.parquet"))],
                    *[p for p in (data_path.glob(f"{country_code}_routine_outliers-*_*.csv"))],
                    parameters_file,
                ],
            )

        else:
            current_run.log_info(
                "🦘 Skipping outliers removal and imputation calculations,"
                " running only the reporting notebook."
            )

        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_dhis2_outliers_removal_imputation_report.ipynb",
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
