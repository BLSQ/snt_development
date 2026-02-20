from pathlib import Path

from openhexa.sdk import current_run, parameter, pipeline, workspace
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    load_configuration_snt,
    pull_scripts_from_repository,
    run_notebook,
    run_report_notebook,
    validate_config,
    create_outliers_db_table,
    save_pipeline_parameters,
)


@pipeline("snt_dhis2_outliers_imputation_path")
@parameter(
    "deviation_mean",
    name="Number of SD around the mean",
    help="Number of standard deviations around the mean (PATH deault: 10)",
    type=int,
    default=10,
    required=False,
)
@parameter(
    "push_db",
    name="Push outliers table to DB",
    help="Push outliers table to DB",
    type=bool,
    default=True,
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
def snt_dhis2_outliers_imputation_path(
    deviation_mean: int,
    push_db: bool,
    run_report_only: bool,
    pull_scripts: bool,
):
    """Outliers imputation pipeline for SNT DHIS2 data."""
    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_dhis2_outliers_imputation_path",
            report_scripts=["snt_dhis2_outliers_imputation_path_report.ipynb"],
            code_scripts=["snt_dhis2_outliers_imputation_path.ipynb"],
        )

    try:
        current_run.log_info("Starting SNT DHIS2 outliers imputation PATH method pipeline...")

        # Define paths
        root_path = Path(workspace.files_path)
        pipeline_path = root_path / "pipelines" / "snt_dhis2_outliers_imputation_path"
        data_path = root_path / "data" / "dhis2" / "outliers_imputation"

        pipeline_path.mkdir(parents=True, exist_ok=True)  # Ensure pipeline path exists
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
            }
            parameters_file = save_pipeline_parameters(
                pipeline_name="snt_dhis2_outliers_imputation_path",
                parameters=input_params,
                output_path=data_path,
                country_code=country_code,
            )

            run_notebook(
                nb_path=pipeline_path / "code" / "snt_dhis2_outliers_imputation_path.ipynb",
                out_nb_path=pipeline_path / "papermill_outputs",
                kernel_name="ir",
                parameters=input_params,
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
                country_code=country_code,
            )

            # Add files to Dataset
            add_files_to_dataset(
                dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_OUTLIERS_IMPUTATION"],
                country_code=country_code,
                file_paths=[
                    *data_path.glob(f"{country_code}_routine_outliers*.parquet"),
                    parameters_file,
                ],
            )

            # Create consolidated outliers DB table
            if push_db:
                create_outliers_db_table(country_code=country_code, data_path=data_path)

        else:
            current_run.log_info("Skipping outliers calculations, running only the reporting notebook.")

        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_dhis2_outliers_imputation_path_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            country_code=country_code,
        )

        # TODO: For the shiny app, we can think in a procedure to collect all
        # outlier tables and push all available results to a DB table
        current_run.log_info("Pipeline finished successfully.")

    except Exception as e:
        current_run.log_error(f"Notebook execution failed: {e}")
        raise


if __name__ == "__main__":
    snt_dhis2_outliers_imputation_path()
