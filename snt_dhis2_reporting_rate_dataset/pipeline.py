from pathlib import Path
from openhexa.sdk import current_run, pipeline, workspace, parameter

from snt_lib.snt_pipeline_utils import (
    pull_scripts_from_repository,
    add_files_to_dataset,
    load_configuration_snt,
    run_notebook,
    run_report_notebook,
    validate_config,
    dataset_file_exists,
    save_pipeline_parameters,
)


@pipeline("snt_dhis2_reporting_rate_dataset")
@parameter(
    "routine_data_choice",
    name="Routine data source",
    help="Select which routine data to use. "
    "'raw' loads formatted routine data, "
    "'imputed' loads outliers-imputed routine data, "
    "'outliers_removed' loads routine data with outliers removed.",
    multiple=False,
    choices=["raw", "imputed", "outliers_removed"],
    type=str,
    default="imputed",
    required=True,
)
@parameter(
    "run_report_only",
    name="Run reporting notebook only",
    help="This will execute only the reporting notebook. Important: "
    "this uses the outputs of the latest run of the full pipeline! Therefore, be aware that:"
    " if you have not run the full pipeline yet, or if the inputs have changed since the last run, "
    "the report may be outdated or incorrect.",
    type=bool,
    default=False,
    required=False,
)
@parameter(
    "pull_scripts",
    name="Pull notebooks from repository",
    help="Pull the latest notebooks from the GitHub repository. "
    "Note: this will overwrite any local changes to the notebooks!",
    type=bool,
    default=False,
    required=False,
)
def snt_dhis2_reporting_rate_dataset(
    routine_data_choice: str, run_report_only: bool, pull_scripts: bool
):
    """Orchestration function. Calls other functions within the pipeline."""
    if pull_scripts:
        current_run.log_info("Pulling pipeline notebooks from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_dhis2_reporting_rate_dataset",
            report_scripts=["snt_dhis2_reporting_rate_dataset_report.ipynb"],
            code_scripts=["snt_dhis2_reporting_rate_dataset.ipynb"],
        )

    try:
        # Set paths
        root_path = Path(workspace.files_path)
        pipeline_path = root_path / "pipelines" / "snt_dhis2_reporting_rate_dataset"
        data_path = root_path / "data" / "dhis2" / "reporting_rate"
        data_path.mkdir(parents=True, exist_ok=True)

        # Load configuration
        snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
        validate_config(snt_config)
        country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]

        if not run_report_only:
            routine_file_candidates = resolve_routine_file_candidates(
                country_code=country_code, routine_data_choice=routine_data_choice
            )
            if routine_data_choice == "raw":
                ds_outliers_id = snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_DATASET_FORMATTED"]
            else:
                ds_outliers_id = snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_OUTLIERS_IMPUTATION"]

            routine_file = next(
                (
                    filename
                    for filename in routine_file_candidates
                    if dataset_file_exists(ds_id=ds_outliers_id, filename=filename)
                ),
                None,
            )
            if routine_file is None:
                current_run.log_warning(
                    f"None of the expected routine files were found in dataset {ds_outliers_id}: "
                    f"{routine_file_candidates}. "
                    "Perhaps the outliers-imputation pipeline has not been run yet. Processing cannot continue."
                )
                return

            nb_parameters = {
                "SNT_ROOT_PATH": root_path.as_posix(),
                "ROUTINE_FILE": routine_file,
            }

            params_file = save_pipeline_parameters(
                pipeline_name="snt_dhis2_reporting_rate_dataset",
                parameters=nb_parameters,
                output_path=data_path,
                country_code=country_code,
            )
            current_run.log_info(f"Saved pipeline parameters to {params_file}")

            run_notebook(
                nb_path=pipeline_path / "code" / "snt_dhis2_reporting_rate_dataset.ipynb",
                out_nb_path=pipeline_path / "papermill_outputs",
                parameters=nb_parameters,
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
                country_code=country_code,
            )

            add_files_to_dataset(
                dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_REPORTING_RATE"],
                country_code=country_code,
                file_paths=[
                    *[p for p in (data_path.glob(f"{country_code}_reporting_rate_dataset.parquet"))],
                    *[p for p in (data_path.glob(f"{country_code}_reporting_rate_dataset.csv"))],
                    params_file,
                ],
            )

        else:
            current_run.log_info("Skipping calculations, running only the reporting.")

        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_dhis2_reporting_rate_dataset_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            country_code=country_code,
        )

        current_run.log_info("Pipeline completed successfully!")

    except Exception as e:
        current_run.log_error(f"Pipeline failed: {e}")
        raise


def resolve_routine_file_candidates(country_code: str, routine_data_choice: str) -> list[str]:
    """Returns ordered candidate filenames for a routine data choice."""
    if routine_data_choice == "raw":
        return [f"{country_code}_routine.parquet"]

    if routine_data_choice == "imputed":
        return [
            f"{country_code}_routine_outliers_imputed.parquet",
            f"{country_code}_routine_outliers-mean_imputed.parquet",
            f"{country_code}_routine_outliers-median_imputed.parquet",
            f"{country_code}_routine_outliers-iqr_imputed.parquet",
            f"{country_code}_routine_outliers-trend_imputed.parquet",
            f"{country_code}_routine_outliers-mg-partial_imputed.parquet",
            f"{country_code}_routine_outliers-mg-complete_imputed.parquet",
        ]

    if routine_data_choice == "outliers_removed":
        return [
            f"{country_code}_routine_outliers_removed.parquet",
            f"{country_code}_routine_outliers-mean_removed.parquet",
            f"{country_code}_routine_outliers-median_removed.parquet",
            f"{country_code}_routine_outliers-iqr_removed.parquet",
            f"{country_code}_routine_outliers-trend_removed.parquet",
            f"{country_code}_routine_outliers-mg-partial_removed.parquet",
            f"{country_code}_routine_outliers-mg-complete_removed.parquet",
        ]

    raise ValueError(f"Unknown routine data choice: {routine_data_choice}")


if __name__ == "__main__":
    snt_dhis2_reporting_rate_dataset()
