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
    "outliers_method",
    name="Outlier processing method",
    help="Specify which method was used to detect outliers in routine data. "
    "Chose 'Routine data (Raw)' to use raw routine data.",
    multiple=False,
    choices=[
        "Routine data (Raw)",
        "Mean (Classic)",
        "Median (Classic)",
        "IQR (Classic)",
        "Trend (PATH)",
        "MG Partial (MagicGlasses2)",
        "MG Complete (MagicGlasses2)",
    ],
    type=str,
    default="Routine data (Raw)",
    required=True,
)
@parameter(
    "use_removed_outliers",
    name="Use routine data with outliers removed (else: uses imputed)",
    help="Enable this option to use routine data after outliers have been removed, "
    "based on the outlier detection method you selected above. "
    " If you leave this off, the pipeline will instead use either:"
    " A) the imputed routine data (where outlier values have been replaced), or"
    " B) the raw routine data, if you chose 'Routine data (Raw)' as your outlier processing method.",
    type=bool,
    default=False,
    required=False,
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
    outliers_method: list, use_removed_outliers: bool, run_report_only: bool, pull_scripts: bool
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
            routine_file = resolve_routine_filename(outliers_method, use_removed_outliers)
            routine_file = f"{country_code}{routine_file}"
            if outliers_method == "Routine data (Raw)":
                ds_outliers_id = snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_DATASET_FORMATTED"]
            else:
                ds_outliers_id = snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_OUTLIERS_IMPUTATION"]

            # Check the file exists in the dataset
            if not dataset_file_exists(ds_id=ds_outliers_id, filename=routine_file):
                current_run.log_warning(
                    f"Routine file {routine_file} not found in the dataset {ds_outliers_id}, "
                    "perhaps the outliers imputation pipeline has not been run yet. "
                    "Processing cannot continue."
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


def resolve_routine_filename(outliers_method: str, is_removed: bool) -> str:
    """Returns the routine data filename based on the selected outliers method.

    Parameters
    ----------
    outliers_method : str
        The method used for outlier removal.
    is_removed : bool
        Whether to return the filename for removed outliers or imputed outliers.

    Returns
    -------
    str
        The filename corresponding to the selected outliers method.

    Raises
    ------
    ValueError
        If the outliers method is unknown.
    """
    if outliers_method == "Routine data (Raw)":
        return "_routine.parquet"

    method_suffix_map = {
        "Mean (Classic)": "mean",
        "Median (Classic)": "median",
        "IQR (Classic)": "iqr",
        "Trend (PATH)": "trend",
        "MG Partial (MagicGlasses2)": "mg-partial",
        "MG Complete (MagicGlasses2)": "mg-complete",
    }

    try:
        suffix = method_suffix_map[outliers_method]
    except KeyError as err:
        raise ValueError(f"Unknown outliers method: {outliers_method}") from err

    return f"_routine_outliers-{suffix}{'_removed' if is_removed else '_imputed'}.parquet"


if __name__ == "__main__":
    snt_dhis2_reporting_rate_dataset()
