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
)


@pipeline("snt_dhis2_reporting_rate_dataset")
@parameter(
    "outliers_method",
    name="Outlier processing method",
    help="Select the routine data results from this outliers method.",
    multiple=False,
    choices=[
        "Routine data (Raw)",
        "Mean (Classic)",
        "Median (Classic)",
        "IQR (Classic)",
        "Trend (PATH)",
        "MG Partial",
        "MG Complete",
    ],
    type=str,
    default=None,
    required=True,
)
@parameter(
    "use_removed_outliers",
    name="Use routine with outliers removed",
    help="Select this option to use the version of the routine data where outlier have been removed.",
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
    name="Pull scripts",
    help="Pull the latest scripts from the repository",
    type=bool,
    default=False,
    required=False,
)
def snt_dhis2_reporting_rate_dataset(
    outliers_method: list, use_removed_outliers: bool, run_report_only: bool, pull_scripts: bool
):
    """Orchestration function. Calls other functions within the pipeline."""
    current_run.log_debug("🚀 STARTING DEBUG OUTPUT")

    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
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

            run_notebook(
                nb_path=pipeline_path / "code" / "snt_dhis2_reporting_rate_dataset.ipynb",
                out_nb_path=pipeline_path / "papermill_outputs",
                parameters={
                    "SNT_ROOT_PATH": root_path.as_posix(),
                    "ROUTINE_FILE": routine_file,
                },
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            )

            add_files_to_dataset(
                dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_REPORTING_RATE"],
                country_code=country_code,
                file_paths=[
                    *[p for p in (data_path.glob(f"{country_code}_reporting_rate_dataset.parquet"))],
                    *[p for p in (data_path.glob(f"{country_code}_reporting_rate_dataset.csv"))],
                ],
            )

        else:
            current_run.log_info("Skipping calculations, running only the reporting.")

        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_dhis2_reporting_rate_dataset_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            nb_parameters=None,
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
        "MG Partial": "mg-partial",
        "MG Complete": "mg-complete",
    }

    try:
        suffix = method_suffix_map[outliers_method]
    except KeyError as err:
        raise ValueError(f"Unknown outliers method: {outliers_method}") from err

    return f"_routine_outliers-{suffix}{'_removed' if is_removed else '_imputed'}.parquet"


if __name__ == "__main__":
    snt_dhis2_reporting_rate_dataset()
