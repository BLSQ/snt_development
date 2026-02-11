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


@pipeline("snt_dhis2_reporting_rate_dataelement")
@parameter(
    "outliers_method",
    name="Outliers detection method",
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
    default=None,
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
    "activity_indicators",
    name="Facility Activity indicators",
    help="Define which data elements will be used to determine the activity of a facility."
    " A facility is considered 'active' if at least one of these indicators has a non-missing value greater than zero.",
    multiple=True,
    choices=["CONF", "SUSP", "TEST", "PRES"],
    type=str,
    default=["CONF", "PRES"],
    required=True,
)
@parameter(
    "volume_activity_indicators",
    name="Volume activity indicators",
    help="Define which data elements will be used to determine the volume of activity at a facility."
    " Volume of activity is used to calculate WEIGHTED reporting rates.",
    multiple=True,
    choices=["CONF", "SUSP", "TEST", "PRES"],
    type=str,
    default=["CONF", "PRES"],
    required=True,
)
@parameter(
    "dataelement_method_denominator",
    name="Denominator method",
    help="How to calculate the total nr of facilities expected to report.",
    type=str,
    choices=["ROUTINE_ACTIVE_FACILITIES", "PYRAMID_OPEN_FACILITIES"],
    required=True,
)
@parameter(
    "use_weighted_reporting_rates",
    name="Use weighted reporting rates",
    help="Weighted reporting rates are calculated using the volume of activity. "
    "If TRUE, these values will populate the REPORTING_RATE column of the exported data. "
    "If FALSE, unweighted reporting rates will be used instead.",
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
    name="Pull notebooks from repository",
    help="Pull the latest notebooks from the GitHub repository. "
    "Note: this will overwrite any local changes to the notebooks!",
    type=bool,
    default=False,
    required=False,
)
def snt_dhis2_reporting_rate_dataelement(
    outliers_method: str,
    use_removed_outliers: bool,
    activity_indicators: str,
    volume_activity_indicators: str,
    dataelement_method_denominator: str,
    use_weighted_reporting_rates: bool,
    run_report_only: bool,
    pull_scripts: bool,
):
    """Orchestration function. Calls other functions within the pipeline."""
    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_dhis2_reporting_rate_dataelement",
            report_scripts=["snt_dhis2_reporting_rate_dataelement_report.ipynb"],
            code_scripts=["snt_dhis2_reporting_rate_dataelement.ipynb"],
        )

    try:
        # Set paths
        root_path = Path(workspace.files_path)
        pipeline_path = root_path / "pipelines" / "snt_dhis2_reporting_rate_dataelement"
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
                "DATAELEMENT_METHOD_DENOMINATOR": dataelement_method_denominator,
                "ACTIVITY_INDICATORS": activity_indicators,
                "VOLUME_ACTIVITY_INDICATORS": volume_activity_indicators,
                "USE_WEIGHTED_REPORTING_RATES": use_weighted_reporting_rates,
            }

            params_file = save_pipeline_parameters(
                pipeline_name="snt_dhis2_reporting_rate_dataelement",
                parameters=nb_parameters,
                output_path=data_path,
                country_code=country_code,
            )
            current_run.log_info(f"Saved pipeline parameters to {params_file}")

            run_notebook(
                nb_path=pipeline_path / "code" / "snt_dhis2_reporting_rate_dataelement.ipynb",
                out_nb_path=pipeline_path / "papermill_outputs",
                parameters=nb_parameters,
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            )

            add_files_to_dataset(
                dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_REPORTING_RATE"],
                country_code=country_code,
                file_paths=[
                    *[p for p in (data_path.glob(f"{country_code}_reporting_rate_dataelement.parquet"))],
                    *[p for p in (data_path.glob(f"{country_code}_reporting_rate_dataelement.csv"))],
                    params_file,
                ],
            )

        else:
            current_run.log_info("Skipping calculations, running only the reporting.")

        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_dhis2_reporting_rate_dataelement_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
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
    snt_dhis2_reporting_rate_dataelement()
