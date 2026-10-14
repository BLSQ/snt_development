import time
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
    check_outputs_generated,
)


@pipeline("snt_dhis2_reporting_rate_dataelement")
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
    "activity_indicators",
    name="Facility Activity indicators",
    help=(
        "Define which data elements will be used to determine the activity of a facility."
        " A facility is considered 'active' if at least one of these indicators has a non-missing "
        "value greater than zero."
    ),
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
    default="ROUTINE_ACTIVE_FACILITIES",
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
    routine_data_choice: str,
    activity_indicators: list,
    volume_activity_indicators: list,
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

    # Set paths
    root_path = Path(workspace.files_path)
    pipeline_path = root_path / "pipelines" / "snt_dhis2_reporting_rate_dataelement"
    data_path = root_path / "data" / "dhis2" / "reporting_rate"
    data_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load configuration
        snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
        validate_config(snt_config)
        country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]
    except Exception as e:
        current_run.log_error(f"Failed to load or validate configuration: {e}")
        raise

    if not run_report_only:
        # Resolve routine data source based on the user's choice
        dataset_id, routine_file = resolve_routine_source(
            country_code=country_code, routine_data_choice=routine_data_choice, snt_config=snt_config
        )

        # Check the file exists in the dataset
        if not dataset_file_exists(ds_id=dataset_id, filename=routine_file):
            pipeline_target = "formatting" if routine_data_choice == "raw" else "outliers imputation"
            current_run.log_error(
                f"Routine file {routine_file} not found in the dataset {dataset_id}, "
                f"perhaps the {pipeline_target} pipeline has not been run yet. "
                "Processing cannot continue."
            )
            raise FileNotFoundError(f"Routine file {routine_file} not found in the dataset {dataset_id}")

        # Build parameters dict
        nb_parameters = {
            "ROOT_PATH": root_path.as_posix(),
            "ROUTINE_FILE": routine_file,
            "DATASET_ID": dataset_id,
            "DATAELEMENT_METHOD_DENOMINATOR": dataelement_method_denominator,
            "ACTIVITY_INDICATORS": activity_indicators,
            "VOLUME_ACTIVITY_INDICATORS": volume_activity_indicators,
            "USE_WEIGHTED_REPORTING_RATES": use_weighted_reporting_rates,
        }

        try:
            parameters_file = save_pipeline_parameters(
                pipeline_name="snt_dhis2_reporting_rate_dataelement",
                parameters=nb_parameters,
                output_path=data_path,
                country_code=country_code,
            )
        except Exception as e:
            current_run.log_error(f"Failed to save pipeline parameters: {e}")
            raise
        current_run.log_info(f"Saved pipeline parameters to {parameters_file}")

        expected_outputs = [
            data_path / f"{country_code}_reporting_rate_dataelement.parquet",
            data_path / f"{country_code}_reporting_rate_dataelement.csv",
        ]

        run_start_ts = time.time()
        try:
            run_notebook(
                nb_path=pipeline_path / "code" / "snt_dhis2_reporting_rate_dataelement.ipynb",
                out_nb_path=pipeline_path / "papermill_outputs",
                parameters=nb_parameters,
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
                country_code=country_code,
            )
        except Exception as e:
            current_run.log_error(f"Failed to run notebook: {e}")
            raise

        check_outputs_generated(file_paths=expected_outputs, run_start_ts=run_start_ts)

        add_files_to_dataset(
            dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_REPORTING_RATE"],
            country_code=country_code,
            file_paths=[*expected_outputs, parameters_file],
        )

    else:
        current_run.log_info("Skipping calculations, running only the reporting.")

    try:
        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_dhis2_reporting_rate_dataelement_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            country_code=country_code,
        )
    except Exception as e:
        current_run.log_error(f"Failed to run reporting notebook: {e}")
        raise

    current_run.log_info("Pipeline completed successfully!")


def resolve_routine_source(country_code: str, routine_data_choice: str, snt_config: dict) -> tuple[str, str]:
    """Resolve the dataset id and routine filename for a routine data choice.

    Parameters
    ----------
    country_code : str
        Country code used as the routine filename prefix.
    routine_data_choice : str
        The routine data to use: "raw", "imputed" or "outliers_removed".
    snt_config : dict
        The loaded SNT configuration, used to look up the dataset identifier.

    Returns
    -------
    tuple[str, str]
        The dataset id holding the routine file, and the routine filename.

    Raises
    ------
    ValueError
        If ``routine_data_choice`` is not a known choice.
    """
    routine_sources = {
        "raw": ("DHIS2_DATASET_FORMATTED", f"{country_code}_routine.parquet"),
        "imputed": ("DHIS2_OUTLIERS_IMPUTATION", f"{country_code}_routine_outliers_imputed.parquet"),
        "outliers_removed": ("DHIS2_OUTLIERS_IMPUTATION", f"{country_code}_routine_outliers_removed.parquet"),
    }
    if routine_data_choice not in routine_sources:
        raise ValueError(f"Unknown routine data choice: {routine_data_choice}")

    dataset_key, routine_file = routine_sources[routine_data_choice]
    return snt_config["SNT_DATASET_IDENTIFIERS"][dataset_key], routine_file


if __name__ == "__main__":
    snt_dhis2_reporting_rate_dataelement()
