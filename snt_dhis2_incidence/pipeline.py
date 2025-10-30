from pathlib import Path
from openhexa.sdk import current_run, parameter, pipeline, workspace
from snt_lib.snt_pipeline_utils import (
    pull_scripts_from_repository,
    run_notebook,
    add_files_to_dataset,
    load_configuration_snt,
    run_report_notebook,
    validate_config,
    dataset_file_exists,
)


@pipeline("snt_dhis2_incidence")
@parameter(
    "n1_method",
    name="Method for N1 calculations",
    help="Calculate N1 using `PRES` or `SUSP-TEST`",
    choices=["PRES", "SUSP-TEST"],
    type=str,
    required=True,
)
@parameter(
    "routine_data_choice",
    name="Routine data to use",
    help="Which routine data to use for the analysis. Options: 'raw' data is simply formatted and aligned;"
    "'raw_without_outliers' is the raw data after outliers removed (based on `outlier_detection_method`);"
    " 'imputed' contains imputed values after outliers removal",
    choices=["raw", "raw_without_outliers", "imputed"],
    type=str,
    required=True,
)
@parameter(
    "outlier_detection_method",
    name="Outlier detection method",
    help="Method to use for outlier detection in the routine data",
    choices=[
        "Mean (Classic)",
        "Median (Classic)",
        "IQR (Classic)",
        "Trend (PATH)",
        "MG Partial",
        "MG Complete",
    ],
    type=str,
    required=True,
)
@parameter(
    "reporting_rate_method",
    name="Reporting rate to use",
    help="Which reporting rate method to use for the analysis. Note: Reporting rate was calculated"
    " previously, and is simply imported here",
    choices=["dataset", "dataelement"],
    type=str,
    required=True,
)
@parameter(
    "use_csb_data",
    name="Use care seeking data (DHS)",
    help="If True, the pipeline will use care seeking data (DHS) for the analysis,"
    " and calculate incidence adjusted for care seeking",
    type=bool,
    default=False,
    required=True,
)
@parameter(
    "use_adjusted_population",
    name="Use adjusted population",
    help="If enabled, use adjusted population data for incidence calculations",
    type=bool,
    default=False,
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
    name="Pull scripts",
    help="Pull the latest scripts from the repository",
    type=bool,
    default=False,
    required=False,
)
def snt_dhis2_incidence(
    n1_method: str,
    routine_data_choice: str,
    outlier_detection_method: str,
    reporting_rate_method: str,
    use_csb_data: bool,
    use_adjusted_population: bool,
    run_report_only: bool,
    pull_scripts: bool,
):
    """Pipeline entry point for running the SNT DHIS2 incidence notebook with specified parameters.

    Parameters
    ----------
    n1_method : str
        Method for N1 calculations (`PRES` or `SUSP-TEST`).
    routine_data_choice : str
        Which routine data to use for the analysis (`raw`, `raw_without_outliers`, or `imputed`).
    outlier_detection_method : str
        Method to use for outlier detection in the routine data.
    reporting_rate_method : str
        Reporting Rate method to use for the analysis (`dataset`, `dataelement`).
    use_csb_data : bool
        If True, use Care Seeking Data (DHS) for the analysis and
        calculate incidence adjusted for care seeking.
    use_adjusted_population : bool
        If True, use adjusted population data for incidence calculations.
    run_report_only : bool
        If True, only the reporting notebook will be executed, skipping the main analysis.
    pull_scripts : bool
        If True, pull the latest scripts from the repository.
    """
    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_dhis2_incidence",
            report_scripts=["snt_dhis2_incidence_report.ipynb"],
            code_scripts=["snt_dhis2_incidence.ipynb"],
        )

    try:
        current_run.log_info("Starting SNT DHIS2 Incidence Pipeline...")
        root_path = Path(workspace.files_path)
        pipeline_path = root_path / "pipelines" / "snt_dhis2_incidence"
        data_path = root_path / "data" / "dhis2" / "incidence"
        data_path.mkdir(parents=True, exist_ok=True)

        # Load configuration
        snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
        validate_config(snt_config)
        country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]

        if not run_report_only:
            routine_file = resolve_routine_filename(outlier_detection_method, routine_data_choice)
            routine_file = f"{country_code}{routine_file}"
            if routine_data_choice == "raw":
                ds_outliers_id = snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_DATASET_FORMATTED"]
            else:
                ds_outliers_id = snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_OUTLIERS_IMPUTATION"]

            # Check the file exists in the dataset
            if not dataset_file_exists(ds_id=ds_outliers_id, filename=routine_file):
                current_run.log_error(
                    f"Routine file {routine_file} not found in the dataset {ds_outliers_id}, "
                    "perhaps the outliers imputation pipeline has not been run yet. "
                    "Processing cannot continue."
                )
                return

            run_notebook(
                nb_path=pipeline_path / "code" / "snt_dhis2_incidence.ipynb",
                out_nb_path=pipeline_path / "papermill_outputs",
                parameters={
                    "N1_METHOD": n1_method,
                    "ROUTINE_DATA_CHOICE": routine_file,
                    "REPORTING_RATE_METHOD": reporting_rate_method,
                    "USE_CSB_DATA": use_csb_data,
                    "USE_ADJUSTED_POPULATION": use_adjusted_population,
                },
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            )

            add_files_to_dataset(
                dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_INCIDENCE"],
                country_code=country_code,
                file_paths=[
                    *[
                        p
                        for p in (
                            data_path.glob(
                                f"{country_code}_incidence_year_routine-data-*_rr-method-*.parquet"
                            )
                        )
                    ],
                    *[
                        p
                        for p in (
                            data_path.glob(f"{country_code}_incidence_year_routine-data-*_rr-method-*.csv")
                        )
                    ],
                ],
            )

        else:
            current_run.log_info("Skipping incidence calculations, running only the reporting.")

        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_dhis2_incidence_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
        )

        current_run.log_info("Pipeline finished!")

    except Exception as e:
        current_run.log_error(f"Notebook execution failed: {e}")
        raise


def resolve_routine_filename(outliers_method: str, routine_choice: bool) -> str:
    """Returns the routine data filename based on the selected outliers method.

    Parameters
    ----------
    outliers_method : str
        The method used for outlier removal.
    routine_choice : str
        The routine data choice.

    Returns
    -------
    str
        The filename corresponding to the selected outliers method.

    Raises
    ------
    ValueError
        If the outliers method is unknown.
    """
    if routine_choice == "raw":
        return "_routine.parquet"

    is_removed = False
    if routine_choice == "raw_without_outliers":
        is_removed = True

    method_suffix_map = {
        "Mean (Classic)": "mean",
        "Median (Classic)": "median",
        "IQR (Classic)": "iqr",
        "Trend (PATH)": "trend",
    }

    try:
        suffix = method_suffix_map[outliers_method]
    except KeyError as err:
        raise ValueError(f"Unknown outliers method: {outliers_method}") from err

    return f"_routine_outliers-{suffix}{'_removed' if is_removed else '_imputed'}.parquet"


if __name__ == "__main__":
    snt_dhis2_incidence()
