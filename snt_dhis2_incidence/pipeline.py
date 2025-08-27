from pathlib import Path
from openhexa.sdk import current_run, parameter, pipeline, workspace
from snt_lib.snt_pipeline_utils import (
    pull_scripts_from_repository,
    run_notebook,
    add_files_to_dataset,
    load_configuration_snt,
    run_report_notebook,
    validate_config,
)


@pipeline("snt_dhis2_incidence")
@parameter(
    "n1_method",
    name="Method for N1 calculations",
    help="Calculate N1 using `PRES` or `SUSP-TEST`?",
    choices=["PRES", "SUSP-TEST"],
    type=str,
    required=True,
)
@parameter(
    "routine_data_choice",
    name="Routine data to use",
    help="Which routine data to use for the analysis. Options: 'raw' data is simply formatted and aligned;"
    "'raw_without_outliers' is the raw data after outliers removed (based on `outlier_detection_method`);"
    " 'imputed' contains imputed values after outliers removal. ",
    choices=["raw", "raw_without_outliers", "imputed"],
    type=str,
    required=True,
)
@parameter(
    "outlier_detection_method",
    name="Outlier detection method",
    help="Method to use for outlier detection in the routine data.",
    choices=["median3mad", "mean3sd", "iqr1-5", "magic_glasses_partial", "magic_glasses_complete"],
    type=str,
    required=True,
)
@parameter(
    "reporting_rate_method",
    name="Reporting rate to use",
    help="Which Reporting rate method to use for the analysis. Note: Reporting rate was calculated"
    " previously, and is simply imported here.",
    choices=["dhis2", "conf", "any"],
    type=str,
    required=True,
)
@parameter(
    "use_csb_data",
    name="Use Care Seeking Data (DHS)?",
    help="If True, the pipeline will use Care Seeking Data (DHS) for the analysis,"
    " and calculate incidence adjusted for care seeking.",
    type=bool,
    default=False,
    required=True,
)
@parameter(
    "adjust_population",
    name="Adjust population",
    help="If enabled, the DHIS2 population data will be adjusted using WorldPop UN adjusted estimates.",
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
    name="Pull Scripts",
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
    adjust_population: bool,
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
        Reporting Rate method to use for the analysis (`dhis2`, `conf`, or `any`).
    use_csb_data : bool
        If True, use Care Seeking Data (DHS) for the analysis and
        calculate incidence adjusted for care seeking.
    adjust_population : bool
        If True, adjust the population data using WorldPop UN adjusted estimates.
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
            input_notebook_path = pipeline_path / "code" / "snt_dhis2_incidence.ipynb"

            run_notebook(
                nb_path=input_notebook_path,
                out_nb_path=pipeline_path / "papermill_outputs",
                parameters={
                    "N1_METHOD": n1_method,
                    "ROUTINE_DATA_CHOICE": routine_data_choice,
                    "OUTLIER_DETECTION_METHOD": outlier_detection_method,
                    "REPORTING_RATE_METHOD": reporting_rate_method,
                    "USE_CSB_DATA": use_csb_data,
                    "ADJUST_POPULATION": adjust_population,
                    "ROOT_PATH": root_path.as_posix(),
                },
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            )

            add_files_to_dataset(
                dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_INCIDENCE"],
                country_code=country_code,
                file_paths=[
                    *[
                        str(p)
                        for p in (
                            data_path.glob(
                                f"{country_code}_incidence_year_routine-data-*_rr-method-*.parquet"
                            )
                        )
                    ],
                    *[
                        str(p)
                        for p in (
                            data_path.glob(f"{country_code}_incidence_year_routine-data-*_rr-method-*.csv")
                        )
                    ],
                    *[
                        str(p)
                        for p in (
                            data_path.glob(
                                f"{country_code}_incidence_mean-*_routine-data-*_rr-method-*.parquet"
                            )
                        )
                    ],
                    *[
                        str(p)
                        for p in (
                            data_path.glob(f"{country_code}_incidence_mean-*_routine-data-*_rr-method-*.csv")
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


if __name__ == "__main__":
    snt_dhis2_incidence()
