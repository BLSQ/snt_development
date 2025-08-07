from pathlib import Path
from openhexa.sdk import current_run, pipeline, workspace, parameter
from snt_lib.snt_pipeline_utils import (
    pull_scripts_from_repository,
    add_files_to_dataset,
    load_configuration_snt,
    run_notebook,
    run_report_notebook,
    validate_config,
)
# Pipeline for calculating DHIS2 reporting rates with configurable parameters.
@pipeline("snt_dhis2_reporting_rate")
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
@parameter(
    "dataelement_method_numerator",
    name="Reporting Rate method 'Data Element': choice of Numerator",
    help="Which indicator(s) to use to count the number of reporting facilities.",
    type=str,
    choices=["CONF", "CONF|SUSP|TEST"],
    required=True
)
@parameter(
    "dataelement_method_denominator",
    name="Reporting Rate method 'Data Element': choice of Denominator",
    help="How to calculate the total nr of facilities expected to report.",
    type=str,
    choices=["DHIS2_EXPECTED_REPORTS", "ACTIVE_FACILITIES"],
    required=True
)
def orchestration_function(run_report_only: bool, 
                           pull_scripts: bool, 
                           dataelement_method_numerator: str, 
                           dataelement_method_denominator: str):
    """Orchestration function. Calls other functions within the pipeline."""
    current_run.log_debug("🚀 STARTING DEBUG OUTPUT")

    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_dhis2_reporting_rate",
            report_scripts=["snt_dhis2_reporting_rate_report.ipynb"],
            code_scripts=["snt_dhis2_reporting_rate.ipynb"],
        )

    try:
        # Set paths
        root_path = Path(workspace.files_path)
        pipeline_path = root_path / "pipelines" / "snt_dhis2_reporting_rate"
        data_path = root_path / "data" / "dhis2" / "reporting_rate"

        # Load configuration
        snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
        validate_config(snt_config)
        country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]

        if not run_report_only:
            run_notebook(
                nb_path=pipeline_path / "code" / "snt_dhis2_reporting_rate.ipynb",
                out_nb_path=pipeline_path / "papermill_outputs",
                parameters={
                    "SNT_ROOT_PATH": root_path.as_posix(),
                    "DATAELEMENT_METHOD_NUMERATOR": dataelement_method_numerator,
                    "DATAELEMENT_METHOD_DENOMINATOR": dataelement_method_denominator
                },
            )            

            add_files_to_dataset(
                dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_REPORTING_RATE"],
                country_code=country_code,
                file_paths=[
                    *[p for p in (data_path.glob(f"{country_code}_reporting_rate_*.parquet"))],
                    *[p for p in (data_path.glob(f"{country_code}_reporting_rate_*.csv"))]
                ],
            )
            
        else:
            current_run.log_info("Skipping calculations, running only the reporting.")

        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_dhis2_reporting_rate_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            nb_parameters=None,
        )

        current_run.log_info("Pipeline completed successfully!")

    except Exception as e:
        current_run.log_error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    orchestration_function()
