from pathlib import Path
from openhexa.sdk import current_run, pipeline, workspace, parameter
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    load_configuration_snt,
    run_notebook,
    run_report_notebook,
    validate_config,
)


@pipeline("snt_dhis2_reporting_rate")
@parameter(
    "run_report_only",
    name="Run reporting only",
    help="This will only execute the reporting notebook",
    type=bool,
    default=False,
    required=False,
)
def dhis2_reporting_rate(run_report_only: bool = False):
    """Pipeline for calculating DHIS2 reporting rates with configurable parameters."""
    current_run.log_debug("🚀 STARTING DEBUG OUTPUT")

    try:
        # Set paths
        root_path = Path(workspace.files_path)
        pipeline_path = root_path / "pipelines" / "snt_dhis2_reporting_rate"
        data_path = root_path / "data" / "dhis2_reporting_rate"

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
                },
            )

            add_files_to_dataset(
                dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_REPORTING_RATE"],
                country_code=country_code,
                file_paths=[
                    data_path / f"{country_code}_reporting_rate_any_month.parquet",
                    data_path / f"{country_code}_reporting_rate_any_month.csv",
                    data_path / f"{country_code}_reporting_rate_conf_month.parquet",
                    data_path / f"{country_code}_reporting_rate_conf_month.csv",
                    data_path / f"{country_code}_reporting_rate_dhis2_month.parquet",
                    data_path / f"{country_code}_reporting_rate_dhis2_month.csv",
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
    dhis2_reporting_rate()
