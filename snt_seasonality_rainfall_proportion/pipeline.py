from pathlib import Path
from openhexa.sdk import current_run, pipeline, workspace, parameter
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    load_configuration_snt,
    validate_config,
    run_report_notebook,
    run_notebook,
    pull_scripts_from_repository,
)


# 4 is default value but we are open to change it to any number of months
@pipeline("snt_seasonality_rainfall_proportion")
@parameter(
    "top_months_n",
    name="Number of top rainfall months",
    help="Number of months used to compute the rainfall proportion.",
    type=int,
    default=4,
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
def snt_seasonality_rainfall_proportion(
    top_months_n: int,
    run_report_only: bool,
    pull_scripts: bool,
):
    """Estimate rainfall seasonality based on proportion of top months.

    For each ADM2 and year:
      - compute monthly rainfall proportions
      - select the top N months
      - sum the proportions of the top months
    Summarize the results across years per ADM2.
    """
    root_path = Path(workspace.files_path)
    pipeline_path = root_path / "pipelines" / "snt_seasonality_rainfall_proportion"
    data_path = root_path / "data" / "seasonality_rainfall_proportion"
    pipeline_path.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)

    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_seasonality_rainfall_proportion",
            report_scripts=["snt_seasonality_rainfall_proportion_report.ipynb"],
            code_scripts=["snt_seasonality_rainfall_proportion.ipynb"],
        )

    if not run_report_only:
        try:
            snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
            validate_config(snt_config)
            country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]

            validate_parameters({"top_months_n": top_months_n})

            run_notebook(
                nb_path=pipeline_path / "code" / "snt_seasonality_rainfall_proportion.ipynb",
                out_nb_path=pipeline_path / "papermill_outputs",
                parameters={
                    "top_months_n": top_months_n,
                },
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            )

            add_files_to_dataset(
                dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["SNT_SEASONALITY_RAINFALL_PROPORTION"],
                country_code=country_code,
                file_paths=[
                    data_path / f"{country_code}_rainfall_proportion.parquet",
                    data_path / f"{country_code}_rainfall_proportion.csv",
                ],
            )

        except Exception as e:
            current_run.log_error(f"Pipeline failed: {e!s}")
            raise

    else:
        current_run.log_info("Skipping calculations, running only the reporting.")

    run_report_notebook(
        nb_file=pipeline_path / "reporting" / "snt_seasonality_rainfall_proportion_report.ipynb",
        nb_output_path=pipeline_path / "reporting" / "outputs",
        error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
    )


def validate_parameters(parameters: dict):
    """Validate numeric parameters."""
    if parameters["top_months_n"] <= 0:
        raise ValueError("Please supply a positive number of months.")
    if not isinstance(parameters["top_months_n"], int):
        raise TypeError("Please supply an integer number of months.")
    if parameters["top_months_n"] > 12:
        raise ValueError("Please supply at most 12 months.")


if __name__ == "__main__":
    snt_seasonality_rainfall_proportion()
