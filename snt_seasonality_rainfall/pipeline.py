from pathlib import Path
from openhexa.sdk import current_run, pipeline, workspace, parameter
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    load_configuration_snt,
    validate_config,
    run_report_notebook,
    run_notebook,
    pull_scripts_from_repository,
    save_pipeline_parameters,
)


@pipeline("snt_seasonality_rainfall")
@parameter(
    "get_minimum_month_block_size",
    name="Minimum number of months per block",
    help="",
    type=int,
    choices=[3, 4, 5],
    default=4,
    required=True,
)
@parameter(
    "get_maximum_month_block_size",
    name="Maximum number of months per block",
    help="",
    type=int,
    default=4,
    choices=[3, 4, 5],
    required=True,
)
@parameter(
    "get_threshold_for_seasonality",
    name="Minimal proportion of cases/rainfall for seasonality",
    help="",
    type=float,
    default=0.6,
    required=True,
)
@parameter(
    "get_threshold_proportion_seasonal_years",
    name="Minimal proportion of seasonal years",
    help="",
    type=float,
    default=0.5,
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
def snt_seasonality_rainfall(
    get_minimum_month_block_size: int,
    get_maximum_month_block_size: int,
    get_threshold_for_seasonality: float,
    get_threshold_proportion_seasonal_years: float,
    run_report_only: bool,
    pull_scripts: bool,
):
    """Retrieve rainfall data by ADM2 level from ERA5 dataset.

    Compute whether or not the admin unit qualifies as seasonal from a rainfall perspective
    Compute the minimal duration of the seasonal block (in months) for each district
    Plot maps and save the output table to dataset
    """
    root_path = Path(workspace.files_path)
    pipeline_path = root_path / "pipelines" / "snt_seasonality_rainfall"
    data_path = root_path / "data" / "seasonality_rainfall"
    pipeline_path.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)

    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_seasonality_rainfall",
            report_scripts=["snt_seasonality_rainfall_report.ipynb"],
            code_scripts=["snt_seasonality_rainfall.ipynb"],
        )

    if not run_report_only:
        try:
            # config input
            snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
            validate_config(snt_config)
            country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]

            input_params = {
                "minimum_month_block_size": get_minimum_month_block_size,
                "maximum_month_block_size": get_maximum_month_block_size,
                "threshold_for_seasonality": get_threshold_for_seasonality,
                "threshold_proportion_seasonal_years": get_threshold_proportion_seasonal_years,
            }
            validate_parameters(input_params)

            run_notebook(
                nb_path=pipeline_path / "code" / "snt_seasonality_rainfall.ipynb",
                out_nb_path=pipeline_path / "papermill_outputs",
                parameters=input_params,
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
                country_code=country_code,
            )

            # Save parameters to JSON file for reporting notebook
            parameters_file = save_pipeline_parameters(
                pipeline_name="snt_seasonality_rainfall",
                parameters=input_params,
                output_path=data_path,
                country_code=country_code,
            )

            add_files_to_dataset(
                dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["SNT_SEASONALITY_RAINFALL"],
                country_code=country_code,
                file_paths=[
                    data_path / f"{country_code}_rainfall_seasonality.parquet",
                    data_path / f"{country_code}_rainfall_seasonality.csv",
                    parameters_file,
                ],
            )

        except Exception as e:
            current_run.log_error(f"Pipeline failed: {e!s}")
            raise

    else:
        current_run.log_info("Skipping calculations, running only the reporting.")
        snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
        country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]

    run_report_notebook(
        nb_file=pipeline_path / "reporting" / "snt_seasonality_rainfall_report.ipynb",
        nb_output_path=pipeline_path / "reporting" / "outputs",
        error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
        country_code=country_code,
    )


def validate_parameters(parameters: dict):
    """Seasonality param validation.

    Args:
        parameters (dict): Dictionary of parameter names and their values.

    Raises:
    ValueError: if param value is negative or a proportion is not between 0 and 1
    ValueError: if smaller value is larger than larger value (min block > max block)
    TypeError: if integer parameter is not an integer
    """
    for current_parameter, current_value in parameters.items():
        if current_value < 0:
            raise ValueError("Please supply only positive values.")
        if current_parameter in ["minimum_periods", "minimum_month_block_size", "maximum_month_block_size"]:
            if not isinstance(current_value, int):
                raise TypeError("Please supply integer values for number of months/periods.")
        else:
            if current_value > 1:
                raise ValueError("Proportions values should be between 0 and 1.")

    if parameters["minimum_month_block_size"] > parameters["maximum_month_block_size"]:
        raise ValueError("The minimum value should not be larger than the maximum one.")


if __name__ == "__main__":
    snt_seasonality_rainfall()
