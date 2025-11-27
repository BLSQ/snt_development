
import os
from pathlib import Path
import papermill as pm
from datetime import datetime
import subprocess  # for the html output
from subprocess import CalledProcessError
from openhexa.sdk import current_run, pipeline, workspace, parameter
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    load_configuration_snt,
    validate_config,
)

@pipeline(name="snt_seasonality_rainfall")
@parameter(
    "get_minimum_month_block_size",
    name="Minimum number of months per block",
    help="",
    type=int,
    choices=[3,4,5],
    default=4,
    required=True,
)
@parameter(
    "get_maximum_month_block_size",
    name="Maximum number of months per block",
    help="",
    type=int,
    default=4,
    choices=[3,4,5],
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

def snt_rain_seas(
    get_minimum_month_block_size,
    get_maximum_month_block_size,
    get_threshold_for_seasonality,
    get_threshold_proportion_seasonal_years
    ):
    """
    Retriev rainfall data by ADM2 level from ERA5 dataset
    Compute whether or not the admin unit qualifies as seasonal from a case and rainfall perspective
    Compute the minimal duration of the seasonal block (in months) for each district
    Plot maps and save the output table to dataset
    """

    # paths
    root_path = Path(workspace.files_path)
    pipeline_path = Path(root_path, "pipelines", "snt_seasonality_rainfall")
    code_path = Path(pipeline_path, "code")
    pm_output_path = Path(pipeline_path, "papermill_outputs")
    reports_path = Path(pipeline_path, "reporting")
    report_outputs_path = Path(reports_path, "outputs")
    data_path = root_path / "data" / "seasonality_rainfall"

    computation_nb="snt_rainfall_seasonality_computation"
    reporting_nb="snt_rainfall_seasonality_report"

    try:
        # config input
        snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
        validate_config(snt_config)
        country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]

        input_params = {
        "minimum_month_block_size": get_minimum_month_block_size,
        "maximum_month_block_size": get_maximum_month_block_size,
        "threshold_for_seasonality": get_threshold_for_seasonality,
        "threshold_proportion_seasonal_years": get_threshold_proportion_seasonal_years
        }

        validate_parameters(input_params)

    except Exception as e:
        current_run.log_error(f"Parameter validation failed: {e}")
        raise

    files_to_ds = [] # what will go to the dataset

    run_computation_nb(
        nb_name=computation_nb,
        nb_path=code_path,
        pm_path=pm_output_path,
        params=input_params
    )
    
    files_to_ds.append(data_path / f"{country_code}_rainfall_seasonality.parquet")
    files_to_ds.append(data_path / f"{country_code}_rainfall_seasonality.csv")

    try:

        add_files_to_dataset(
            dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["SNT_SEASONALITY_RAINFALL"],
            country_code=country_code,
            file_paths=files_to_ds,
        )
    
    except Exception as e:
        current_run.log_error(f"Dataset upload failed: {e}")
        raise

    run_report_nb(
        nb_name=reporting_nb,
        nb_path=reports_path,
        nb_out_path=report_outputs_path,
        params=input_params
    )

def validate_parameters(parameters: dict):
    """seasonality param validation

    Parameters
    parameters: dictionary of parameter names and their values

    Raises
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


def run_computation_nb(nb_name, nb_path, pm_path, params):
    in_path = Path(nb_path, f"{nb_name}.ipynb")
    out_path = Path(pm_path, f"{nb_name}-output.ipynb")
    current_run.log_info(f"Executing {in_path}.")
    try:
        pm.execute_notebook(
            input_path=in_path,
            output_path=out_path,
            parameters=params
            )
    except Exception as e:
        raise Exception(f"Error executing the notebook {type(e)}: {e}") from e

def generate_html_report(output_notebook_path: Path, output_dir: Path) -> None:
    """Generate an HTML report from a Jupyter notebook.

    Parameters
    ----------
    output_notebook_path : Path
        Path to the output notebook file.

    Raises
    ------
    RuntimeError
        If an error occurs during the conversion process.
    """
    if not output_notebook_path.is_file() or output_notebook_path.suffix.lower() != ".ipynb":
        raise RuntimeError(f"Invalid notebook path: {output_notebook_path}")

    if not output_dir.is_dir():
        raise RuntimeError(f"Output directory does not exist: {output_dir}")

    output_filename = output_notebook_path.with_suffix(".html").name
    report_path = output_dir / output_filename

    current_run.log_info(f"Generating HTML report {report_path}")
    try:
        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to=html",
                f"--output={output_filename}",
                f"--output-dir={str(output_dir)}",
                str(output_notebook_path),
            ],
            check=True,
        )
    except CalledProcessError as e:
        raise RuntimeError(f"Error converting notebook to HTML (exit {e.returncode}): {e}") from e

    current_run.add_file_output(str(report_path))

def run_report_nb(nb_name, nb_path, nb_out_path, params):
    in_path = Path(nb_path, f"{nb_name}.ipynb")
    out_path = Path(nb_out_path, f"{nb_name}-output.ipynb")
    current_run.log_info(f"Executing {in_path}.")
    try:
        pm.execute_notebook(
            input_path=in_path,
            output_path=out_path,
            parameters=params
            )
        generate_html_report(
            output_notebook_path=out_path,
            output_dir=nb_out_path
            )
    except Exception as e:
        raise Exception(f"Error executing the notebook {type(e)}: {e}") from e


if __name__ == "__main__":
    snt_rain_seas()