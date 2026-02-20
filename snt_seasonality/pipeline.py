from datetime import datetime
from pathlib import Path
import papermill as pm
from openhexa.sdk import current_run, parameter, pipeline, workspace
from snt_lib.snt_pipeline_utils import (
    pull_scripts_from_repository,
    add_files_to_dataset,
    load_configuration_snt,
    run_report_notebook,
    validate_config,
    save_pipeline_parameters,
)


@pipeline("snt_seasonality")
@parameter(
    "run_precipitation",
    name="Run precipitation seasonality",
    help="",
    type=bool,
    default=True,
)
@parameter(
    "run_cases",
    name="Run cases seasonality",
    help="",
    type=bool,
    default=True,
)
@parameter(
    "minimum_periods",
    name="Minimum number of periods",
    help="",
    type=int,
    default=48,
    required=True,
)
@parameter(
    "maximum_proportion_missings_overall",
    name="Maximum proportion of missing datapoints overall",
    help="",
    type=float,
    default=0.1,
    required=True,
)
@parameter(
    "maximum_proportion_missings_per_district",
    name="Maximum proportion of missing datapoints per distric",
    help="",
    type=float,
    default=0.2,
    required=True,
)
@parameter(
    "minimum_month_block_size",
    name="Minimum month block size",
    help="",
    type=int,
    default=3,
    required=True,
)
@parameter(
    "maximum_month_block_size",
    name="Maximum month block size",
    help="",
    type=int,
    default=5,
    required=True,
)
@parameter(
    "threshold_for_seasonality",
    name="Threshold for seasonality",
    help="",
    type=float,
    default=0.6,
    required=True,
)
@parameter(
    "threshold_proportion_seasonal_years",
    name="Threshold proportion seasonal years",
    help="",
    type=float,
    default=0.5,
    required=True,
)
@parameter(
    "run_report_only",
    name="Run reporting only",
    help="This will only execute the reporting notebook",
    type=bool,
    default=False,
)
@parameter(
    "pull_scripts",
    name="Pull Scripts",
    help="Pull the latest scripts from the repository",
    type=bool,
    default=False,
    required=False,
)
def snt_seasonality(
    run_precipitation: bool,
    run_cases: bool,
    minimum_periods: int,
    maximum_proportion_missings_overall: float,
    maximum_proportion_missings_per_district: float,
    minimum_month_block_size: int,
    maximum_month_block_size: int,
    threshold_for_seasonality: float,
    threshold_proportion_seasonal_years: float,
    run_report_only: bool,
    pull_scripts: bool,
):
    """Computes whether or not the admin unit qualifies as seasonal from a case and precipitation perspective.

    Duration of the seasonal block is in months.

    """
    # paths
    root_path = Path(workspace.files_path)
    pipeline_path = root_path / "pipelines" / "snt_seasonality"
    data_path = root_path / "data" / "seasonality"

    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_seasonality",
            report_scripts=["snt_seasonality_report.ipynb"],
            code_scripts=["snt_seasonality.ipynb"],
        )

    try:
        # Load configuration
        snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
        validate_config(snt_config)
        country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]

        params = {
            "minimum_periods": minimum_periods,
            "maximum_proportion_missings_overall": maximum_proportion_missings_overall,
            "maximum_proportion_missings_per_district": maximum_proportion_missings_per_district,
            "minimum_month_block_size": minimum_month_block_size,
            "maximum_month_block_size": maximum_month_block_size,
            "threshold_for_seasonality": threshold_for_seasonality,
            "threshold_proportion_seasonal_years": threshold_proportion_seasonal_years,
            "run_precipitation": run_precipitation,
            "run_cases": run_cases,
        }
        validate_parameters(params)

        if not run_report_only:
            parameters_file = save_pipeline_parameters(
                pipeline_name="snt_seasonality",
                parameters=params,
                output_path=data_path,
                country_code=country_code,
            )
            files_to_ds = [parameters_file]
            error_messages = ["ERROR 1", "ERROR 2", "ERROR 3"]
            seasonality_nb = pipeline_path / "code" / "snt_seasonality.ipynb"

            # Precipitation seasonality
            if run_precipitation:
                current_run.log_info(f"Running precipitation analysis with notebook : {seasonality_nb}")
                try:
                    nb_params = {**params, "type_of_seasonality": "precipitation"}
                    run_notebook_for_type(
                        nb_path=seasonality_nb,
                        seasonality_type="precipitation",
                        out_nb_path=pipeline_path / "papermill_outputs",
                        parameters=nb_params,
                    )
                    files_to_ds.append(data_path / f"{country_code}_precipitation_seasonality.parquet")
                    files_to_ds.append(data_path / f"{country_code}_precipitation_seasonality.csv")
                    files_to_ds.append(data_path / f"{country_code}_precipitation_imputed.parquet")
                    files_to_ds.append(data_path / f"{country_code}_precipitation_imputed.csv")
                except Exception as e:
                    if any(msg in str(e) for msg in error_messages):
                        current_run.log_warning(
                            "The precipitation analysis cannot be performed. Process stopped."
                        )
                    else:
                        raise Exception(
                            f"Unexpected error occurred during the precipitation seasonality execution: {e}"
                        ) from e

            # Cases seasonality
            if run_cases:
                current_run.log_info(f"Running cases analysis with notebook : {seasonality_nb}")
                try:
                    nb_params = {**params, "type_of_seasonality": "cases"}
                    run_notebook_for_type(
                        nb_path=seasonality_nb,
                        seasonality_type="cases",
                        out_nb_path=pipeline_path / "papermill_outputs",
                        parameters=nb_params,
                    )
                    files_to_ds.append(data_path / f"{country_code}_cases_seasonality.parquet")
                    files_to_ds.append(data_path / f"{country_code}_cases_seasonality.csv")
                    files_to_ds.append(data_path / f"{country_code}_cases_imputed.parquet")
                    files_to_ds.append(data_path / f"{country_code}_cases_imputed.csv")
                except Exception as e:
                    if any(msg in str(e) for msg in error_messages):
                        current_run.log_warning(
                            "The cases seasonality analysis cannot be performed. Process stopped."
                        )
                    else:
                        raise Exception(
                            f"Unexpected error occurred during the cases seasonality execution: {e}."
                        ) from e

            add_files_to_dataset(
                dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["SNT_SEASONALITY"],
                country_code=country_code,
                file_paths=files_to_ds,
            )
        else:
            current_run.log_info("Skipping processing, running only the reporting.")

        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_seasonality_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            country_code=country_code,
        )

        current_run.log_info("Pipeline completed successfully!")

    except Exception as e:
        current_run.log_error(f"Pipeline failed: {e}")
        raise


def validate_parameters(parameters: dict):
    """Validate the pipeline parameters for correct types and value ranges.

    Parameters
    ----------
    parameters : dict
        Dictionary of parameter names and their values.

    Raises
    ------
    ValueError
        If a parameter value is negative or a proportion is not between 0 and 1.
    TypeError
        If a parameter expected to be an integer is not an integer.
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


def run_notebook_for_type(
    nb_path: Path, seasonality_type: str, out_nb_path: Path, parameters: dict, kernel_name: str = "ir"
):
    """Execute a Jupyter notebook using Papermill.

    Parameters
    ----------
    nb_name : str
        The name of the notebook to execute (without the .ipynb extension).
    nb_path : Path
        The path to the directory containing the notebook.
    seasonality_type : str
        Type of analysis to be added in the output notebook name.
    out_nb_path : Path
        The path to the directory where the output notebook will be saved.
    parameters : dict
        A dictionary of parameters to pass to the notebook.
    kernel_name : str, optional
        The name of the kernel to use for execution (default is "ir" for R, python3 for Python).
    """
    current_run.log_info(f"Executing notebook: {nb_path}")
    file_stem = nb_path.stem
    extension = nb_path.suffix
    execution_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_nb_full_path = out_nb_path / f"{file_stem}_{seasonality_type}_OUTPUT_{execution_timestamp}{extension}"
    out_nb_path.mkdir(parents=True, exist_ok=True)

    try:
        pm.execute_notebook(
            input_path=nb_path,
            output_path=out_nb_full_path,
            parameters=parameters,
            kernel_name=kernel_name,
            request_save_on_cell_execute=False,
        )
    except Exception as e:
        raise Exception(f"Error executing the notebook {type(e)}: {e}") from e


if __name__ == "__main__":
    snt_seasonality()
