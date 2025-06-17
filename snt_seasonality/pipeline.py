import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from subprocess import CalledProcessError

import geopandas as gpd
import pandas as pd
import papermill as pm
from nbclient.exceptions import CellTimeoutError
from openhexa.sdk import current_run, parameter, pipeline, workspace
from openhexa.sdk.datasets.dataset import DatasetVersion


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
):
    """Computes whether or not the admin unit qualifies as seasonal from a case and precipitation perspective.

    Duration of the seasonal block is in months.

    """
    # paths
    root_path = Path(workspace.files_path)
    pipeline_path = root_path / "pipelines" / "snt_seasonality"
    data_path = root_path / "data" / "seasonality"

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
        }
        validate_parameters(params)

        if not run_report_only:
            files_to_ds = []
            error_messages = ["ERROR 1", "ERROR 2", "ERROR 3"]
            seasonality_nb = pipeline_path / "code" / "SNT_seasonality.ipynb"

            # Precipitation seasonality
            if run_precipitation:
                current_run.log_info(f"Running precipitation analysis with notebook : {seasonality_nb}")
                try:
                    params["type_of_seasonality"] = "precipitation"
                    run_notebook_for_type(
                        nb_path=seasonality_nb,
                        seasonality_type="precipitation",
                        out_nb_path=pipeline_path / "papermill_outputs",
                        parameters=params,
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
                    params["type_of_seasonality"] = "cases"
                    run_notebook_for_type(
                        nb_path=seasonality_nb,
                        seasonality_type="cases",
                        out_nb_path=pipeline_path / "papermill_outputs",
                        parameters=params,
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
            nb_file=pipeline_path / "reporting" / "SNT_seasonality_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
        )

        current_run.log_info("Pipeline completed successfully!")

    except Exception as e:
        current_run.log_error(f"Pipeline failed: {e}")
        raise


def load_configuration_snt(config_path: str) -> dict:
    """Load the SNT configuration from a JSON file.

    Parameters
    ----------
    config_path : str
        Path to the configuration JSON file.

    Returns
    -------
    dict
        The loaded configuration as a dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    ValueError
        If the configuration file contains invalid JSON.
    Exception
        For any other unexpected errors.
    """
    try:
        # Load the JSON file
        with Path.open(config_path, "r") as file:
            config_json = json.load(file)
        current_run.log_info(f"SNT configuration loaded: {config_path}")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: The file {config_path} was not found.") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Error: The file contains invalid JSON {e}") from e
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}") from e

    return config_json


def validate_config(config: dict) -> None:
    """Validate that the critical configuration values are set properly."""
    try:
        snt_config = config["SNT_CONFIG"]
        dataset_ids = config["SNT_DATASET_IDENTIFIERS"]
        definitions = config["DHIS2_DATA_DEFINITIONS"]
    except KeyError as e:
        raise KeyError(f"Missing top-level key in config: {e}") from e

    # Required keys in SNT_CONFIG
    required_snt_keys = [
        "COUNTRY_CODE",
        "DHIS2_ADMINISTRATION_1",
        "DHIS2_ADMINISTRATION_2",
        "ANALYTICS_ORG_UNITS_LEVEL",
        "POPULATION_ORG_UNITS_LEVEL",
        "SHAPES_ORG_UNITS_LEVEL",
    ]
    for key in required_snt_keys:
        if key not in snt_config or snt_config[key] in [None, ""]:
            raise ValueError(f"Missing or empty configuration for: SNT_CONFIG.{key}")

    # Required dataset identifiers
    required_dataset_keys = [
        "DHIS2_DATASET_EXTRACTS",
        "DHIS2_DATASET_FORMATTED",
        "DHIS2_REPORTING_RATE",
        "DHIS2_INCIDENCE",
        "WORLDPOP_DATASET_EXTRACTS",
        "ERA5_DATASET_CLIMATE",
        "SNT_SEASONALITY",
        "SNT_MAP_EXTRACT",
    ]
    for key in required_dataset_keys:
        if key not in dataset_ids or dataset_ids[key] in [None, ""]:
            raise ValueError(f"Missing or empty configuration for: SNT_DATASET_IDENTIFIERS.{key}")

    # Check population indicator
    pop_indicators = definitions.get("POPULATION_INDICATOR_DEFINITIONS", {})
    tot_population = pop_indicators.get("TOT_POPULATION", [])
    if not tot_population:
        raise ValueError("Missing or empty TOT_POPULATION indicator definition.")

    # Check at least one indicator under DHIS2_INDICATOR_DEFINITIONS
    indicator_defs = definitions.get("DHIS2_INDICATOR_DEFINITIONS", {})
    flat_indicators = [val for sublist in indicator_defs.values() for val in sublist]
    if not flat_indicators:
        raise ValueError("No indicators defined under DHIS2_INDICATOR_DEFINITIONS.")


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


def load_dataset_file_from(dataset_id: str, filename: str) -> object:
    """Load a file from a dataset by its ID and filename.

    Parameters
    ----------
    dataset_id : str
        The ID of the dataset to load the file from.
    filename : str
        The name of the file to load.

    Returns
    -------
    object
        The loaded file object.

    Raises
    ------
    Exception
        If there is an error loading the file from the dataset.
    """
    try:
        snt_dataset = workspace.get_dataset(dataset_id)
        return snt_dataset.latest_version.get_file(filename=filename)
    except Exception as e:
        raise Exception(f"Error loading the file {filename} from dataset {dataset_id}") from e


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


def add_files_to_dataset(
    dataset_id: str,
    country_code: str,
    file_paths: list[str],
) -> bool:
    """Add files to a new dataset version.

    Parameters
    ----------
    dataset_id : str
        The ID of the dataset to which files will be added.
    country_code : str
        The country code used for naming the dataset version.
    file_paths : list[str]
        A list of file paths to be added to the dataset.

    Raises
    ------
    ValueError
        If the dataset ID is not specified in the configuration.

    Returns
    -------
    bool
        True if at least one file was added successfully, False otherwise.
    """
    added_any = False

    for file in file_paths:
        src = Path(file)
        if not src.exists():
            current_run.log_warning(f"File not found: {src}")
            continue

        try:
            # Determine file extension
            ext = src.suffix.lower()
            if ext == ".parquet":
                df = pd.read_parquet(src)
                tmp_suffix = ".parquet"
            elif ext == ".csv":
                df = pd.read_csv(src)
                tmp_suffix = ".csv"
            elif ext == ".geojson":
                gdf = gpd.read_file(src)
                tmp_suffix = ".geojson"
            else:
                current_run.log_warning(f"Unsupported file format: {src.name}")
                continue

            with tempfile.NamedTemporaryFile(suffix=tmp_suffix) as tmp:
                if ext == ".parquet":
                    df.to_parquet(tmp.name)
                elif ext == ".csv":
                    df.to_csv(tmp.name, index=False)
                elif ext == ".geojson":
                    gdf.to_file(tmp.name, driver="GeoJSON")

                if not added_any:
                    new_version = get_new_dataset_version(ds_id=dataset_id, prefix=f"{country_code}_snt")
                    current_run.log_info(f"New dataset version created : {new_version.name}")
                    added_any = True
                new_version.add_file(tmp.name, filename=src.name)
                current_run.log_info(f"File {src.name} added to dataset version : {new_version.name}")
        except Exception as e:
            current_run.log_warning(f"File {src.name} cannot be added : {e}")
            continue

    if not added_any:
        current_run.log_info("No valid files found. Dataset version was not created.")
        return False

    return True


def get_new_dataset_version(ds_id: str, prefix: str = "ds") -> DatasetVersion:
    """Create and return a new dataset version.

    Parameters
    ----------
    ds_id : str
        The ID of the dataset for which a new version will be created.
    prefix : str, optional
        Prefix for the dataset version name (default is "ds").

    Returns
    -------
    DatasetVersion
        The newly created dataset version.

    Raises
    ------
    Exception
        If an error occurs while creating the new dataset version.
    """
    existing_datasets = workspace.list_datasets()
    if ds_id in [eds.slug for eds in existing_datasets]:
        dataset = workspace.get_dataset(ds_id)
    else:
        current_run.log_warning(f"Dataset with ID {ds_id} not found, creating a new one.")
        dataset = workspace.create_dataset(
            name=ds_id.replace("-", "_").upper(), description="SNT Process dataset"
        )

    version_name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    try:
        new_version = dataset.create_version(version_name)
    except Exception as e:
        raise Exception(f"An error occurred while creating the new dataset version: {e}") from e

    return new_version


def run_report_notebook(
    nb_file: Path,
    nb_output_path: Path,
    ready: bool = True,
) -> None:
    """Execute a Jupyter notebook using Papermill.

    Parameters
    ----------
    nb_file : Path
        The full file path to the notebook.
    nb_output_path : Path
        The path to the directory where the output notebook will be saved.
    ready : bool, optional
        Whether the notebook should be executed (default is True).
    """
    if not ready:
        current_run.log_info("Reporting execution skipped.")
        return

    current_run.log_info(f"Executing report notebook: {nb_file}")
    execution_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    nb_output_full_path = nb_output_path / f"{nb_file.stem}_OUTPUT_{execution_timestamp}.ipynb"
    nb_output_path.mkdir(parents=True, exist_ok=True)

    try:
        pm.execute_notebook(input_path=nb_file, output_path=nb_output_full_path)
    except CellTimeoutError as e:
        raise CellTimeoutError(f"Notebook execution timed out: {e}") from e
    except Exception as e:
        raise Exception(f"Error executing the notebook {type(e)}: {e}") from e
    generate_html_report(nb_output_full_path)


def generate_html_report(output_notebook_path: Path, out_format: str = "html") -> None:
    """Generate an HTML report from a Jupyter notebook.

    Parameters
    ----------
    output_notebook_path : Path
        Path to the output notebook file.
    out_format : str
        output extension

    Raises
    ------
    RuntimeError
        If an error occurs during the conversion process.
    """
    if not output_notebook_path.is_file() or output_notebook_path.suffix.lower() != ".ipynb":
        raise RuntimeError(f"Invalid notebook path: {output_notebook_path}")

    report_path = output_notebook_path.with_suffix(".html")
    current_run.log_info(f"Generating HTML report {report_path}")
    cmd = [
        "jupyter",
        "nbconvert",
        f"--to={out_format}",
        str(output_notebook_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except CalledProcessError as e:
        raise CalledProcessError(f"Error converting notebook to HTML (exit {e.returncode}): {e}") from e

    current_run.add_file_output(report_path.as_posix())


if __name__ == "__main__":
    snt_seasonality()
