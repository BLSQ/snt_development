import json
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import geopandas as gpd
import papermill as pm
from openhexa.sdk import current_run, pipeline, workspace, parameter
from openhexa.sdk.datasets.dataset import DatasetVersion
import subprocess
from subprocess import CalledProcessError
from nbclient.exceptions import CellTimeoutError


@pipeline("snt_dhis2_formatting")
@parameter(
    "run_report_only",
    name="Run reporting only",
    help="This will only execute the reporting notebook",
    type=bool,
    default=False,
    required=False,
)
def snt_dhis2_formatting(run_report_only: bool):
    """Write your pipeline orchestration here.

    Pipeline functions should only call tasks and should never perform IO operations or
    expensive computations.
    """
    # set paths
    snt_root_path = Path(workspace.files_path)
    snt_pipeline_path = snt_root_path / "pipelines" / "snt_dhis2_formatting"
    snt_dhis2_formatted_path = snt_root_path / "data" / "dhis2" / "formatted"

    try:
        if not run_report_only:
            # Load configuration
            snt_config_dict = load_configuration_snt(
                config_path=snt_root_path / "configuration" / "SNT_config.json"
            )

            # Validate configuration
            validate_config(snt_config_dict)

            # get country identifier for naming
            country_code = snt_config_dict["SNT_CONFIG"].get("COUNTRY_CODE", None)
            if country_code is None:
                current_run.log_warning("COUNTRY_CODE is not specified in the configuration.")

            # NOTE: check if the configuration is valid in load_configuration_snt function (!)
            # is_valid_configuration(snt_config_dict)

            # format data for SNT
            dhis2_analytics_formatting(snt_root_path=snt_root_path, pipeline_root_path=snt_pipeline_path)
            dhis2_population_formatting(snt_root_path=snt_root_path, pipeline_root_path=snt_pipeline_path)
            dhis2_shapes_formatting(snt_root_path=snt_root_path, pipeline_root_path=snt_pipeline_path)
            dhis2_pyramid_formatting(snt_root_path=snt_root_path, pipeline_root_path=snt_pipeline_path)

            # add files to a new dataset version
            files_ready = add_files_to_dataset(
                dataset_id=snt_config_dict["SNT_DATASET_IDENTIFIERS"].get("DHIS2_DATASET_FORMATTED", None),
                country_code=country_code,
                file_paths=[
                    snt_dhis2_formatted_path / f"{country_code}_routine.parquet",
                    snt_dhis2_formatted_path / f"{country_code}_routine.csv",
                    snt_dhis2_formatted_path / f"{country_code}_population.parquet",
                    snt_dhis2_formatted_path / f"{country_code}_population.csv",
                    snt_dhis2_formatted_path / f"{country_code}_shapes.geojson",
                    snt_dhis2_formatted_path / f"{country_code}_pyramid.parquet",
                    snt_dhis2_formatted_path / f"{country_code}_pyramid.csv",
                ],
            )
        else:
            files_ready = True

        run_report_notebook(
            nb_file=snt_pipeline_path / "reporting" / "SNT_dhis2_indicators_report.ipynb",
            nb_output_path=snt_pipeline_path / "reporting" / "outputs",
            ready=files_ready,
        )

    except Exception as e:
        current_run.log_error(f"Error in SNT DHIS2 formatting: {e}")
        raise


def load_configuration_snt(config_path: str) -> dict:
    """Load the SNT configuration from a JSON file.

    Parameters
    ----------
    config_path : str
        The path to the configuration JSON file.

    Returns
    -------
    dict
        The loaded configuration as a dictionary.

    Raises
    ------
    FileExistsError
        If the configuration file is not found.
    ValueError
        If the configuration file contains invalid JSON.
    RuntimeError
        If an unexpected error occurs during loading.
    """
    try:
        with Path.open(config_path, "r") as file:
            config_json = json.load(file)

        current_run.log_info(f"SNT configuration loaded: {config_path}")

    except FileNotFoundError as e:
        raise FileExistsError(f"Error: The file was not found {e}.") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Error: The file contains invalid JSON {e}.") from e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}") from e

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
        "WORLDPOP_DATASET_EXTRACTS",
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


def dhis2_analytics_formatting(
    snt_root_path: Path,
    pipeline_root_path: Path,
) -> None:
    """Format DHIS2 analytics data for SNT."""
    current_run.log_info("Formatting DHIS2 analytics data.")

    # set parameters for notebook
    nb_parameter = {
        "SNT_ROOT_PATH": str(snt_root_path),
    }

    try:
        run_notebook(
            nb_name="SNT_dhis2_routine_format",
            nb_path=pipeline_root_path / "code",
            out_nb_path=pipeline_root_path / "papermill_outputs",
            parameters=nb_parameter,
        )
    except Exception as e:
        raise Exception(f"Error in formatting analytics data: {e}") from e


def dhis2_population_formatting(
    snt_root_path: Path,
    pipeline_root_path: Path,
) -> None:
    """Format DHIS2 population data for SNT."""
    current_run.log_info("Formatting DHIS2 population data.")

    # set parameters for notebook
    nb_parameter = {
        "SNT_ROOT_PATH": str(snt_root_path),
    }
    try:
        run_notebook(
            nb_name="SNT_dhis2_population_format",
            nb_path=pipeline_root_path / "code",
            out_nb_path=pipeline_root_path / "papermill_outputs",
            parameters=nb_parameter,
        )
    except Exception as e:
        raise Exception(f"Error in formatting population data: {e}") from e


def dhis2_shapes_formatting(
    snt_root_path: Path,
    pipeline_root_path: Path,
) -> None:
    """Format DHIS2 shapes data for SNT."""
    current_run.log_info("Formatting DHIS2 shapes data.")

    # set parameters for notebook
    nb_parameter = {
        "SNT_ROOT_PATH": str(snt_root_path),
    }
    try:
        run_notebook(
            nb_name="SNT_dhis2_shapes_format",
            nb_path=pipeline_root_path / "code",
            out_nb_path=pipeline_root_path / "papermill_outputs",
            parameters=nb_parameter,
        )
    except Exception as e:
        raise Exception(f"Error in formatting shapes data: {e}") from e

    # current_run.log_info(
    #     f"SNT population formatted data saved under: {snt_root_path / 'data' / 'dhis2_formatted'}"
    # )


def dhis2_pyramid_formatting(
    snt_root_path: Path,
    pipeline_root_path: Path,
) -> None:
    """Format DHIS2 pyramid data for SNT."""
    current_run.log_info("Formatting DHIS2 pyramid data.")

    # set parameters for notebook
    nb_parameter = {
        "SNT_ROOT_PATH": str(snt_root_path),
    }
    try:
        run_notebook(
            nb_name="SNT_dhis2_pyramid_format",
            nb_path=pipeline_root_path / "code",
            out_nb_path=pipeline_root_path / "papermill_outputs",
            parameters=nb_parameter,
        )
    except Exception as e:
        raise Exception(f"Error in formatting pyramid data: {e}") from e


def run_notebook(nb_name: str, nb_path: Path, out_nb_path: Path, parameters: dict):
    """Execute a Jupyter notebook using Papermill.

    Parameters
    ----------
    nb_name : str
        The name of the notebook to execute (without the .ipynb extension).
    nb_path : str
        The path to the directory containing the notebook.
    out_nb_path : str
        The path to the directory where the output notebook will be saved.
    parameters : dict
        A dictionary of parameters to pass to the notebook.
    """
    nb_full_path = nb_path / f"{nb_name}.ipynb"
    current_run.log_info(f"Executing notebook: {nb_full_path}")
    execution_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_nb_fname = f"{nb_name}_OUTPUT_{execution_timestamp}.ipynb"
    out_nb_full_path = out_nb_path / out_nb_fname

    try:
        pm.execute_notebook(input_path=nb_full_path, output_path=out_nb_full_path, parameters=parameters)
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
    if dataset_id is None:
        raise ValueError(
            "DHIS2_DATASET_FORMATTED is not specified in the configuration."
        )  # TODO: make the error to refer to the corresponding dataset..

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

    Also creates a new dataset if it does not exist.

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
    nb_file : str
        The full file path to the notebook.
    nb_output_path : str
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

    current_run.add_file_output(str(report_path))


if __name__ == "__main__":
    snt_dhis2_formatting()
