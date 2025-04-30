import json
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import papermill as pm
from openhexa.sdk import current_run, pipeline, workspace
from openhexa.sdk.datasets.dataset import DatasetVersion


@pipeline("snt_dhis2_formatting")
def snt_dhis2_formatting():
    """Write your pipeline orchestration here.

    Pipeline functions should only call tasks and should never perform IO operations or
    expensive computations.
    """
    # set paths
    snt_root_path = Path(workspace.files_path)
    snt_pipeline_path = snt_root_path / "pipelines" / "snt_dhis2_formatting"
    snt_dhis2_formatted_path = snt_root_path / "dhis2_formatted"

    try:
        # Load configuration
        snt_config_dict = load_configuration_snt(
            config_path=snt_root_path / "configuration" / "SNT_config.json"
        )
        # get country identifier for naming
        country_code = snt_config_dict["SNT_CONFIG"].get("COUNTRY_CODE", None)
        if country_code is None:
            current_run.log_warning("COUNTRY_CODE is not specified in the configuration.")

        # NOTE: check if the configuration is valid in load_configuration_snt function (!)
        # is_valid_configuration(snt_config_dict)

        # format data for SNT
        dhis2_analytics_formatting(
            snt_config=snt_config_dict, snt_root_path=snt_root_path, pipeline_root_path=snt_pipeline_path
        )
        dhis2_population_formatting(
            snt_config=snt_config_dict, snt_root_path=snt_root_path, pipeline_root_path=snt_pipeline_path
        )
        dhis2_shapes_formatting(
            snt_config=snt_config_dict, snt_root_path=snt_root_path, pipeline_root_path=snt_pipeline_path
        )
        dhis2_pyramid_formatting(
            snt_config=snt_config_dict, snt_root_path=snt_root_path, pipeline_root_path=snt_pipeline_path
        )

        # add files to a new dataset version
        add_files_to_dataset(
            dataset_id=snt_config_dict["SNT_DATASET_IDENTIFIERS"].get("DHIS2_DATASET_FORMATTED", None),
            country_code=country_code,
            file_paths=[
                snt_dhis2_formatted_path / "routine_data" / f"{country_code}_dhis2_raw_analytics.parquet",
                snt_dhis2_formatted_path / "population_data" / f"{country_code}_dhis2_raw_population.parquet",
                snt_dhis2_formatted_path / "shapes_data" / f"{country_code}_raw_shapes.parquet",
                snt_dhis2_formatted_path / "pyramid_data" / f"{country_code}_dhis2_pyramid.parquet",
            ],
        )

    except Exception as e:
        raise Exception(f"Error in SNT DHIS2 formatting: {e}") from e


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


def dhis2_analytics_formatting(
    snt_root_path: Path,
    pipeline_root_path: Path,
) -> int:
    """Format DHIS2 analytics data for SNT."""
    current_run.log_info("Formatting DHIS2 analytics data.")

    # set parameters for notebook
    nb_parameter = {
        "SNT_ROOT_PATH": snt_root_path,
    }
    try:
        run_notebook(
            nb_name="SNT_dhis2_routine_format",
            nb_path=pipeline_root_path / "code",
            out_nb_path=pipeline_root_path / "papermill_outputs",
            parameters=nb_parameter,
        )
    except Exception as e:
        raise Exception(f"Error in formatting population data: {e}") from e


def dhis2_population_formatting(
    snt_root_path: str,
    pipeline_root_path: str,
) -> None:
    """Format DHIS2 population data for SNT."""
    current_run.log_info("Formatting DHIS2 population data.")

    # set parameters for notebook
    nb_parameter = {
        "SNT_ROOT_PATH": snt_root_path,
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
    snt_config: dict,
    snt_root_path: str,
    pipeline_root_path: str,
) -> None:
    """Format DHIS2 shapes data for SNT."""
    current_run.log_info("Formatting DHIS2 shapes data.")

    # set parameters for notebook
    nb_parameter = {
        "SNT_ROOT_PATH": snt_root_path,
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
    # snt_config: dict,
    snt_root_path: str,
    pipeline_root_path: str,
) -> None:
    """Format DHIS2 pyramid data for SNT."""
    current_run.log_info("Formatting DHIS2 shapes data.")

    # set parameters for notebook
    nb_parameter = {
        "SNT_ROOT_PATH": snt_root_path,
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


def run_notebook(nb_name: str, nb_path: str, out_nb_path: str, parameters: dict):
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
    nb_full_path = Path(nb_path) / f"{nb_name}.ipynb"
    current_run.log_info(f"Executing notebook: {nb_full_path}")
    execution_timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    out_nb_fname = f"{nb_name}_OUTPUT_{execution_timestamp}.ipynb"
    out_nb_full_path = Path(out_nb_path) / out_nb_fname

    try:
        pm.execute_notebook(input_path=nb_full_path, output_path=out_nb_full_path, parameters=parameters)
    except Exception as e:
        raise Exception(f"Error executing the notebook {type(e)}: {e}") from e


def add_files_to_dataset(
    dataset_id: str,
    country_code: str,
    file_paths: list[str],
) -> None:
    """Add files to a dataset version in the workspace.

    Parameters
    ----------
    dataset_id : str
        The ID of the dataset to which files will be added.
    country_code : str
        The country code for the dataset.
    file_paths : str
        Paths to the files to be added to the dataset.

    Raises
    ------
    Exception
        If an error occurs while creating a new dataset version or adding files.
    """
    if dataset_id is None:
        raise ValueError("DHIS2_DATASET_EXTRACTS is not specified in the configuration.")

    new_version = get_new_dataset_version(ds_id=dataset_id, prefix=f"{country_code}_snt")

    for file in file_paths:
        src = Path(file)
        if not src.exists():
            current_run.log_warning(f"File not found: {src}")
            continue

        try:
            df = pd.read_parquet(src)
            with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
                df.to_parquet(tmp.name)
                new_version.add_file(tmp.name, filename=src.name)
        except Exception as e:
            current_run.log_warning(f"Dataset file cannot be saved : {e}")
            continue

    current_run.log_info(f"New dataset version created : {new_version.name}")


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
    dataset = workspace.get_dataset(ds_id)
    version_name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    try:
        new_version = dataset.create_version(version_name)
    except Exception as e:
        raise Exception(f"An error occurred while creating the new dataset version: {e}") from e

    return new_version


if __name__ == "__main__":
    snt_dhis2_formatting()
