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
    snt_dhis2_formatted_path = snt_root_path / "data" / "dhis2_formatted"

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
        dhis2_analytics_formatting(snt_root_path=snt_root_path, pipeline_root_path=snt_pipeline_path)
        dhis2_population_formatting(snt_root_path=snt_root_path, pipeline_root_path=snt_pipeline_path)
        dhis2_shapes_formatting(snt_root_path=snt_root_path, pipeline_root_path=snt_pipeline_path)
        dhis2_pyramid_formatting(snt_root_path=snt_root_path, pipeline_root_path=snt_pipeline_path)

        # add files to a new dataset version
        add_files_to_dataset(
            dataset_id=snt_config_dict["SNT_DATASET_IDENTIFIERS"].get("DHIS2_DATASET_FORMATTED", None),
            country_code=country_code,
            file_paths=[
                snt_dhis2_formatted_path / f"{country_code}_routine_data.parquet",
                snt_dhis2_formatted_path / f"{country_code}_routine_data.csv",
                snt_dhis2_formatted_path / f"{country_code}_population_data.parquet",
                snt_dhis2_formatted_path / f"{country_code}_population_data.csv",
                snt_dhis2_formatted_path / f"{country_code}_shapes_data.geojson",
                snt_dhis2_formatted_path / f"{country_code}_pyramid_data.parquet",
                snt_dhis2_formatted_path / f"{country_code}_pyramid_data.csv",
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
    current_run.log_info("Formatting DHIS2 shapes data.")

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
) -> None:
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
    """
    if dataset_id is None:
        raise ValueError("DHIS2_DATASET_EXTRACTS is not specified in the configuration.")

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
            else:
                current_run.log_warning(f"Unsupported file format: {src.name}")
                continue

            with tempfile.NamedTemporaryFile(suffix=tmp_suffix) as tmp:
                if ext == ".parquet":
                    df.to_parquet(tmp.name)
                else:
                    df.to_csv(tmp.name, index=False)

                if not added_any:
                    new_version = get_new_dataset_version(ds_id=dataset_id, prefix=f"{country_code}_snt")
                    current_run.log_info(f"New dataset version created : {new_version.name}")
                    added_any = True
                new_version.add_file(tmp.name, filename=src.name)
                current_run.log_info(f"File {src.name} added to dataset version : {new_version.name}")
        except Exception as e:
            current_run.log_warning(f"File {src.name} cannot be saved : {e}")
            continue

    if not added_any:
        current_run.log_info("No valid files found. Dataset version was not created.")


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
