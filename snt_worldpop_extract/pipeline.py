import json
from datetime import datetime
import tempfile
from pathlib import Path
import pandas as pd

# import papermill as pm
from openhexa.sdk import current_run, pipeline, workspace
from openhexa.sdk.datasets.dataset import DatasetVersion
from worlpopclient import WorldPopClient


@pipeline("snt_worldpop_extract")
def snt_worldpop_extract():
    """Write your pipeline orchestration here."""
    # set paths
    snt_root_path = Path(workspace.files_path)

    # get configuration
    snt_config_dict = load_configuration_snt(config_path=snt_root_path / "configuration" / "SNT_config.json")

    # get country identifier for file naming
    country_code = snt_config_dict["SNT_CONFIG"].get("COUNTRY_CODE", None)
    if country_code is None:
        current_run.log_warning("COUNTRY_CODE is not specified in the configuration.")

    # Set output directory
    output_dir = snt_root_path / "data" / "worldpop_raw" / "population_data"
    pop_file_path = retrieve_population(country=country_code, output_dir=output_dir)

    add_files_to_dataset(
        dataset_id=snt_config_dict["SNT_DATASET_IDENTIFIERS"].get("WORLDPOP_DATASET_EXTRACT", None),
        country_code=country_code,
        file_paths=[pop_file_path],
    )

    # run_report_notebook(
    #     nb_file=pipeline_path / "reporting" / "SNT_dhis2_extract_report.ipynb",
    #     nb_output_path=pipeline_path / "reporting" / "outputs",
    #     ready=files_ready,
    # )


def retrieve_population(country: str, output_path: Path) -> Path:
    """Retrieve raster population data from worldpop.

    Returns
    -------
    Path
        The path to the saved WorldPop population data file.
    """
    current_run.log_info("Retrieve raster population data from worldpop ")
    wpop_client = WorldPopClient()

    try:
        all_datasets = wpop_client.get_datasets_by_country(country_iso3=country)
        last_year = get_latest_population_year(all_datasets)
        current_run.log_info(f"Latest available population data for year : {last_year}")
    except Exception as e:
        current_run.log_error(f"Error retrieving datasets for country {country}: {e}")
        raise

    # Create output directory if it doesn't exist
    Path.mkdir(output_path, exist_ok=True)

    try:
        pop_data_path = wpop_client.get_population_grid_for_country_and_year(
            country_iso3=country,
            year=last_year,
            output_dir=output_path,
            fname=f"wpop_{country}_pop_{last_year}.tif",
        )
        current_run.log_info(f"WorldPop population data saved: {pop_data_path}")
        return pop_data_path
    except ValueError as e:
        current_run.log_warning(f"No data population retrieved {e}")
        return None
    except Exception as e:
        current_run.log_error(f"Error retrieving WorldPop population data: {e}")
        raise


def get_latest_population_year(datasets: list) -> list:
    """Returns the latest population year from a list of dataset dictionaries.

    Parameters
    ----------
    datasets : list
        List of dictionaries, each containing a 'popyear' key.

    Returns
    -------
    int
        The maximum population year found in the datasets.
    """
    return max([ds["popyear"] for ds in datasets])


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


def add_files_to_dataset(
    dataset_id: str,
    country_code: str,
    file_paths: list[Path],
) -> bool:
    """Add files to a dataset version in the workspace.

    Parameters
    ----------
    dataset_id : str
        The ID of the dataset to which files will be added.
    country_code : str
        The country code for the dataset.
    file_paths : list[Path]
        Paths to the files to be added to the dataset.

    Raises
    ------
    Exception
        If an error occurs while creating a new dataset version or adding files.

    Returns
    -------
    Bool
        True if files were successfully added to the dataset version, False otherwise.
    """
    if dataset_id is None:
        raise ValueError(
            "WORLDPOP_DATASET_EXTRACTS is not specified in the configuration."
        )  # TODO: make the error to refer to the corresponding dataset..
    if country_code is None:
        current_run.log_warning("COUNTRY_CODE is not specified in the configuration.")

    added_any = False

    for file in file_paths:
        if not file.exists():
            current_run.log_warning(f"File not found: {file}")
            continue

        try:
            # Determine file extension
            ext = file.suffix.lower()
            if ext == ".parquet":
                df = pd.read_parquet(file)
                tmp_suffix = ".parquet"
            elif ext == ".csv":
                df = pd.read_csv(file)
                tmp_suffix = ".csv"
            else:
                current_run.log_warning(f"Unsupported file format: {file.name}")
                continue

            with tempfile.NamedTemporaryFile(suffix=tmp_suffix) as tmp:
                if ext == ".parquet":
                    df.to_parquet(tmp.name)
                else:
                    df.to_csv(tmp.name, index=False)

                if not added_any:
                    new_version = get_new_dataset_version(ds_id=dataset_id, prefix=f"{country_code}_worldpop")
                    current_run.log_info(f"New dataset version created : {new_version.name}")
                    added_any = True
                new_version.add_file(tmp.name, filename=file.name)
                current_run.log_info(f"File {file.name} added to dataset version : {new_version.name}")
        except Exception as e:
            current_run.log_warning(f"File {file.name} cannot be added : {e}")
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
    dataset = workspace.get_dataset(ds_id)
    version_name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    try:
        new_version = dataset.create_version(version_name)
    except Exception as e:
        raise Exception(f"An error occurred while creating the new dataset version: {e}") from e

    return new_version


if __name__ == "__main__":
    snt_worldpop_extract()
