import json
import subprocess
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from subprocess import CalledProcessError
from nbclient.exceptions import CellTimeoutError

import pandas as pd
import papermill as pm

from openhexa.sdk import current_run, pipeline, workspace
from openhexa.sdk.datasets.dataset import DatasetVersion
from worlpopclient import WorldPopClient


@pipeline("snt_worldpop_extract")
def snt_worldpop_extract():
    """Write your pipeline orchestration here."""
    # set paths
    snt_root_path = Path(workspace.files_path)
    pipeline_path = snt_root_path / "pipelines" / "snt_worldpop_extract"

    try:
        # get configuration
        snt_config_dict = load_configuration_snt(
            config_path=snt_root_path / "configuration" / "SNT_config.json"
        )

        # Validate configuration
        validate_config(snt_config_dict)

        # get country identifier for file naming
        country_code = snt_config_dict["SNT_CONFIG"].get("COUNTRY_CODE", None)
        if country_code is None:
            raise ValueError("COUNTRY_CODE is not specified in the configuration.")

        # Set output directory
        output_dir = snt_root_path / "data" / "worldpop" / "raw" / "population"
        pop_file_path = retrieve_population_data(country=country_code, output_path=output_dir)
        if pop_file_path is None:
            current_run.log_warning("No population data retrieved.")
            return

        # Add files to dataset
        add_files_to_dataset(
            dataset_id=snt_config_dict["SNT_DATASET_IDENTIFIERS"].get("WORLDPOP_DATASET_EXTRACTS", None),
            country_code=country_code,
            file_paths=[pop_file_path],
        )

        # Run report notebook
        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "SNT_wpop_population_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
        )
    except Exception as e:
        current_run.log_error(f"An error occurred in the pipeline: {e}")
        raise


def retrieve_population_data(country: str, output_path: Path) -> Path:
    """Retrieve raster population data from worldpop.

    Returns
    -------
    Path
        The path to the saved WorldPop population data file.
    """
    current_run.log_info("Retrieving population data grid from worldpop.")
    wpop_client = WorldPopClient()
    current_run.log_info(f"Connected to WorldPop endpoint : {wpop_client.base_url}")

    try:
        all_datasets = wpop_client.get_datasets_by_country(country_iso3=country)
        last_year = get_latest_population_year(all_datasets)
        current_run.log_info(f"Latest population data for {country} is from {last_year}.")
    except Exception as e:
        current_run.log_error(f"Error retrieving datasets for country {country}: {e}")
        raise

    # Create output directory if it doesn't exist
    Path.mkdir(output_path, exist_ok=True)

    try:
        pop_tif_path = wpop_client.get_population_geotiff(
            country_iso3=country,
            year=last_year,
            output_dir=output_path,
            fname=f"{country}_wpop_population_{last_year}.tif",
        )
        pop_compressed_path = wpop_client.compress_geotiff(
            src_path=pop_tif_path,
            dst_path=pop_tif_path.with_name(f"{country}_wpop_population_{last_year}_compressed.tif"),
        )
        current_run.log_info(f"WorldPop population data saved: {pop_compressed_path}")
        return pop_compressed_path
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
    if country_code is None:
        current_run.log_warning("COUNTRY_CODE is not specified in the configuration.")

    added_any = False

    for file in file_paths:
        if not file.exists():
            current_run.log_warning(f"File not found: {file}")
            continue

        ext = file.suffix.lower()
        tmp_suffix = file.suffix

        try:
            write_tmp: Callable[[str], None]

            if ext == ".parquet":
                df: pd.DataFrame = pd.read_parquet(file)

                def write_tmp(path: str, df: pd.DataFrame = df) -> None:
                    df.to_parquet(path)

            elif ext == ".csv":
                df: pd.DataFrame = pd.read_csv(file)

                def write_tmp(path: str, df: pd.DataFrame = df) -> None:
                    df.to_csv(path, index=False)

            elif ext == ".tif":

                def write_tmp(path: str, file: Path = file) -> None:
                    shutil.copy(file, path)

            else:
                current_run.log_warning(f"Unsupported file format: {file.name}")
                continue

            with tempfile.NamedTemporaryFile(suffix=tmp_suffix) as tmp:
                write_tmp(tmp.name)

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
) -> None:
    """Execute a Jupyter notebook using Papermill.

    Parameters
    ----------
    nb_file : Path
        The full file path to the notebook.
    nb_output_path : Path
        The path to the directory where the output notebook will be saved.
    """
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

    current_run.add_file_output(str(report_path))


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


if __name__ == "__main__":
    snt_worldpop_extract()
