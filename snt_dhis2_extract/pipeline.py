import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl

from openhexa.sdk import current_run, parameter, pipeline, workspace
from openhexa.sdk.datasets.dataset import DatasetVersion
from openhexa.toolbox.dhis2 import DHIS2
from openhexa.toolbox.dhis2.dataframe import get_organisation_units
from openhexa.toolbox.dhis2.periods import period_from_string


@pipeline("snt_dhis2_extract")
@parameter(
    "start",
    name="Period (start)",
    help="Start of DHIS2 period (YYYYMM)",
    type=int,
    default=None,
    required=False,
)
@parameter(
    "end",
    name="Period (end)",
    help="End of DHIS2 period (YYYYMM)",
    type=int,
    default=None,
    required=False,
)
def snt_dhis2_extract(start: int, end: int):
    """Write your pipeline code here.

    Pipeline functions should only call tasks and should never perform IO operations or
    expensive computations.
    """
    if start is None or end is None:
        current_run.log_error("Empty period input start-end")
        raise ValueError

    if end < start:
        current_run.log_error("End period should be greater than start period")
        raise ValueError

    # Set paths
    snt_root_path = Path(workspace.files_path).joinpath("SNT Process")
    pipeline_root_path = Path(workspace.files_path).joinpath("pipelines", "snt_dhis2_extract")

    try:
        # Load configuration
        snt_config_dict = load_configuration_snt(
            config_path=Path(snt_root_path).joinpath("configuration", "SNT_config.json")
        )

        # DHIS2 connection
        dhis2_client = get_dhis2_client(
            snt_config=snt_config_dict, cache_folder=Path().joinpath(pipeline_root_path, ".cache")
        )

        analytics_ready = download_dhis2_analytics(
            start=start,
            end=end,
            dhis2_client=dhis2_client,
            snt_config=snt_config_dict,
            output_dir=Path(snt_root_path).joinpath("data", "raw_DHIS2", "routine_data"),
            overwrite=False,
        )

        population_ready = download_dhis2_population(
            start=start,
            end=end,
            dhis2_client=dhis2_client,
            snt_config=snt_config_dict,
            output_dir=Path(snt_root_path).joinpath("data", "raw_DHIS2", "population_data"),
            overwrite=True,
            ready=analytics_ready,
        )

        # # RUN Shapes download and formatting
        # shapes_ready = download_dhis2_shapes(
        #     snt_root_path=snt_root_path,
        #     population_process_ready=population_process_ready,
        # )

        # # download country pyramid reference
        # get_dhis2_pyramid(
        #     snt_root_path=snt_root_path,
        #     pyramid_data=True,
        #     shapes_process_ready=shapes_process_ready,
        # )

        add_files_to_dataset(
            dataset_id="snt-dhis2-extracts",
            file_paths=[
                Path(snt_root_path).joinpath(
                    "data", "raw_DHIS2", "routine_data", "dhis2_raw_analytics.parquet"
                ),
                Path(snt_root_path).joinpath(
                    "data", "raw_DHIS2", "population_data", "dhis2_raw_population.parquet"
                ),
                # Path(snt_root_path).joinpath(
                #     "data", "raw_DHIS2", "population_data", "population_data.parquet"
                # ),
                # Path(snt_root_path).joinpath("data", "raw_DHIS2", "shapes_data", "shapes_data.parquet"),
                # Path(snt_root_path).joinpath("data", "raw_DHIS2", "pyramid_data.parquet"),
            ],
            ready=population_ready,  ## CHANGE THIS!!
        )

    except Exception as e:
        current_run.log_error(f"Error in pipeline execution: {e}")
        raise


@snt_dhis2_extract.task
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


@snt_dhis2_extract.task
def get_dhis2_client(snt_config: dict, cache_folder: Path) -> DHIS2:
    """Create and return a DHIS2 client instance.

    Parameters
    ----------
    snt_config : dict
        Configuration dictionary containing DHIS2 connection details.
    cache_folder : Path
        Path to the folder where cache files will be stored.

    Returns
    -------
    DHIS2
        An instance of the DHIS2 client for API interaction.

    Raises
    ------
    KeyError
        If the required DHIS2 connection key is missing in the configuration.
    Exception
        If an error occurs while connecting to the DHIS2 server.
    """
    try:
        dhis2_connection_id = snt_config["SNT_CONFIG"].get("DHIS2_CONNECTION", None)
    except KeyError as e:
        raise KeyError(f"Error: The key {e} is missing in the JSON structure.") from e
    except Exception as e:
        raise Exception(f"An error occurred while connecting to the server: {e}") from e

    try:
        con = workspace.dhis2_connection(dhis2_connection_id)
        dhis2_client = DHIS2(con, cache_dir=cache_folder)  # set a default cache folder
        current_run.log_info(f"Connected to {con.url}")
    except Exception as e:
        raise Exception(f"An error occurred while connecting to the server: {e}") from e

    return dhis2_client


# Task 2 run DHIS2 analytics extract and formatting
@snt_dhis2_extract.task
def download_dhis2_analytics(
    start: int,
    end: int,
    dhis2_client: DHIS2,
    snt_config: dict,
    output_dir: str,
    overwrite: bool = True,
    ready: bool = True,
) -> bool:
    """Download and save DHIS2 analytics data for the specified period.

    Parameters
    ----------
    start : int
        Start period in YYYYMM format.
    end : int
        End period in YYYYMM format.
    dhis2_client : DHIS2
        DHIS2 client instance for API interaction.
    snt_config : dict
        Configuration dictionary for SNT settings.
    output_dir : str
        Directory to save the downloaded analytics data.
    snt_root_path : str
        Root path for SNT processing.
    pipeline_root_path : str
        Root path for the pipeline.
    overwrite : bool, optional
        Whether to overwrite existing files (default is True).
    ready : bool, optional
        ready to run the task (default is True).

    Returns
    -------
    bool
        True if the analytics data is successfully downloaded and saved.

    Raises
    ------
    KeyError
        If a required key is missing in the configuration dictionary.
    Exception
        If an error occurs during the download or processing of data.
    """
    current_run.log_info("Downloading DHIS2 analytics data.")

    # Org units level selection default to level 2
    org_unit_level = snt_config["SNT_CONFIG"].get("ORG_UNITS_LEVEL_EXTRACT", None)
    if org_unit_level is None:
        raise ValueError("ORG_UNITS_LEVEL_EXTRACT is not specified in the configuration.")

    # get levels and list of OU ids
    org_units = pl.DataFrame(dhis2_client.meta.organisation_units())
    org_units_list = org_units.filter(pl.col("level") == org_unit_level)["id"].to_list()

    # Unique list of data elements
    data_elements = get_unique_data_elements(
        snt_config["DHIS2_DATA_DEFINITIONS"].get("DHIS2_INDICATOR_DEFINITIONS", None)
    )
    if len(data_elements) == 0:
        raise ValueError("No routine DHSI2 data elements found in the configuration.")

    try:
        p1 = period_from_string(str(start))
        p2 = period_from_string(str(end))
        prange = p1.get_range(p2)
        periods = [str(pe) for pe in prange]
    except Exception as e:
        raise Exception(f"Error ocurred when computing periods start: {start} end : {end} Error : {e}") from e

    try:
        for p in periods:
            current_run.log_info(f"Downloading period : {p}")
            fp = Path(output_dir).joinpath(f"raw_analytics_{p}_lvl_{org_unit_level}.parquet")

            if fp.exists() and not overwrite:
                current_run.log_info(f"File {fp} already exists. Skipping download.")
                continue

            try:
                data_values = dhis2_client.analytics.get(
                    data_elements=data_elements,
                    periods=[p],
                    org_units=org_units_list,
                )

                current_run.log_info(f"Data downloaded for period {p}: {len(data_values)}")
                df = pd.DataFrame(data_values)
                df.to_parquet(fp, engine="pyarrow", index=False)

            except Exception as e:
                current_run.log_warning(f"An error occurred while downloading data for period {p} : {e}")
                continue

    except Exception as e:
        raise Exception(f"Error while downloading: {e}") from e

    # merge all parquet files into one
    try:
        merge_parquet_files(
            input_dir=output_dir,
            output_dir=output_dir,
            output_fname="dhis2_raw_analytics.parquet",
            file_pattern="raw_analytics_*.parquet",
        )
    except Exception as e:
        raise Exception(f"Error while merging parquet files: {e}") from e

    return True


def merge_parquet_files(
    input_dir: str,
    output_dir: str,
    output_fname: str,
    file_pattern: str = "raw_*.parquet",
) -> None:
    """Collect all parquet files from input directory and merge them into a single parquet file.

    Parameters
    ----------
    input_dir : str
        Path to the input directory containing parquet files.
    output_dir : str
        Path to the output directory where the merged file will be saved.
    output_fname : str
        Name of the merged parquet file.
    file_pattern : str, optional
        File name pattern to be merged (default is "raw_*.parquet").
    """
    # Get all parquet files in the input directory
    files = [f.name for f in Path(input_dir).glob(file_pattern)]
    if len(files) == 0:
        current_run.log_warning(f"No files found in {input_dir} matching pattern {file_pattern}.")
    else:
        current_run.log_info(f"Found {len(files)} files to merge.")

        # Read and concatenate all parquet files into a single DataFrame
        df_list = []
        for f in files:
            try:
                df = pd.read_parquet(Path(input_dir) / f)
                df_list.append(df)
            except Exception as e:
                current_run.log_warning(f"Could not read file {f}: {e}")
                continue
        df_merged = pd.concat(df_list, ignore_index=True)

        # Ensure the output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir).joinpath(output_fname)
        df_merged.to_parquet(output_path, engine="pyarrow", index=False)
        current_run.log_info(f"Merged file saved at : {output_path}")


# task 3
@snt_dhis2_extract.task
def download_dhis2_population(
    start: int,
    end: int,
    dhis2_client: DHIS2,
    snt_config: dict,
    output_dir: Path,
    overwrite: bool = True,
    ready: bool = True,
) -> bool:
    """Download and save DHIS2 population data for the specified period.

    Parameters
    ----------
    start : int
        Start period in YYYYMM format.
    end : int
        End period in YYYYMM format.
    dhis2_client : DHIS2
        DHIS2 client instance for API interaction.
    snt_config : dict
        Configuration dictionary for SNT settings.
    output_dir : Path
        Directory to save the downloaded population data.
    overwrite : bool, optional
        Whether to overwrite existing files (default is True).
    ready : bool, optional
        Whether the task is ready to run (default is True).

    Returns
    -------
    bool
        True if the population data is successfully downloaded and saved.

    Raises
    ------
    ValueError
        If required configuration keys are missing or invalid.
    Exception
        If an error occurs during the download or processing of data.
    """
    current_run.log_info("Downloading DHIS2 population data.")

    # Get population data elements
    data_elements = get_unique_data_elements(
        snt_config["DHIS2_DATA_DEFINITIONS"].get("POPULATION_INDICATOR_DEFINITIONS", None)
    )
    if len(data_elements) == 0:
        raise ValueError(
            "No population data elements (POPULATION_INDICATOR_DEFINITIONS) found in the configuration."
        )

    # default to level 2 (all countries have at least this?)
    org_unit_level_extract = snt_config["SNT_CONFIG"].get("POPULATION_ORG_UNITS_LEVEL", None)
    if org_unit_level_extract is None:
        raise ValueError("POPULATION_ORG_UNITS_LEVEL is not specified in the configuration.")

    try:
        # Compute periods
        p1 = period_from_string(str(start))
        p2 = period_from_string(str(end))
        prange = p1.get_range(p2)
        periods = [str(pe) for pe in prange]
    except Exception as e:
        raise Exception(f"Error ocurred when computing periods start: {start} end : {end} Error : {e}") from e

    try:
        column_selection = f"level_{org_unit_level_extract}_id"
        df_pyramid = get_organisation_units(dhis2_client, max_level=org_unit_level_extract)
        df_lvl_selection = df_pyramid.filter(pl.col("level") == org_unit_level_extract).drop(
            ["id", "name", "level", "opening_date", "closed_date", "geometry"]
        )
        org_unit_uids = (
            df_lvl_selection.select(column_selection).unique().get_column(column_selection).to_list()
        )

        for p in periods:
            current_run.log_info(f"Downloading period : {p}")
            fp = Path(output_dir).joinpath(f"raw_population_{p}_lvl_{org_unit_level_extract}.parquet")

            if fp.exists() and not overwrite:
                current_run.log_info(f"File {fp} already exists. Skipping download.")
                continue

            try:
                population_values = dhis2_client.analytics.get(
                    data_elements=data_elements,
                    periods=[p],
                    org_units=org_unit_uids,
                )

                current_run.log_info(f"Data downloaded for period {p}: {len(population_values)}")
                population_values_df = pd.DataFrame(population_values)
                merged_df = population_values_df.merge(
                    df_lvl_selection.to_pandas(), how="left", left_on="ou", right_on=column_selection
                )
                merged_df.to_parquet(fp, engine="pyarrow", index=False)

            except Exception as e:
                current_run.log_warning(f"An error occurred while downloading data for period {p} : {e}")
                continue

    except Exception as e:
        raise Exception(f"Error {e}") from e

    # merge all parquet files into one
    try:
        merge_parquet_files(
            input_dir=output_dir,
            output_dir=output_dir,
            output_fname="dhis2_raw_population.parquet",
            file_pattern="raw_population_*.parquet",
        )
    except Exception as e:
        raise Exception(f"Error while merging parquet files: {e}") from e

    return True


# task 4
@snt_dhis2_extract.task
def download_dhis2_shapes(
    output_dir: Path,
    snt_config: dict,
    ready: bool,
) -> bool:
    """Download and save DHIS2 shapes data.

    Parameters
    ----------
    output_dir : Path
        Directory to save the downloaded shapes data.
    snt_config : dict
        Configuration dictionary for SNT settings.
    ready : bool
        Whether the task is ready to run.

    Returns
    -------
    bool
        True if the shapes data is successfully downloaded and saved.
    """
    current_run.log_info("Downloading DHIS2 shapes data.")

    try:
        pass
    except Exception as e:
        current_run.log_error(f"Papermill Error: {e}")
        raise

    return True


@snt_dhis2_extract.task
def download_dhis2_pyramid(output_dir: Path, ready: bool) -> None:
    """Download and save DHIS2 pyramid data.

    Parameters
    ----------
    output_dir : Path
        Directory to save the downloaded pyramid data.
    ready : bool
        Whether the task is ready to run.
    """
    current_run.log_info("Downloading DHIS2 pyramid data.")

    try:
        pass
    except Exception as e:
        raise Exception(f"An error occured while downloading the DHIS2 pyramid : {e}") from e

    # current_run.log_info(f"DHIS2 pyramid data saved under {'configuration')}.")


def get_unique_data_elements(data_dictionary: dict) -> list[str]:
    """Extract unique data elements from a dictionary of indicator definitions.

    This is a helper function to get the list of data elements from indicator definitions in the config file.

    Parameters
    ----------
    data_dictionary : dict
        A dictionary where keys are indicator names and values are lists of data element strings.

    Returns
    -------
    list[str]
        A list of unique data element identifiers.
    """
    unique_elements = []

    for value_list in data_dictionary.values():
        for item in value_list:
            # Split by dot and add each part to the set, ignore COC
            unique_elements.append(item.split(".")[0])

    return list(set(unique_elements))


@snt_dhis2_extract.task
def add_files_to_dataset(dataset_id: str, file_paths: list[str], ready: bool = True) -> None:
    """Add files to a dataset version in the workspace.

    Parameters
    ----------
    dataset_id : str
        The ID of the dataset to which files will be added.
    file_paths : str
        Paths to the files to be added to the dataset.
    ready : bool, optional
        Whether the task is ready to run (default is True).

    Raises
    ------
    Exception
        If an error occurs while creating a new dataset version or adding files.
    """
    new_version = get_new_dataset_version(ds_id=dataset_id, prefix="dhis2_data")
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
    version_name = f"{prefix}_{datetime.now().strftime('%Y_%m_%d_%H%M')}"

    try:
        new_version = dataset.create_version(version_name)
    except Exception as e:
        raise Exception(f"An error occurred while creating the new dataset version: {e}") from e

    return new_version


if __name__ == "__main__":
    snt_dhis2_extract()
