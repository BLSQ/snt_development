import json
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl
from openhexa.sdk import current_run, parameter, pipeline, workspace
from openhexa.sdk.datasets.dataset import DatasetVersion
from openhexa.sdk.workspaces.connection import DHIS2Connection
from openhexa.toolbox.dhis2 import DHIS2
from openhexa.toolbox.dhis2.dataframe import get_organisation_units
from openhexa.toolbox.dhis2.periods import period_from_string


@pipeline("snt_dhis2_extract", timeout=28800)
@parameter(
    "dhis2_connection",
    name="DHIS2 connection",
    help="DHIS2 connection ID",
    type=DHIS2Connection,
    default=None,
    required=True,
)
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
@parameter(
    "overwrite",
    name="Overwrite",
    help="Overwrite existing files",
    type=bool,
    default=True,
    required=False,
)
def snt_dhis2_extract(dhis2_connection: DHIS2Connection, start: int, end: int, overwrite: bool) -> None:
    """Write your pipeline code here.

    Pipeline functions should only call tasks and should never perform IO operations or
    expensive computations.
    """
    if start is None or end is None:
        current_run.log_error("Empty period input start-end")
        raise ValueError

    if start > end:
        current_run.log_error(f"End period {end} should be greater than start period {start}")
        raise ValueError

    # Set paths
    snt_root_path = Path(workspace.files_path) / "SNT Process"
    # pipeline_root_path = Path(workspace.files_path) / "pipelines", "snt_dhis2_extract"
    raw_dhis2_data_path = Path(snt_root_path) / "data" / "raw_DHIS2"

    try:
        # Load configuration
        snt_config_dict = load_configuration_snt(
            config_path=Path(snt_root_path) / "configuration" / "SNT_config.json"
        )

        # DHIS2 connection
        dhis2_client = get_dhis2_client(
            dhis2_connection=dhis2_connection,
            cache_folder=None,  # Path() / pipeline_root_path / ".cache"
        )

        analytics_ready = download_dhis2_analytics(
            start=start,
            end=end,
            dhis2_client=dhis2_client,
            snt_config=snt_config_dict,
            output_dir=Path(raw_dhis2_data_path) / "routine_data",
            overwrite=overwrite,
        )

        pop_ready = download_dhis2_population(
            start=start,
            end=end,
            dhis2_client=dhis2_client,
            snt_config=snt_config_dict,
            output_dir=Path(raw_dhis2_data_path) / "population_data",
            overwrite=overwrite,
        )

        shapes_ready = download_dhis2_shapes(
            output_dir=Path(raw_dhis2_data_path) / "shapes_data",
            dhis2_client=dhis2_client,
            snt_config=snt_config_dict,
        )

        pyramid_ready = download_dhis2_pyramid(
            output_dir=Path(raw_dhis2_data_path) / "pyramid_data",
            dhis2_client=dhis2_client,
            snt_config=snt_config_dict,
        )

        add_files_to_dataset(
            dataset_id="snt-dhis2-extracts",
            dhis2_connection=dhis2_connection,
            snt_config=snt_config_dict,
            file_paths=[
                Path(raw_dhis2_data_path) / "routine_data" / "dhis2_raw_analytics.parquet",
                Path(raw_dhis2_data_path) / "population_data" / "dhis2_raw_population.parquet",
                Path(raw_dhis2_data_path) / "shapes_data" / "raw_shapes.parquet",
                Path(raw_dhis2_data_path) / "pyramid_data" / "dhis2_pyramid.parquet",
            ],
            analytics_ready=analytics_ready,
            pop_ready=pop_ready,
            shapes_ready=shapes_ready,
            pyramid_ready=pyramid_ready,
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


def get_dhis2_client(dhis2_connection: DHIS2Connection, cache_folder: Path) -> DHIS2:
    """Create and return a DHIS2 client instance.

    Parameters
    ----------
    dhis2_connection : DHIS2Connection
        The DHIS2 connection object containing the connection details.
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
        dhis2_client = DHIS2(dhis2_connection, cache_dir=cache_folder)  # set a default cache folder
        current_run.log_info(f"Connected to {dhis2_connection.url}")
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

    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Org units level selection default to level 2
    org_unit_level = snt_config["SNT_CONFIG"].get("ANALYTICS_ORG_UNITS_LEVEL", None)
    if org_unit_level is None:
        raise ValueError("ANALYTICS_ORG_UNITS_LEVEL is not specified in the configuration.")

    # get levels and list of OU ids
    df_pyramid = get_organisation_units(dhis2_client, max_level=org_unit_level)
    df_pyramid = df_pyramid.filter(pl.col("level") == org_unit_level).drop(
        ["level", "opening_date", "closed_date", "geometry"]
    )
    org_units_list = df_pyramid["id"].unique().to_list()
    df_pyramid = df_pyramid.to_pandas()
    current_run.log_info(
        f"Downloading analytics for {len(org_units_list)} org units at level {org_unit_level}"
    )

    # Unique list of data elements
    data_elements = get_unique_data_elements(
        snt_config["DHIS2_DATA_DEFINITIONS"].get("DHIS2_INDICATOR_DEFINITIONS", None)
    )
    if len(data_elements) == 0:
        raise ValueError("No routine DHSI2 data elements found in the configuration.")

    try:
        p1 = period_from_string(str(start))
        p2 = period_from_string(str(end))
        periods = [p1] if p1 == p2 else p1.get_range(p2)
    except Exception as e:
        raise Exception(f"Error ocurred when computing periods start: {start} end : {end} Error : {e}") from e

    try:
        for p in periods:
            current_run.log_info(f"Downloading routine data period : {p}")
            fp = Path(output_dir) / f"raw_analytics_{p}.parquet"

            if fp.exists() and not overwrite:
                current_run.log_info(f"File {fp} already exists. Skipping download.")
                continue

            try:
                data_values = dhis2_client.analytics.get(
                    data_elements=data_elements,
                    periods=[p],
                    org_units=org_units_list,
                )

                if len(data_values) == 0:
                    current_run.log_warning(f"No data found for period {p}.")
                    continue

                current_run.log_info(f"Rountine data downloaded for period {p}: {len(data_values)}")
                df = pd.DataFrame(data_values)

                # Add dx and co names
                df = dhis2_client.meta.add_dx_name_column(dataframe=df)
                df = dhis2_client.meta.add_coc_name_column(dataframe=df)

                # Add parent level names (left join with pyramid)
                merging_col = f"level_{org_unit_level}_id"
                parent_cols = [
                    f"level_{ou}{suffix}"
                    for ou in range(1, org_unit_level + 1)
                    for suffix in ["_id", "_name"]
                ]
                df_orgunits = df.merge(
                    df_pyramid[parent_cols], how="left", left_on="ou", right_on=merging_col
                )

                # save raw data file
                df_orgunits.to_parquet(fp, engine="pyarrow", index=False)
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
        output_path = Path(output_dir) / output_fname
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

    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get population data elements
    data_elements = get_unique_data_elements(
        snt_config["DHIS2_DATA_DEFINITIONS"].get("POPULATION_INDICATOR_DEFINITIONS", None)
    )
    if len(data_elements) == 0:
        raise ValueError(
            "No population data elements (POPULATION_INDICATOR_DEFINITIONS) found in the configuration."
        )

    # default to level 2 (all countries have at least this?)
    org_unit_level = snt_config["SNT_CONFIG"].get("POPULATION_ORG_UNITS_LEVEL", None)
    if org_unit_level is None:
        raise ValueError("POPULATION_ORG_UNITS_LEVEL is not specified in the configuration.")

    try:
        p1 = period_from_string(str(start))
        p2 = period_from_string(str(end))
        periods = [p1] if p1 == p2 else p1.get_range(p2)
    except Exception as e:
        raise Exception(f"Error ocurred when computing periods start: {start} end : {end} Error : {e}") from e

    try:
        column_selection = f"level_{org_unit_level}_id"
        df_pyramid = get_organisation_units(dhis2_client, max_level=org_unit_level)
        df_lvl_selection = df_pyramid.filter(pl.col("level") == org_unit_level).drop(
            ["id", "name", "level", "opening_date", "closed_date", "geometry"]
        )
        org_unit_uids = (
            df_lvl_selection.select(column_selection).unique().get_column(column_selection).to_list()
        )

        for p in periods:
            current_run.log_info(f"Downloading population for period : {p}")
            fp = Path(output_dir) / f"raw_population_{p}.parquet"

            if fp.exists() and not overwrite:
                current_run.log_info(f"File {fp} already exists. Skipping download.")
                continue

            try:
                population_values = dhis2_client.analytics.get(
                    data_elements=data_elements,
                    periods=[p],
                    org_units=org_unit_uids,
                )

                if len(population_values) == 0:
                    current_run.log_warning(f"No data found for period {p}.")
                    continue

                current_run.log_info(f"Population data downloaded for period {p}: {len(population_values)}")
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
    dhis2_client: DHIS2,
    snt_config: dict,
) -> bool:
    """Download and save DHIS2 shapes data.

    Parameters
    ----------
    output_dir : Path
        Directory to save the downloaded shapes data.
    dhis2_client : DHIS2
        DHIS2 client instance for API interaction.
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

    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    org_levels = snt_config["SNT_CONFIG"].get("SHAPES_ORG_UNITS_LEVEL", None)
    if org_levels is None:
        raise ValueError("Organisation level for Shape not defined SHAPES_ORG_UNITS_LEVEL")

    current_run.log_info(f"Downloading shapes for level : {org_levels}")

    try:
        df_pyramid = get_organisation_units(dhis2_client, max_level=org_levels)
        df_lvl_selection = df_pyramid.filter(pl.col("level") == org_levels).drop(
            ["id", "name", "level", "opening_date", "closed_date"]
        )
        current_run.log_info(f"{df_lvl_selection.shape[0]} shapes downloaded for level {org_levels}.")
    except Exception as e:
        raise Exception(f"Error while retrieving shapes data: {e}") from e

    try:
        fp = Path(output_dir) / "raw_shapes.parquet"
        df_lvl_selection_pd = df_lvl_selection.to_pandas()
        df_lvl_selection_pd.to_parquet(fp, engine="pyarrow", index=False)
        current_run.log_info(f"Shapes data saved at : {fp}")
    except Exception as e:
        raise Exception(f"Error while saving shapes data: {e}") from e

    return True


@snt_dhis2_extract.task
def download_dhis2_pyramid(output_dir: Path, dhis2_client: DHIS2, snt_config: dict) -> None:
    """Download and save DHIS2 pyramid data.

    Parameters
    ----------
    output_dir : Path
        Directory to save the downloaded pyramid data.
    dhis2_client : DHIS2
        DHIS2 client instance for API interaction.
    snt_config : dict
        Configuration dictionary for SNT settings.
    ready : bool
        Whether the task is ready to run.
    """
    current_run.log_info("Downloading DHIS2 pyramid data.")

    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    org_unit_level = snt_config["SNT_CONFIG"].get("ANALYTICS_ORG_UNITS_LEVEL", None)
    if org_unit_level is None:
        raise ValueError("ANALYTICS_ORG_UNITS_LEVEL is not specified in the configuration.")

    try:
        df_pyramid = get_organisation_units(dhis2_client, max_level=org_unit_level)
        df_lvl_selection = df_pyramid.filter(pl.col("level") == org_unit_level).drop(
            ["id", "name", "level", "opening_date", "closed_date", "geometry"]
        )
    except Exception as e:
        raise Exception(f"An error occured while downloading the DHIS2 pyramid : {e}") from e

    try:
        fp = Path(output_dir) / "dhis2_pyramid.parquet"
        df_lvl_selection_pd = df_lvl_selection.to_pandas()
        df_lvl_selection_pd.to_parquet(fp, engine="pyarrow", index=False)
        current_run.log_info(f"DHIS2 pyramid data saved at : {fp}")
    except Exception as e:
        raise Exception(f"Error while saving DHIS2 pyramid data: {e}") from e


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
def add_files_to_dataset(
    dataset_id: str,
    dhis2_connection: DHIS2Connection,
    snt_config: dict,
    file_paths: list[str],
    ready: bool = True,
) -> None:
    """Add files to a dataset version in the workspace.

    Parameters
    ----------
    dataset_id : str
        The ID of the dataset to which files will be added.
    dhis2_connection : DHIS2Connection
        The DHIS2 connection object to obtain the connection name (NOT IMPLEMENTED).
    snt_config : dict
        Configuration dictionary for SNT settings.
    file_paths : str
        Paths to the files to be added to the dataset.
    ready : bool, optional
        Whether the task is ready to run (default is True).

    Raises
    ------
    Exception
        If an error occurs while creating a new dataset version or adding files.
    """
    org_unit_level = snt_config["SNT_CONFIG"].get("ANALYTICS_ORG_UNITS_LEVEL", None)
    new_version = get_new_dataset_version(ds_id=dataset_id, prefix=f"DHIS2_ou_level{org_unit_level}")
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
    snt_dhis2_extract()
