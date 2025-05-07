import json
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import papermill as pm
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
    snt_root_path = Path(workspace.files_path)
    dhis2_raw_data_path = snt_root_path / "data" / "dhis2_raw"

    # Set up folders
    snt_folders_setup(snt_root_path)

    try:
        # Load configuration
        # NOTE: check if the configuration is valid in load_configuration_snt function (!)
        # is_valid_configuration(snt_config_dict) ## contains CONFIG? TEST?
        snt_config_dict = load_configuration_snt(
            config_path=snt_root_path / "configuration" / "SNT_config.json"
        )

        # get country identifier for file naming
        country_code = snt_config_dict["SNT_CONFIG"].get("COUNTRY_CODE", None)
        if country_code is None:
            current_run.log_warning("COUNTRY_CODE is not specified in the configuration.")

        # DHIS2 connection
        dhis2_client = get_dhis2_client(
            dhis2_connection=dhis2_connection,
            cache_folder=None,  # snt_root_path / snt_dhis2_extract / ".cache"
        )

        # get the dhis2 pyramid
        dhis2_pyramid = get_dhis2_pyramid(dhis2_client=dhis2_client, snt_config=snt_config_dict)

        pop_ready = download_dhis2_population(
            start=start,
            end=end,
            source_pyramid=dhis2_pyramid,
            dhis2_client=dhis2_client,
            snt_config=snt_config_dict,
            output_dir=dhis2_raw_data_path / "population_data",
            overwrite=overwrite,
        )

        shapes_ready = download_dhis2_shapes(
            source_pyramid=dhis2_pyramid,
            output_dir=dhis2_raw_data_path / "shapes_data",
            snt_config=snt_config_dict,
        )

        pyramid_ready = download_dhis2_pyramid(
            source_pyramid=dhis2_pyramid,
            output_dir=dhis2_raw_data_path / "pyramid_data",
            snt_config=snt_config_dict,
        )

        analytics_ready = download_dhis2_analytics(
            start=start,
            end=end,
            source_pyramid=dhis2_pyramid,
            dhis2_client=dhis2_client,
            snt_config=snt_config_dict,
            output_dir=dhis2_raw_data_path / "routine_data",
            overwrite=overwrite,
            ready=pop_ready,
        )

        add_files_to_dataset(
            dataset_id=snt_config_dict["SNT_DATASET_IDENTIFIERS"].get("DHIS2_DATASET_EXTRACTS", None),
            country_code=country_code,
            org_unit_level=snt_config_dict["SNT_CONFIG"].get("ANALYTICS_ORG_UNITS_LEVEL", None),
            file_paths=[
                dhis2_raw_data_path / "routine_data" / f"{country_code}_dhis2_raw_analytics.parquet",
                dhis2_raw_data_path / "population_data" / f"{country_code}_dhis2_raw_population.parquet",
                dhis2_raw_data_path / "shapes_data" / f"{country_code}_dhis2_raw_shapes.parquet",
                dhis2_raw_data_path / "pyramid_data" / f"{country_code}_dhis2_raw_pyramid.parquet",
            ],
            analytics_ready=analytics_ready,
            pop_ready=pop_ready,
            shapes_ready=shapes_ready,
            pyramid_ready=pyramid_ready,
        )

    except Exception as e:
        current_run.log_error(f"Error in pipeline execution: {e}")
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


def get_dhis2_pyramid(dhis2_client: DHIS2, snt_config: dict) -> pl.DataFrame:
    """Get the DHIS2 pyramid data.

    Parameters
    ----------
    dhis2_client : DHIS2
        The DHIS2 client instance for API interaction.
    snt_config : dict
        Configuration dictionary for SNT settings.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the DHIS2 pyramid data.

    Raises
    ------
    Exception
        If an error occurs while retrieving the pyramid data.
    """
    try:
        current_run.log_info("Downloading DHIS2 pyramid data.")
        # retrieve the pyramid data
        dhis2_pyramid = get_organisation_units(dhis2_client)

        country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE", None)
        adm1 = snt_config["SNT_CONFIG"].get("DHIS2_ADMINISTRATION_1", None)
        adm2 = snt_config["SNT_CONFIG"].get("DHIS2_ADMINISTRATION_2", None)

        # NOTE: Filtering for Burkina Faso due to mixed levels in the pyramid (district: "DS")
        if country_code == "BFA" and ("level_4" in adm1 or "level_4" in adm2):
            current_run.log_info("District level (4) filtering for Burkina Faso pyramid")
            dhis2_pyramid = dhis2_pyramid.filter(
                (pl.col("level_4_name").str.starts_with("DS")) & (pl.col("geometry").is_not_null())
            )

        current_run.log_info(f"{country_code} DHIS2 pyramid data retrieved: {len(dhis2_pyramid)} records")
    except Exception as e:
        raise Exception(f"An error occurred while retrieving the DHIS2 pyramid data: {e}") from e

    return dhis2_pyramid


# Task 2 run DHIS2 analytics extract and formatting
@snt_dhis2_extract.task
def download_dhis2_analytics(
    start: int,
    end: int,
    source_pyramid: pl.DataFrame,
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
    source_pyramid : pl.DataFrame
        DataFrame containing the DHIS2 pyramid data.
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
    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE", None)
    if country_code is None:
        raise ValueError("COUNTRY_CODE is not specified in the configuration.")

    # Org units level selection for reporting
    max_level = source_pyramid["level"].max()
    org_unit_level = snt_config["SNT_CONFIG"].get("ANALYTICS_ORG_UNITS_LEVEL", None)
    if org_unit_level is None or org_unit_level < 1 or org_unit_level > max_level:
        raise ValueError(
            f"Incorrect ANALYTICS_ORG_UNITS_LEVEL value, please configure a value between 1 and {max_level}."
        )

    # get levels and list of OU ids
    df_pyramid = source_pyramid.filter(pl.col("level") == org_unit_level).drop(
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
        current_run.log_info(f"Downloading routine data for period : {periods[0]} to {periods[-1]}")
        for p in periods:
            fp = Path(output_dir) / f"{country_code}_raw_analytics_{p}.parquet"

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
                    current_run.log_warning(f"No analytics data found for period {p}.")
                    continue

                current_run.log_info(
                    f"Rountine data downloaded for period {p}: {len(data_values)} data values"
                )
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
            output_fname=f"{country_code}_dhis2_raw_analytics.parquet",
            file_pattern=f"{country_code}_raw_analytics_*.parquet",
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

        # to UPPER case
        df_merged.columns = df_merged.columns.str.upper()

        # Ensure the output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / output_fname
        df_merged.to_parquet(output_path, engine="pyarrow", index=False)
        current_run.log_info(f"Merged file saved at : {output_path}")


@snt_dhis2_extract.task
def download_dhis2_population(
    start: int,
    end: int,
    source_pyramid: pl.DataFrame,
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
    source_pyramid : pl.DataFrame
        DataFrame containing the DHIS2 pyramid data.
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
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE", None)

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
    max_level = source_pyramid["level"].max()
    if org_unit_level is None or org_unit_level < 1 or org_unit_level > max_level:
        raise ValueError(f"Incorrect POPULATION_ORG_UNITS_LEVEL value, please set between 1 and {max_level}.")

    try:
        p1 = period_from_string(str(start))
        p2 = period_from_string(str(end))
        periods = [p1] if p1 == p2 else p1.get_range(p2)
    except Exception as e:
        raise Exception(f"Error ocurred when computing periods start: {start} end : {end} Error : {e}") from e

    try:
        # Get the organisation units for the specified level
        df_pyramid = source_pyramid.filter(pl.col("level") == org_unit_level).drop(
            ["level", "opening_date", "closed_date", "geometry"]
        )
        org_unit_uids = df_pyramid["id"].unique().to_list()
        df_pyramid = df_pyramid.to_pandas()

        current_run.log_info(f"Downloading population for period : {periods[0]} to {periods[-1]}")
        for p in periods:
            fp = Path(output_dir) / f"{country_code}_raw_population_{p}.parquet"

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
                    current_run.log_warning(f"No population data found for period {p}.")
                    continue

                current_run.log_info(
                    f"Population data downloaded for period {p}: {len(population_values)} data values"
                )
                population_values_df = pd.DataFrame(population_values)

                # Add parent level names (left join with pyramid)
                merging_col = f"level_{org_unit_level}_id"
                parent_cols = [
                    f"level_{ou}{suffix}"
                    for ou in range(1, org_unit_level + 1)
                    for suffix in ["_id", "_name"]
                ]
                merged_df = population_values_df.merge(
                    df_pyramid[parent_cols], how="left", left_on="ou", right_on=merging_col
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
            output_fname=f"{country_code}_dhis2_raw_population.parquet",
            file_pattern=f"{country_code}_raw_population_*.parquet",
        )
    except Exception as e:
        raise Exception(f"Error while merging parquet files: {e}") from e

    return True


@snt_dhis2_extract.task
def download_dhis2_shapes(
    source_pyramid: pl.DataFrame,
    output_dir: Path,
    snt_config: dict,
) -> bool:
    """Download and save DHIS2 shapes data.

    Parameters
    ----------
    source_pyramid : pl.DataFrame
        DataFrame containing the DHIS2 pyramid data.
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
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE", None)

    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    org_levels = snt_config["SNT_CONFIG"].get("SHAPES_ORG_UNITS_LEVEL", None)
    max_level = source_pyramid["level"].max()
    if org_levels is None or org_levels > max_level or org_levels < 1:
        raise ValueError(
            f"Incorrect SHAPES_ORG_UNITS_LEVEL value, please configure a value between 1 and {max_level}."
        )

    try:
        df_lvl_selection = source_pyramid.filter(pl.col("level") == org_levels).drop(
            ["id", "name", "level", "opening_date", "closed_date"]
        )
        current_run.log_info(f"{df_lvl_selection.shape[0]} shapes downloaded for level {org_levels}.")
    except Exception as e:
        raise Exception(f"Error while filtering shapes data: {e}") from e

    try:
        fp = Path(output_dir) / f"{country_code}_dhis2_raw_shapes.parquet"
        df_lvl_selection_pd = df_lvl_selection.to_pandas()
        df_lvl_selection_pd.columns = df_lvl_selection_pd.columns.str.upper()  # to UPPER case
        df_lvl_selection_pd.to_parquet(fp, engine="pyarrow", index=False)
        current_run.log_info(f"Shapes data saved at : {fp}")
    except Exception as e:
        raise Exception(f"Error while saving shapes data: {e}") from e

    return True


@snt_dhis2_extract.task
def download_dhis2_pyramid(source_pyramid: pl.DataFrame, output_dir: Path, snt_config: dict) -> None:
    """Download and save DHIS2 pyramid data.

    Parameters
    ----------
    source_pyramid : pl.DataFrame
        DataFrame containing the DHIS2 pyramid data.
    output_dir : Path
        Directory to save the downloaded pyramid data.
    snt_config : dict
        Configuration dictionary for SNT settings.
    ready : bool
        Whether the task is ready to run.
    """
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE", None)

    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    org_unit_level = snt_config["SNT_CONFIG"].get("ANALYTICS_ORG_UNITS_LEVEL", None)
    if org_unit_level is None:
        raise ValueError("ANALYTICS_ORG_UNITS_LEVEL is not specified in the configuration.")

    try:
        df_lvl_selection = source_pyramid.filter(pl.col("level") == org_unit_level).drop(
            ["id", "name", "level", "opening_date", "closed_date", "geometry"]
        )
    except Exception as e:
        raise Exception(f"An error occured while downloading the DHIS2 pyramid : {e}") from e

    try:
        fp = Path(output_dir) / f"{country_code}_dhis2_raw_pyramid.parquet"
        df_lvl_selection_pd = df_lvl_selection.to_pandas()
        df_lvl_selection_pd.columns = df_lvl_selection_pd.columns.str.upper()  # to UPPER case
        df_lvl_selection_pd.to_parquet(fp, engine="pyarrow", index=False)
        current_run.log_info(f"{country_code} DHIS2 pyramid data saved at : {fp}")
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
    country_code: str,
    org_unit_level: str,
    file_paths: list[str],
    analytics_ready: bool,
    pop_ready: bool,
    shapes_ready: bool,
    pyramid_ready: bool,
) -> None:
    """Add files to a dataset version in the workspace.

    Parameters
    ----------
    dataset_id : str
        The ID of the dataset to which files will be added.
    country_code : str
        The country code for the dataset.
    org_unit_level : str
        The level of the organisation unit for which the DHIS2 Analytics have been downloaded.
    file_paths : str
        Paths to the files to be added to the dataset.
    analytics_ready : bool, optional
        Whether the task is ready to run after analytics.
    pop_ready : bool, optional
            Whether the task is ready to run after population data.
    shapes_ready : bool, optional
        Whether the task is ready to run after shapes data.
    pyramid_ready : bool, optional
        Whether the task is ready to run after pyramid data.

    Raises
    ------
    Exception
        If an error occurs while creating a new dataset version or adding files.
    """
    if dataset_id is None:
        raise ValueError("DHIS2_DATASET_EXTRACTS is not specified in the configuration.")
    if country_code is None:
        current_run.log_warning("COUNTRY_CODE is not specified in the configuration.")

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
                    new_version = get_new_dataset_version(
                        ds_id=dataset_id, prefix=f"{country_code}_dhis2_level{org_unit_level}"
                    )
                    current_run.log_info(f"New dataset version created : {new_version.name}")
                    added_any = True
                new_version.add_file(tmp.name, filename=src.name)
                current_run.log_info(f"File {src.name} added to dataset version : {new_version.name}")
        except Exception as e:
            current_run.log_warning(f"File {src.name} cannot be added : {e}")
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


def snt_folders_setup(root_path: Path) -> None:
    """Set up the required folder structure for the SNT pipeline.

    Parameters
    ----------
    root_path : Path
        The root directory where the folder structure will be created.

    This function creates a predefined set of folders under the specified root path
    to ensure the necessary directory structure for the SNT pipeline.
    NOTE : The option is to use the snt_config.json file(!).
    """
    folders_to_create = [
        "configuration",
        "code",
        "data/dhis2_raw/population_data",
        "data/dhis2_raw/pyramid_data",
        "data/dhis2_raw/routine_data",
        "data/dhis2_raw/shapes_data",
        "data/dhis2_formatted",
        "pipelines/snt_dhis2_extract",
        "pipelines/snt_dhis2_formatting/code",
        "pipelines/snt_dhis2_formatting/papermill_outputs",
    ]
    for relative_path in folders_to_create:
        full_path = root_path / relative_path
        full_path.mkdir(parents=True, exist_ok=True)


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


if __name__ == "__main__":
    snt_dhis2_extract()
