import re
import tempfile
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd
import papermill as pm
import polars as pl
from nbclient.exceptions import CellTimeoutError
from openhexa.sdk import current_run, parameter, pipeline, workspace
from openhexa.sdk.workspaces.connection import DHIS2Connection
from openhexa.toolbox.dhis2 import DHIS2
from papermill.exceptions import PapermillExecutionError
from openhexa.toolbox.dhis2.dataframe import get_organisation_units
from openhexa.toolbox.dhis2.periods import period_from_string
from snt_lib.snt_pipeline_utils import (
    handle_rkernel_error_with_labels,
    pull_scripts_from_repository,
    generate_html_report,
    get_new_dataset_version,
    load_configuration_snt,
    validate_config,
    delete_raw_files,
)


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
    required=True,
)
@parameter(
    "end",
    name="Period (end)",
    help="End of DHIS2 period (YYYYMM)",
    type=int,
    default=None,
    required=True,
)
@parameter(
    "overwrite",
    name="Overwrite",
    help="Overwrite existing files",
    type=bool,
    default=True,
    required=False,
)
@parameter(
    "pull_scripts",
    name="Pull Scripts",
    help="Pull the latest scripts from the repository",
    type=bool,
    default=False,
    required=False,
)
@parameter(
    "run_report_only",
    name="Run reporting only",
    help="This will only execute the reporting notebook",
    type=bool,
    default=False,
    required=False,
)
def snt_dhis2_extract(
    dhis2_connection: DHIS2Connection,
    start: int,
    end: int,
    overwrite: bool,
    run_report_only: bool,
    pull_scripts: bool,
) -> None:
    """Write your pipeline code here.

    Pipeline functions should only call tasks and should never perform IO operations or
    expensive computations.
    """
    try:
        validate_period_range(start, end)
    except ValueError as e:
        current_run.log_error(f"Start and end period validation failed: {e}")
        raise

    # Set paths
    snt_root_path = Path(workspace.files_path)
    pipeline_path = snt_root_path / "pipelines" / "snt_dhis2_extract"
    dhis2_raw_data_path = snt_root_path / "data" / "dhis2" / "extracts_raw"
    dhis2_raw_data_path.mkdir(parents=True, exist_ok=True)
    current_run.log_debug(f"output directory: {dhis2_raw_data_path}")

    # pull pipeline scripts if requested
    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_dhis2_extract",
            report_scripts=["snt_dhis2_extract_report.ipynb"],
            code_scripts=[],
        )

    try:
        if not run_report_only:
            # Load configuration
            snt_config_dict = load_configuration_snt(
                config_path=snt_root_path / "configuration" / "SNT_config.json"
            )

            # Validate configuration
            validate_config(snt_config_dict)

            # get country identifier for file naming
            country_code = snt_config_dict["SNT_CONFIG"].get("COUNTRY_CODE", None)
            if country_code is None:
                current_run.log_warning("COUNTRY_CODE is not specified in the configuration.")

            # DHIS2 connection
            dhis2_client = get_dhis2_client(
                dhis2_connection=dhis2_connection, cache_folder=pipeline_path / ".cache"
            )

            # get dhis2 pyramid
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

            reporting_ready = download_dhis2_reporting_rates(
                start=start,
                end=end,
                source_pyramid=dhis2_pyramid,
                dhis2_client=dhis2_client,
                snt_config=snt_config_dict,
                output_dir=dhis2_raw_data_path / "reporting_data",
                overwrite=overwrite,
                ready=analytics_ready,
            )

            files_ready = add_files_to_dataset(
                dataset_id=snt_config_dict["SNT_DATASET_IDENTIFIERS"].get("DHIS2_DATASET_EXTRACTS", None),
                country_code=country_code,
                org_unit_level=snt_config_dict["SNT_CONFIG"].get("ANALYTICS_ORG_UNITS_LEVEL", None),
                file_paths=[
                    dhis2_raw_data_path / "routine_data" / f"{country_code}_dhis2_raw_analytics.parquet",
                    dhis2_raw_data_path / "population_data" / f"{country_code}_dhis2_raw_population.parquet",
                    dhis2_raw_data_path / "shapes_data" / f"{country_code}_dhis2_raw_shapes.parquet",
                    dhis2_raw_data_path / "pyramid_data" / f"{country_code}_dhis2_raw_pyramid.parquet",
                    dhis2_raw_data_path / "reporting_data" / f"{country_code}_dhis2_raw_reporting.parquet",
                ],
                analytics_ready=analytics_ready,
                pop_ready=pop_ready,
                shapes_ready=shapes_ready,
                pyramid_ready=pyramid_ready,
                reporting_ready=reporting_ready,
            )

        else:
            files_ready = True
            current_run.log_info("Skipping data extraction and running report only.")

        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_dhis2_extract_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            ready=files_ready,
        )

    except Exception as e:
        current_run.log_error(f"Error in pipeline execution: {e}")
        raise


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
        cache_folder.mkdir(parents=True, exist_ok=True)
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
        current_run.log_info("Downloading DHIS2 organisation units data.")
        # retrieve the pyramid data
        dhis2_pyramid = get_organisation_units(dhis2_client)

        country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE", "")
        country_code = country_code.upper().strip()  # not None
        current_run.log_debug(f"Country code found: {country_code}")

        # NOTE: Filtering for Burkina Faso due to mixed levels in the pyramid (district: "DS")
        if country_code == "BFA":
            current_run.log_info("Filtering district names at level 4 for Burkina Faso pyramid.")
            dhis2_pyramid = dhis2_pyramid.filter(pl.col("level_4_name").str.starts_with("DS"))

        # NOTE: Filtering for Niger due to mixed levels in the pyramid (district: "DS")
        if country_code == "NER":
            current_run.log_info("Filtering district names at level 3 for Niger pyramid.")
            dhis2_pyramid = dhis2_pyramid.filter(pl.col("level_3_name").str.starts_with("DS"))

        current_run.log_info(f"{country_code} DHIS2 pyramid data retrieved: {len(dhis2_pyramid)} records")
    except Exception as e:
        raise Exception(f"An error occurred while retrieving the DHIS2 pyramid data: {e}") from e

    return dhis2_pyramid


@snt_dhis2_extract.task
def download_dhis2_reporting_rates(
    start: int,
    end: int,
    source_pyramid: pl.DataFrame,
    dhis2_client: DHIS2,
    snt_config: dict,
    output_dir: Path,
    overwrite: bool = True,
    ready: bool = True,
) -> bool:
    """Download and save DHIS2 reporting rates data.

    Returns
    -------
    bool
        True if the reporting rates data is successfully downloaded and saved.
    """
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE", None)
    if country_code is None:
        raise ValueError("COUNTRY_CODE is not specified in the configuration.")

    # Org units level selection for reporting
    org_unit_level_name = snt_config["SNT_CONFIG"].get("DHIS2_ADMINISTRATION_2", None)
    if org_unit_level_name is None:
        raise ValueError("DHIS2_ADMINISTRATION_2 value not configured.")

    # Extract the numeric level from a string like "level_2_name"
    match = re.search(r"level_(\d+)_", org_unit_level_name)
    if not match:
        raise ValueError(f"Could not extract org unit level from '{org_unit_level_name}'")
    org_unit_level = int(match.group(1))

    # get levels and list of OU ids
    df_pyramid = source_pyramid.filter(pl.col("level") == org_unit_level).drop(
        ["level", "opening_date", "closed_date", "geometry"]
    )
    org_units_list = df_pyramid["id"].unique().to_list()
    current_run.log_info(
        f"Downloading analytics for {len(org_units_list)} org units at level {org_unit_level}"
    )

    # setup periods
    p1 = period_from_string(str(start))
    p2 = period_from_string(str(end))
    periods = [p1] if p1 == p2 else p1.get_range(p2)

    rep_datasets = snt_config["DHIS2_DATA_DEFINITIONS"]["DHIS2_REPORTING_RATES"].get("REPORTING_DATASETS", [])
    rep_indicators = snt_config["DHIS2_DATA_DEFINITIONS"]["DHIS2_REPORTING_RATES"].get(
        "REPORTING_INDICATORS", {}
    )

    try:
        ds_downloaded = handle_reporting_datasets(
            reporting_datasets=rep_datasets,
            country_code=country_code,
            org_units_list=org_units_list,
            org_unit_level=org_unit_level,
            pyramid_df=df_pyramid,
            dhis2_client=dhis2_client,
            periods=periods,
            output=output_dir,
            overwrite=overwrite,
        )
        ind_downloaded = handle_reporting_indicators(
            ds_downloaded=ds_downloaded,
            reporting_indicators=rep_indicators,
            country_code=country_code,
            org_units_list=org_units_list,
            org_unit_level=org_unit_level,
            pyramid_df=df_pyramid,
            dhis2_client=dhis2_client,
            periods=periods,
            output=output_dir,
            overwrite=overwrite,
        )
    except Exception as e:
        raise Exception(f"Error while retrieving reporting rates: {e}") from e

    if not (ds_downloaded or ind_downloaded):
        current_run.log_info("No reporting rates to download.")

    return True


def handle_reporting_datasets(
    reporting_datasets: list,
    country_code: str,
    org_units_list: list,
    org_unit_level: int,
    pyramid_df: pl.DataFrame,
    dhis2_client: DHIS2,
    periods: list,
    output: Path,
    overwrite: bool,
) -> bool:
    """Download reporting datasets from DHIS2 and save them after validation and formatting.

    Returns
    -------
    bool
        True if the reporting datasets are successfully downloaded and saved, False otherwise.
    """
    if len(reporting_datasets) == 0:
        current_run.log_info("No reporting datasets provided to download.")
        return False

    # Download datasets metadata
    current_run.log_info("Downloading datasets metadata from DHIS2.")
    datasets_metadata = dhis2_client.meta.datasets()
    datasets_metadata_df = pl.DataFrame(datasets_metadata)

    # validate reporting_rates
    valid_reporting_rates = validate_reporting_rates(reporting_datasets, datasets_metadata_df)
    if len(valid_reporting_rates) == 0:
        current_run.log_info("No valid reporting dataset configured. Process skipped.")
        return False

    output_path = output / "datasets"
    output_path.mkdir(parents=True, exist_ok=True)

    # delete the available reporting file (clean)
    delete_raw_files(output, pattern=f"{country_code}_dhis2_raw_reporting.parquet")

    try:
        if overwrite:
            delete_raw_files(output_path, pattern=f"{country_code}_raw_reporting_ds_*.parquet")
    except Exception as e:
        raise Exception(f"Error while cleaning old raw reporting rate files: {e}") from e

    try:
        current_run.log_info(f"Downloading reporting data for period : {periods[0]} to {periods[-1]}")
        for p in periods:
            fp = output_path / f"{country_code}_raw_reporting_ds_{p}.parquet"

            if fp.exists():
                current_run.log_info(f"File {fp} already exists. Skipping download.")
                continue

            reporting_rate_period = []
            for rate in valid_reporting_rates:
                dataset_uid = rate.get("DATASET")
                metrics = rate.get("METRICS", [])
                reporting_des = [f"{ds}.{metric}" for ds, metric in product([dataset_uid], metrics)]
                # Get dataset name
                try:
                    # Filter and select the name
                    filtered_name_series = (
                        datasets_metadata_df.filter(pl.col("id") == dataset_uid).select("name").to_series()
                    )
                    if filtered_name_series.is_empty():
                        dataset_name = "[Name not found]"
                    else:
                        dataset_name = filtered_name_series.item(0)
                        if dataset_name is None or not dataset_name:
                            dataset_name = "[Name not found]"
                except Exception as e:
                    dataset_name = "[Name not found]"
                    current_run.log_debug(f"An unexpected error occurred during dataset name retrieval: {e}")

                current_run.log_info(
                    f"Downloading reporting rates {list(metrics.keys())} from : "
                    f"{dataset_name} ({dataset_uid}) period : {p}"
                )
                try:
                    data_elements = dhis2_client.analytics.get(
                        data_elements=reporting_des,
                        periods=[p],
                        org_units=org_units_list,
                        include_cocs=False,
                    )
                except Exception as e:
                    current_run.log_warning(f"An error occurred while downloading data for period {p} : {e}")
                    continue

                if len(data_elements) == 0:
                    current_run.log_warning(f"No data found for period {p}")
                    continue

                # Format the extracted data
                data_elements_df = pl.DataFrame(data_elements)
                df_formatted = raw_reporting_ds_format(
                    df=data_elements_df,
                    dataset_name=dataset_name,
                    org_unit_level=org_unit_level,
                    pyramid_metadata=pyramid_df,
                )
                reporting_rate_period.append(df_formatted)

            # concat
            reporting_rates_concat = pl.concat(reporting_rate_period, how="vertical")
            reporting_rates_concat.write_parquet(fp, use_pyarrow=True)

    except Exception as e:
        raise Exception(f"Error while downloading reporting rates: {e}") from e

    try:
        # merge all parquet files into one
        merge_parquet_files(
            input_dir=output_path,
            output_dir=output,
            output_fname=f"{country_code}_dhis2_raw_reporting.parquet",
            file_pattern=f"{country_code}_raw_reporting_ds_*.parquet",
        )
    except Exception as e:
        raise Exception(f"Error while merging reporting parquet files: {e}") from e

    return True


def handle_reporting_indicators(
    ds_downloaded: bool,
    reporting_indicators: dict,
    country_code: str,
    org_units_list: list,
    org_unit_level: int,
    pyramid_df: pl.DataFrame,
    dhis2_client: DHIS2,
    periods: list,
    output: Path,
    overwrite: bool,
) -> bool:
    """Download reporting indicators from DHIS2 and save them after validation and formatting.

    Returns
    -------
    bool
        True if the reporting indicators are successfully downloaded and saved, False otherwise.
    """
    if ds_downloaded:
        current_run.log_info("Reporting dataset downloaded. skip indicators")
        return False

    if len(reporting_indicators) == 0:
        current_run.log_info("No reporting indicators provided to download.")
        return False

    # reporting_indicators = {"ACTUAL_REPORTS" : "","EXPECTED_REPORTS": None}
    valid_indicators = {
        key: value for key, value in reporting_indicators.items() if isinstance(value, str) and value.strip()
    }

    if not valid_indicators:
        current_run.log_info("No reporting indicators provided to download in configuration.")
        return False

    output_path = output / "indicators"
    output_path.mkdir(parents=True, exist_ok=True)

    # delete the available reporting file (clean)
    delete_raw_files(output, pattern=f"{country_code}_dhis2_raw_reporting.parquet")
    current_run.log_info(f"Downloading indicators metrics : {valid_indicators}")

    try:
        if overwrite:
            delete_raw_files(output_path, pattern=f"{country_code}_raw_reporting_ind_*.parquet")
    except Exception as e:
        raise Exception(f"Error while cleaning raw indicator rate files: {e}") from e

    try:
        current_run.log_info(f"Downloading reporting indicators for period : {periods[0]} to {periods[-1]}")
        for p in periods:
            fp = output_path / f"{country_code}_raw_reporting_ind_{p}.parquet"

            if fp.exists():
                current_run.log_info(f"File {fp} already exists. Skipping download.")
                continue

            current_run.log_info(f"Downloading reporting indicators metrics for period : {p}")
            try:
                data_elements = dhis2_client.analytics.get(
                    indicators=list(valid_indicators.values()),
                    periods=[p],
                    org_units=org_units_list,
                    include_cocs=False,
                )
            except Exception as e:
                current_run.log_warning(f"An error occurred while downloading data for period {p} : {e}")
                continue

            if len(data_elements) == 0:
                current_run.log_warning(f"No data found for period {p}")
                continue

            # Format the extracted data
            data_elements_df = pd.DataFrame(data_elements)
            data_elements_df = dhis2_client.meta.add_dx_name_column(dataframe=data_elements_df)
            if not isinstance(data_elements_df, pd.DataFrame):
                raise TypeError("Expected a pandas DataFrame output after adding dx_names")  # pyright
            df_formatted = raw_reporting_ind_format(
                df=data_elements_df,
                metrics=valid_indicators,
                org_unit_level=org_unit_level,
                pyramid_metadata=pyramid_df,
            )

            # df_formatted.write_parquet(fp, use_pyarrow="pyarrow")
            df_formatted.to_parquet(fp, index=False)

    except Exception as e:
        raise Exception(f"Error while downloading reporting rates: {e}") from e

    try:
        # merge all parquet files into one
        merge_parquet_files(
            input_dir=output_path,
            output_dir=output,
            output_fname=f"{country_code}_dhis2_raw_reporting.parquet",
            file_pattern=f"{country_code}_raw_reporting_ind_*.parquet",
        )
    except Exception as e:
        raise Exception(f"Error while merging reporting indicators parquet files: {e}") from e

    return True


def raw_reporting_ds_format(
    df: pl.DataFrame,
    dataset_name: str,
    org_unit_level: int,
    pyramid_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """Format reporting data by splitting data element codes, add dataset names and join with pyramid.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing raw reporting data.
    dataset_name : str
        Name of the dataset to which the reporting data belongs.
    org_unit_level : int
        Organisational unit level for merging.
    pyramid_metadata : pl.DataFrame
        DataFrame containing pyramid (org unit) metadata.

    Returns
    -------
    pl.DataFrame
        Formatted DataFrame with additional columns and joined metadata.
    """
    merging_col = f"level_{org_unit_level}_id"
    df = (
        df.with_columns(
            [
                pl.col("dx").str.split(".").list.get(0).alias("product_uid"),
                pl.col("dx").str.split(".").list.get(1).alias("product_metric"),
                pl.col("value").cast(pl.Float64),
                pl.lit(dataset_name).alias("product_name"),
            ]
        )
        .drop("dx")
        .rename({"ou": merging_col})
    )

    # Add parent level names (left join with pyramid)
    parent_cols = [
        f"level_{ou}{suffix}" for ou in range(1, org_unit_level + 1) for suffix in ["_id", "_name"]
    ]

    return df.join(
        pyramid_metadata.select(parent_cols), how="left", left_on=merging_col, right_on=merging_col
    )


def raw_reporting_ind_format(
    df: pd.DataFrame,
    metrics: dict,
    org_unit_level: int,
    pyramid_metadata: pl.DataFrame,
) -> pd.DataFrame:
    """Format reporting indicator data by mapping metrics and joining with pyramid metadata.

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame with mapped metrics and joined pyramid metadata.
    """
    # Invert the dictionary: value -> key
    metric_map = {v: k for k, v in metrics.items()}
    df["product_metric"] = df["dx"].map(metric_map)

    # rename
    df = df.rename(columns={"dx": "product_uid", "dx_name": "product_name"})

    merging_col = f"level_{org_unit_level}_id"
    parent_cols = [
        f"level_{ou}{suffix}" for ou in range(1, org_unit_level + 1) for suffix in ["_id", "_name"]
    ]
    pyramid_pd = pyramid_metadata.to_pandas()
    merged = df.merge(pyramid_pd[parent_cols], how="left", left_on="ou", right_on=merging_col)
    return merged.drop(columns=["ou"])


def validate_reporting_rates(rates: list, datasets_metadata: pl.DataFrame) -> list:
    """Validate the reporting rates configuration against available DHIS2 datasets metadata.

    Parameters
    ----------
    rates : list
        List of reporting rate definitions to validate.
    datasets_metadata : pl.DataFrame
        DataFrame containing metadata about available DHIS2 datasets.

    Returns
    -------
    list
        List of valid reporting rate definitions.
    """
    valid_rates = []
    for rate in rates:
        valid_rate = {"DATASET": None, "METRICS": []}

        dataset_uid = rate.get("DATASET", "").strip()
        metrics = rate.get("METRICS")
        if dataset_uid in datasets_metadata["id"]:
            valid_rate["DATASET"] = dataset_uid
        else:
            if dataset_uid:
                current_run.log_warning(
                    f"Reporting dataset {dataset_uid} not found in DHIS2 available datasets."
                )

        if metrics and len(metrics) > 0:
            valid_rate["METRICS"] = metrics
        else:
            current_run.log_warning(f"No metrics defined for reporting dataset {dataset_uid}")

        # Only add valid rates with both dataset and metrics defined
        if valid_rate["DATASET"] and valid_rate["METRICS"]:
            valid_rates.append(valid_rate)

    return valid_rates


@snt_dhis2_extract.task
def download_dhis2_analytics(
    start: int,
    end: int,
    source_pyramid: pl.DataFrame,
    dhis2_client: DHIS2,
    snt_config: dict,
    output_dir: Path,
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
    output_dir.mkdir(parents=True, exist_ok=True)

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

    p1 = period_from_string(str(start))
    p2 = period_from_string(str(end))
    periods = [p1] if p1 == p2 else p1.get_range(p2)

    try:
        if overwrite:
            delete_raw_files(output_dir, pattern=f"{country_code}_raw_analytics_*.parquet")
    except Exception as e:
        raise Exception(f"Error while deleting raw analytics files: {e}") from e

    try:
        current_run.log_info(f"Downloading routine data for period : {periods[0]} to {periods[-1]}")
        for p in periods:
            fp = output_dir / f"{country_code}_raw_analytics_{p}.parquet"

            if fp.exists():
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
                df_orgunits = df.merge(  # type: ignore[reportAttributeAccessIssue]
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
    input_dir: Path,
    output_dir: Path,
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
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_fname
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

    The period for population data has a yearly (YYYY) frequency.

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
    if country_code is None:
        raise ValueError("COUNTRY_CODE is not specified in the configuration.")

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract the numeric level from a string like "level_2_name"
    amd2_level = snt_config["SNT_CONFIG"].get("DHIS2_ADMINISTRATION_2", None)
    match = re.search(r"level_(\d+)_", amd2_level)
    if not match:
        raise ValueError(f"Could not extract org unit level from DHIS2_ADMINISTRATION_2 '{amd2_level}'")
    org_unit_level = int(match.group(1))
    max_level = source_pyramid["level"].max()
    current_run.log_debug(f"Population org unit level : {org_unit_level} max level: {max_level}")
    if org_unit_level is None or org_unit_level < 1 or org_unit_level > max_level:
        raise ValueError(f"Incorrect DHIS2_ADMINISTRATION_2 value, please set between 1 and {max_level}.")

    org_unit_uids = source_pyramid.filter(pl.col("level") == org_unit_level)["id"].unique().to_list()
    p1 = period_from_string(str(start)[:4])
    p2 = period_from_string(str(end)[:4])
    periods = [p1] if p1 == p2 else p1.get_range(p2)

    pop_indicators: dict = snt_config["DHIS2_DATA_DEFINITIONS"].get("POPULATION_INDICATOR_DEFINITIONS", {})
    if len(pop_indicators) == 0:
        current_run.log_warning("No population indicators defined under POPULATION_INDICATOR_DEFINITIONS.")
        return False

    try:
        if overwrite:
            # clean all <country_code> raw files
            delete_raw_files(output_dir / "indicators_raw", pattern=f"{country_code}_raw_*_*.parquet")
    except Exception as e:
        raise Exception(f"Error while deleting raw analytics files: {e}") from e

    current_run.log_info(f"Downloading population for period : {periods[0]} to {periods[-1]}")

    # Loop over population definitions and retrieve data depending on type
    for indicator in pop_indicators.items():
        name = indicator[0]
        ids = indicator[1].get("ids", [])
        indicator_type = indicator[1].get("type", [])
        for period in periods:
            fp = output_dir / "indicators_raw" / f"{country_code}_raw_{name}_{period}.parquet"
            if fp.exists():
                current_run.log_info(f"File {fp} already exists. Skipping download.")
                continue

            if len(ids) == 0:
                # raise ValueError(f"No ids defined for population indicator {name}.")
                current_run.log_warning(f"No ids defined for population indicator {name}. Skipping.")
                continue

            if indicator_type not in ("indicator", "dataElement"):
                raise ValueError(
                    f"Invalid population indicator type for {name}. Must be 'indicator' or 'dataElement'."
                )

            # download based on type
            current_run.log_info(f"Downloading {name} for period {period}")
            if indicator_type == "dataElement":
                download_data_elements(
                    dhis2_client=dhis2_client,
                    data_elements=list(set(ids)),
                    period=period,
                    org_units=org_unit_uids,
                    filepath=output_dir / "indicators_raw" / f"{country_code}_raw_{name}_{period}.parquet",
                )
            else:  # indicators
                download_indicators(
                    dhis2_client=dhis2_client,
                    indicators=list(set(ids)),
                    period=period,
                    org_units=org_unit_uids,
                    filepath=output_dir / "indicators_raw" / f"{country_code}_raw_{name}_{period}.parquet",
                )

    try:
        # collect all results and merge into one file
        current_run.log_info("Merging population files.")
        merge_parquet_files(
            input_dir=output_dir / "indicators_raw",
            output_dir=output_dir,
            output_fname=f"{country_code}_dhis2_raw_population.parquet",
            file_pattern=f"{country_code}_raw_*_*.parquet",
        )
    except Exception as e:
        raise Exception(f"Error while merging population parquet files: {e}") from e

    return True


def download_data_elements(
    dhis2_client: DHIS2,
    data_elements: list,
    period: str,
    org_units: list[str],
    filepath: Path,
) -> None:
    """Download population data elements from DHIS2 and save them as a parquet file.

    Parameters
    ----------
    dhis2_client : DHIS2
        DHIS2 client instance for API interaction.
    data_elements : list
        List of data element IDs to download.
    period : str
        Period for which to download the data.
    org_units : list[str]
        List of organisation unit IDs.
    filepath : Path
        Path to save the downloaded data as a parquet file.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    population_values = dhis2_client.analytics.get(
        data_elements=data_elements,
        periods=[period],
        org_units=org_units,
    )

    if len(population_values) == 0:
        current_run.log_warning(f"No population data found for period {period}.")
        return

    current_run.log_info(
        f"Population data element(s) downloaded for period {period}: {len(population_values)} data values"
    )
    population_values_df = pl.DataFrame(population_values)
    population_values_df.write_parquet(filepath)


def download_indicators(
    dhis2_client: DHIS2,
    indicators: list,
    period: str,
    org_units: list[str],
    filepath: Path,
) -> None:
    """Download population indicator data from DHIS2 and save it as a parquet file.

    Parameters
    ----------
    dhis2_client : DHIS2
        DHIS2 client instance for API interaction.
    indicators : list
        List of indicator IDs to download.
    period : str
        Period for which to download the data.
    org_units : list[str]
        List of organisation unit IDs.
    filepath : Path
        Path to save the downloaded data as a parquet file.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    population_values = dhis2_client.analytics.get(
        indicators=indicators,
        periods=[period],
        org_units=org_units,
        include_cocs=False,
    )
    if len(population_values) == 0:
        current_run.log_warning(f"No population indicator data found for period {period}.")
        return

    current_run.log_info(
        f"Population indicator data downloaded for period {period}: {len(population_values)} data values"
    )
    population_values_df = pl.DataFrame(population_values)
    # Add a CO column with None values
    population_values_df = population_values_df.with_columns(pl.lit(None).alias("co"))
    population_values_df.write_parquet(filepath)


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
    output_dir.mkdir(parents=True, exist_ok=True)

    # get org unit level
    # TODO: move this validation to validate_config()
    admin_level_2 = snt_config["SNT_CONFIG"].get("DHIS2_ADMINISTRATION_2")
    match = re.search(r"\d+", admin_level_2)
    if match:
        org_level = int(match.group())
    else:
        raise ValueError(
            f"Invalid DHIS2_ADMINISTRATION_2 format expected: 'level_NUMBER_name' received: {admin_level_2}"
        )

    max_level = source_pyramid["level"].max()
    if max_level is None:
        raise ValueError("source_pyramid['level'] is empty, cannot determine max level")

    if org_level is None or org_level > max_level or org_level < 1:  # type: ignore[operator]
        raise ValueError(
            f"Incorrect DHIS2_ADMINISTRATION_2 value, please set a value between level 1 and {max_level}."
        )

    try:
        df_lvl_selection = source_pyramid.filter(pl.col("level") == org_level).drop(
            ["id", "name", "level", "opening_date", "closed_date"]
        )
        current_run.log_info(f"{df_lvl_selection.shape[0]} shapes downloaded for level {org_level}.")
    except Exception as e:
        raise Exception(f"Error while filtering shapes data: {e}") from e

    try:
        fp = output_dir / f"{country_code}_dhis2_raw_shapes.parquet"
        df_lvl_selection_pd = df_lvl_selection.to_pandas()
        df_lvl_selection_pd.columns = df_lvl_selection_pd.columns.str.upper()  # to UPPER case
        df_lvl_selection_pd.to_parquet(fp, engine="pyarrow", index=False)
        current_run.log_info(f"{country_code} DHIS2 Shapes data saved at : {fp}")
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
    output_dir.mkdir(parents=True, exist_ok=True)

    org_unit_level = snt_config["SNT_CONFIG"].get("ANALYTICS_ORG_UNITS_LEVEL", None)
    if org_unit_level is None:
        raise ValueError("ANALYTICS_ORG_UNITS_LEVEL is not specified in the configuration.")

    try:
        df_lvl_selection = source_pyramid.filter(pl.col("level") == org_unit_level).drop(
            ["id", "name", "level"]
        )
    except Exception as e:
        raise Exception(f"An error occured while downloading the DHIS2 pyramid : {e}") from e

    try:
        fp = output_dir / f"{country_code}_dhis2_raw_pyramid.parquet"
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
    reporting_ready: bool,
) -> bool:
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
    reporting_ready : bool, optional
        Whether the task is ready to run after reporting data.

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
        return False

    return True


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
        "data",
        "pipelines",
        "results",
    ]
    for relative_path in folders_to_create:
        full_path = root_path / relative_path
        full_path.mkdir(parents=True, exist_ok=True)


@snt_dhis2_extract.task
def run_report_notebook(
    nb_file: Path,
    nb_output_path: Path,
    nb_parameters: dict | None = None,
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
    nb_parameters : dict | None, optional
        A dictionary of parameters to pass to the notebook (default is None).
    """
    if not ready:
        current_run.log_info("Reporting execution skipped.")
        return

    current_run.log_info(f"Executing report notebook: {nb_file}")
    execution_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    nb_output_full_path = nb_output_path / f"{nb_file.stem}_OUTPUT_{execution_timestamp}.ipynb"
    nb_output_path.mkdir(parents=True, exist_ok=True)
    warning_raised = False
    try:
        pm.execute_notebook(input_path=nb_file, output_path=nb_output_full_path, parameters=nb_parameters)
    except PapermillExecutionError as e:
        handle_rkernel_error_with_labels(
            e,
            error_labels={"[WARNING]": "warning"},
        )  # for labeled R kernel errors
        warning_raised = True
    except CellTimeoutError as e:
        raise CellTimeoutError(f"Notebook execution timed out: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error executing the notebook ({type(e).__name__}): {e}") from e
    if not warning_raised:
        generate_html_report(nb_output_full_path)


def validate_yyyymm(value: int) -> None:
    """Validate that the input is an integer in yyyymm format."""
    yyyymm_str = str(value)
    if len(yyyymm_str) != 6:
        raise ValueError("Input must be a 6-digit integer in yyyymm format.")

    # year = int(yyyymm_str[:4])
    month = int(yyyymm_str[4:])
    if month < 1 or month > 12:
        raise ValueError(f"Month must be between 01 and 12 in {value}, got {month}.")


def validate_period_range(start: int, end: int) -> None:
    """Validate that start and end are valid yyyymm values and ordered correctly."""
    if start is None or end is None:
        raise ValueError("Start or end period is missing.")

    validate_yyyymm(start)
    validate_yyyymm(end)

    if start > end:
        raise ValueError("Start period must not be after end period.")


if __name__ == "__main__":
    snt_dhis2_extract()
