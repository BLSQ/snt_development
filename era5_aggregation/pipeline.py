import tempfile
from datetime import datetime
import json
import zipfile
from io import BytesIO
from pathlib import Path
from shutil import copyfile

import geopandas as gpd
import polars as pl
import pandas as pd
from openhexa.sdk import Dataset, current_run, parameter, pipeline, workspace
from openhexa.sdk.datasets import DatasetFile
from openhexa.sdk.datasets.dataset import DatasetVersion
from openhexa.toolbox.era5.aggregate import (
    aggregate,
    aggregate_per_month,
    aggregate_per_week,
    build_masks,
    get_transform,
    merge,
)
from openhexa.toolbox.era5.cds import VARIABLES


@pipeline("ERA5 Aggregate")
@parameter(
    "boundaries_dataset",
    name="Boundaries dataset",
    type=Dataset,
    help="Input dataset containing boundaries geometries",
    required=True,
)
@parameter(
    "boundaries_file",
    name="Boundaries filename in dataset",
    type=str,
    help="Filename of the boundaries file to use in the boundaries dataset",
    required=False,
    default="district.parquet",
)
@parameter(
    "boundaries_column_uid",
    name="Boundaries column UID",
    type=str,
    help="Column name containing unique identifier for boundaries geometries",
    required=True,
    default="id",
)
def era5_aggregate(
    boundaries_dataset: Dataset,
    boundaries_column_uid: str,
    boundaries_file: str | None = None,
):
    """Aggregate ERA5 climate data by applying spatial and temporal aggregation to raw input files.

    Parameters
    ----------
    input_dir : str
        Input directory with raw ERA5 extracts.
    output_dir : str
        Output directory for the aggregated data.
    boundaries_dataset : Dataset
        Input dataset containing boundaries geometries.
    boundaries_column_uid : str
        Column name containing unique identifier for boundaries geometries.
    boundaries_file : str, optional
        Filename of the boundaries file to use in the boundaries dataset.
    """
    input_dir = Path(workspace.files_path) / "data" / "era5" / "raw"
    output_dir = Path(workspace.files_path) / "data" / "era5" / "aggregate"

    boundaries = read_boundaries(boundaries_dataset, filename=boundaries_file)

    snt_config_dict = load_configuration_snt(
        config_path=Path(workspace.files_path) / "configuration" / "SNT_config.json"
    )

    # get country identifier for file naming
    country_code = snt_config_dict["SNT_CONFIG"].get("COUNTRY_CODE", None)
    if country_code is None:
        current_run.log_warning("COUNTRY_CODE is not specified in the configuration.")

    # subdirs containing raw data are named after variable names
    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
    variables = [d.name for d in subdirs if d.name in VARIABLES]

    if not variables:
        msg = "No variables found in input directory"
        current_run.log_error(msg)
        raise FileNotFoundError(msg)

    filename_list = []
    for variable in variables:
        current_run.log_info(f"Running aggregations for {variable}.")

        daily = get_daily(
            input_dir=input_dir / variable,
            boundaries=boundaries,
            variable=variable,
            column_uid=boundaries_column_uid,
        )

        current_run.log_info(
            f"Applied spatial aggregation to {variable} data for {len(boundaries)} boundaries"
        )

        # only apply sum aggregation for accumulated variables such as total precipitation
        sum_aggregation = variable == "total_precipitation"

        weekly = aggregate_per_week(
            daily=daily,
            column_uid="boundary_id",
            use_epidemiological_weeks=False,
            sum_aggregation=sum_aggregation,
        )

        current_run.log_info(f"Applied weekly aggregation to {variable} data ({len(weekly)} rows)")

        epi_weekly = aggregate_per_week(
            daily=daily,
            column_uid="boundary_id",
            use_epidemiological_weeks=True,
            sum_aggregation=sum_aggregation,
        )

        current_run.log_info(f"Applied epi. weekly aggregation to {variable} data ({len(epi_weekly)} rows)")

        monthly = aggregate_per_month(
            daily=daily,
            column_uid="boundary_id",
            sum_aggregation=sum_aggregation,
        )

        current_run.log_info(f"Applied monthly aggregation to {variable} data ({len(monthly)} rows)")

        dst_dir = output_dir / variable
        dst_dir.mkdir(parents=True, exist_ok=True)

        daily_fname = dst_dir / f"{country_code}_{variable}_daily.parquet"
        daily = apply_snt_formatting(daily, boundaries_column=boundaries_column_uid, aggregation="daily")
        daily.write_parquet(daily_fname)
        # current_run.add_file_output(daily_fname.as_posix())

        weekly_fname = dst_dir / f"{country_code}_{variable}_weekly.parquet"
        weekly = apply_snt_formatting(weekly, boundaries_column=boundaries_column_uid, aggregation="weekly")
        weekly.write_parquet(weekly_fname)
        # current_run.add_file_output(weekly_fname.as_posix())

        epi_weekly_fname = dst_dir / f"{country_code}_{variable}_epi_weekly.parquet"
        epi_weekly = apply_snt_formatting(
            epi_weekly, boundaries_column=boundaries_column_uid, aggregation="epi_weekly"
        )
        epi_weekly.write_parquet(epi_weekly_fname)
        # current_run.add_file_output(epi_weekly_fname.as_posix())

        monthly_fname = dst_dir / f"{country_code}_{variable}_monthly.parquet"
        monthly = apply_snt_formatting(
            monthly, boundaries_column=boundaries_column_uid, aggregation="monthly"
        )
        monthly.write_parquet(monthly_fname)
        current_run.add_file_output(monthly_fname.as_posix())

        # collect filenames for adding to dataset (only monthly for now)
        filename_list.append(
            # daily_fname,
            # weekly_fname,
            # epi_weekly_fname,
            monthly_fname,
        )

    add_files_to_dataset(
        dataset_id=snt_config_dict["SNT_DATASET_IDENTIFIERS"].get("ERA5_DATASET_CLIMATE", None),
        country_code=country_code,
        file_paths=filename_list,
    )


def read_boundaries(boundaries_dataset: Dataset, filename: str | None = None) -> gpd.GeoDataFrame:
    """Read boundaries geographic file from input dataset.

    Parameters
    ----------
    boundaries_dataset : Dataset
        Input dataset containing a "*district*.parquet" geoparquet file
    filename : str
        Filename of the boundaries file to read if there are several.
        If set to None, the 1st parquet file found will be loaded.

    Returns
    -------
    gpd.GeoDataFrame
        Geopandas GeoDataFrame containing boundaries geometries

    Raises
    ------
    FileNotFoundError
        If the boundaries file is not found
    """
    ds = boundaries_dataset.latest_version

    ds_file: DatasetFile | None = None
    for f in ds.files:
        if f.filename == filename:
            if f.filename.endswith(".parquet"):
                ds_file = f
            if f.filename.endswith(".geojson") or f.filename.endswith(".gpkg"):
                ds_file = f

    if ds_file is None:
        msg = f"File {filename} not found in dataset {ds.name}"
        current_run.log_error(msg)
        raise FileNotFoundError(msg)

    if ds_file.filename.endswith(".parquet"):
        return gpd.read_parquet(BytesIO(ds_file.read()))

    return gpd.read_file(BytesIO(ds_file.read()))


def get_daily(input_dir: Path, boundaries: gpd.GeoDataFrame, variable: str, column_uid: str) -> pl.DataFrame:
    """Aggregate daily ERA5 data for a given variable and set of boundaries.

    Parameters
    ----------
    input_dir : Path
        Directory containing GRIB files for the variable.
    boundaries : gpd.GeoDataFrame
        GeoDataFrame containing boundary geometries.
    variable : str
        ERA5 variable name.
    column_uid : str
        Column name in boundaries containing unique identifiers.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with daily aggregated values for each boundary.
    """
    # build xarray dataset by merging all available grib files across the time dimension
    with tempfile.TemporaryDirectory() as tmpdir:
        for file in input_dir.glob("*.grib"):
            # if the .grib file is actually a zip file, extract the data.grib file
            # and copy its content instead with same filename
            if zipfile.is_zipfile(file):
                with zipfile.ZipFile(file.as_posix(), "r") as zip_file:
                    bytes_read = zip_file.read("data.grib")
                with Path.open(Path(tmpdir, file.name), "wb") as f:
                    f.write(bytes_read)

            # if it's not a zip file, copy the grib file directly
            else:
                copyfile(src=file.as_posix(), dst=Path(tmpdir, file.name).as_posix())

        ds = merge(Path(tmpdir))
        ncols = len(ds.longitude)
        nrows = len(ds.latitude)
        transform = get_transform(ds)

        # build binary raster masks for each boundary geometry for spatial aggregation
        masks = build_masks(boundaries, nrows, ncols, transform)

        var = VARIABLES[variable]["shortname"]

        daily = aggregate(ds=ds, var=var, masks=masks, boundaries_id=boundaries[column_uid])

    # kelvin to celsius
    if variable == "2m_temperature":
        daily = daily.with_columns(
            [
                pl.col("mean") - 273.15,
                pl.col("min") - 273.15,
                pl.col("max") - 273.15,
            ]
        )

    # m to mm
    if variable == "total_precipitation":
        daily = daily.with_columns(
            [
                pl.col("mean") * 1000,
                pl.col("min") * 1000,
                pl.col("max") * 1000,
            ]
        )

    return daily


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
            "ERA5_DATASET_CLIMATE is not specified in the configuration."
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


def apply_snt_formatting(
    df: pd.DataFrame, boundaries_column: str, aggregation: str = "monthly"
) -> pd.DataFrame:
    """Apply SNT formatting to the aggregated ERA5 data.

    This function is a placeholder for any future formatting requirements specific to SNT.
    Currently, it does not perform any operations but can be extended as needed.

    Returns
    -------
    pd.DataFrame
        The formatted DataFrame with SNT-specific formatting applied.
    """
    aggregation_columns = {
        "daily": [boundaries_column, "DATE", "MEAN", "MIN", "MAX", "WEEK", "PERIOD", "EPI_WEEK"],
        "weekly": [boundaries_column, "WEEK", "MEAN", "MIN", "MAX"],
        "epi_weekly": [boundaries_column, "EPI_WEEK", "MEAN", "MIN", "MAX"],
        "monthly": [boundaries_column, "PERIOD", "MEAN", "MIN", "MAX"],
    }

    if aggregation in aggregation_columns:
        df.columns = aggregation_columns[aggregation]
    else:
        raise ValueError(f"Unknown aggregation type: {aggregation}")

    if aggregation == "monthly":
        df = df.with_columns(
            [
                pl.col("PERIOD").cast(pl.Utf8).str.slice(0, 4).cast(pl.Int32).alias("YEAR"),
                pl.col("PERIOD").cast(pl.Utf8).str.slice(4, 2).cast(pl.Int32).alias("MONTH"),
            ]
        )

    return df
