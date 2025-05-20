import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from shutil import copyfile

import geopandas as gpd
import polars as pl
from openhexa.sdk import Dataset, current_run, parameter, pipeline, workspace
from openhexa.sdk.datasets import DatasetFile
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
    "input_dir",
    type=str,
    name="Input directory",
    help="Input directory with raw ERA5 extracts",
    default="data/era5/raw",
)
@parameter(
    "output_dir",
    type=str,
    name="Output directory",
    help="Output directory for the aggregated data",
    default="data/era5/aggregate",
)
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
    input_dir: str,
    output_dir: str,
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
    input_dir = Path(workspace.files_path, input_dir)
    output_dir = Path(workspace.files_path, output_dir)

    boundaries = read_boundaries(boundaries_dataset, filename=boundaries_file)

    # subdirs containing raw data are named after variable names
    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
    variables = [d.name for d in subdirs if d.name in VARIABLES]

    if not variables:
        msg = "No variables found in input directory"
        current_run.log_error(msg)
        raise FileNotFoundError(msg)

    for variable in variables:
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

        daily.write_parquet(dst_dir / f"{variable}_daily.parquet")
        current_run.add_file_output(Path(dst_dir, f"{variable}_daily.parquet").as_posix())

        weekly.write_parquet(dst_dir / f"{variable}_weekly.parquet")
        current_run.add_file_output(Path(dst_dir, f"{variable}_weekly.parquet").as_posix())

        epi_weekly.write_parquet(dst_dir / f"{variable}_epi_weekly.parquet")
        current_run.add_file_output(Path(dst_dir, f"{variable}_epi_weekly.parquet").as_posix())

        monthly.write_parquet(dst_dir / f"{variable}_monthly.parquet")
        current_run.add_file_output(Path(dst_dir, f"{variable}_monthly.parquet").as_posix())


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
