from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from math import ceil
from pathlib import Path

import geopandas as gpd
from openhexa.sdk import (
    CustomConnection,
    Dataset,
    current_run,
    parameter,
    pipeline,
    workspace,
)
from openhexa.sdk.datasets import DatasetFile
from openhexa.toolbox.era5.cds import CDS, VARIABLES


@pipeline("ERA5 Extract")
@parameter(
    "start_date",
    type=str,
    name="Start date",
    help="Start date of extraction period.",
    default="2018-01-01",
)
@parameter(
    "end_date",
    type=str,
    name="End date",
    help="End date of extraction period. Latest available by default.",
    required=False,
)
@parameter(
    "cds_connection",
    name="Climate data store",
    type=CustomConnection,
    help="Credentials for connection to the Copernicus Climate Data Store",
    required=True,
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
    "variable",
    name="Variable",
    type=str,
    choices=[
        "10 metre U wind component",
        "10 metre V wind component",
        "2 metre temperature",
        "Total precipitation",
        "Volumetric soil water layer 11",
    ],
    help="ERA5-Land variable of interest",
)
@parameter(
    "output_dir",
    name="Output directory",
    type=str,
    help="Output directory for the extracted data",
    required=True,
    default="data/era5/raw",
)
def era5_extract(
    start_date: str,
    end_date: str,
    cds_connection: CustomConnection,
    boundaries_dataset: Dataset,
    variable: str,
    output_dir: str,
    boundaries_file: str | None = None,
) -> None:
    """Download ERA5 products from the Climate Data Store."""
    cds = CDS(key=cds_connection.key)
    current_run.log_info("Successfully connected to the Climate Data Store")

    boundaries = read_boundaries(boundaries_dataset, filename=boundaries_file)
    bounds = get_bounds(boundaries)
    current_run.log_info(f"Using area of interest: {bounds}")

    if not end_date:
        end_date = datetime.now().astimezone(timezone.utc).strftime("%Y-%m-%d")
        current_run.log_info(f"End date set to {end_date}")

    output_dir = Path(workspace.files_path, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # find variable code and shortname from fullname provided in parameters
    var_code = None
    var_shortname = None
    for code, meta in VARIABLES.items():
        if meta["name"] == variable:
            var_code = code
            var_shortname = meta["shortname"]
            break
    if var_code is None or var_shortname is None:
        msg = f"Variable {variable} not supported"
        current_run.log_error(msg)
        raise ValueError(msg)

    # default hours to download depending on climate variable
    time = {
        "2m_temperature": [0, 6, 12, 18],
        "total_precipitation": [23],
        "volumetric_soil_water_layer_1": [0, 6, 12, 18],
    }

    download(
        client=cds,
        variable=var_code,
        start=start_date,
        end=end_date,
        output_dir=output_dir,
        area=bounds,
        time=time.get(var_code, [0, 6, 12, 18]),
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

    Raises
    ------
    FileNotFoundError
        If the boundaries file is not found

    Returns
    -------
    gpd.GeoDataFrame
        Geopandas GeoDataFrame containing boundaries geometries
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


def get_bounds(boundaries: gpd.GeoDataFrame) -> tuple[int]:
    """Extract bounding box coordinates of the input geodataframe.

    Parameters
    ----------
    boundaries : gpd.GeoDataFrame
        Geopandas GeoDataFrame containing boundaries geometries

    Returns
    -------
    tuple[int]
        Bounding box coordinates in the order (ymax, xmin, ymin, xmax)
    """
    xmin, ymin, xmax, ymax = boundaries.total_bounds
    xmin = ceil(xmin - 0.5)
    ymin = ceil(ymin - 0.5)
    xmax = ceil(xmax + 0.5)
    ymax = ceil(ymax + 0.5)
    return ymax, xmin, ymin, xmax


def download(
    client: CDS,
    variable: str,
    start: str,
    end: str,
    output_dir: Path,
    area: tuple[float],
    time: list[int] | None = None,
) -> None:
    """Download ERA5 products from the Climate Data Store.

    Parameters
    ----------
    client : CDS
        CDS client object
    variable : str
        ERA5 product variable (ex: "2m_temperature", "total_precipitation")
    start : str
        Start date of extraction period (YYYY-MM-DD)
    end : str
        End date of extraction period (YYYY-MM-DD)
    output_dir : Path
        Output directory for the extracted data (a subfolder named after the variable will be
        created)
    area : tuple[float]
        Bounding box coordinates in the order (ymax, xmin, ymin, xmax)
    time : list[int] | None, optional
        Hours of interest as integers (between 0 and 23). Set to all hours if None.

    Raise
    -----
    ValueError
        If the variable is not supported
    """
    if variable not in VARIABLES:
        msg = f"Variable {variable} not supported"
        current_run.log_error(msg)
        raise ValueError(msg)

    start = datetime.strptime(start, "%Y-%m-%d").astimezone(timezone.utc)
    end = datetime.strptime(end, "%Y-%m-%d").astimezone(timezone.utc)

    dst_dir = output_dir / variable
    dst_dir.mkdir(parents=True, exist_ok=True)

    client.download_between(variable=variable, start=start, end=end, dst_dir=dst_dir, area=area, time=time)

    current_run.log_info(f"Downloaded raw data for variable `{variable}`")
