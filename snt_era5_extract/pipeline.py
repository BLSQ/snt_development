from __future__ import annotations

from datetime import datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta
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
from snt_lib.snt_pipeline_utils import (    
    load_configuration_snt,    
    validate_config,
)
from openhexa.sdk.datasets import DatasetFile
from openhexa.toolbox.era5.cds import CDS, VARIABLES


@pipeline("SNT ERA5 Extract")
@parameter(
    "start_date",
    type=str,
    name="Start date",
    help="Start date of extraction period.",
    default="2018-01-01",
    required=True,
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
def era5_extract(
    start_date: str,
    end_date: str,
    cds_connection: CustomConnection,    
) -> None:
    """Download ERA5 products from the Climate Data Store."""
    root_path = Path(workspace.files_path)

    cds = CDS(key=cds_connection.key)
    current_run.log_info("Successfully connected to the Climate Data Store")
    variable = "Total precipitation" 
    current_run.log_info(f"Downloading ERA5 data for variable: {variable}")

    try:
        if not is_valid_ymd(start_date):
            raise ValueError(f"Invalid start date format: {start_date}. Expected format: YYYY-MM-DD")
        if not is_valid_ymd(end_date):
            raise ValueError(f"Invalid end date format: {end_date}. Expected format: YYYY-MM-DD")

        snt_config_dict = load_configuration_snt(
            config_path=root_path / "configuration" / "SNT_config.json"
            )
        validate_config(snt_config_dict)        
        dhis2_formatted_dataset = snt_config_dict["SNT_DATASET_IDENTIFIERS"].get("DHIS2_DATASET_FORMATTED")
        country_code = snt_config_dict["SNT_CONFIG"].get("COUNTRY_CODE")       

        # get boundaries geometries from formatted dataset
        boundaries = read_boundaries(dhis2_formatted_dataset, filename=f"{country_code}_shapes.geojson") 
        bounds = get_bounds(boundaries)
        current_run.log_info(f"Using area of interest: {bounds}")

        if start_date:
            # Ensure start date is the first of the month
            start_date = start_date[0:8] + "01"  
        current_run.log_info(f"Start date set to {start_date}")

        if not end_date:
            # end_date = datetime.now().astimezone(timezone.utc).strftime("%Y-%m-%d")
            end_date = (datetime.now().astimezone(timezone.utc).replace(day=1) - relativedelta(days=1)).strftime(
                "%Y-%m-%d"
            )
            current_run.log_info(f"End date set to last day of previous month {end_date}")
        else:
            # Push the end date to the last day of the previous month
            end_date = to_last_day_previous_month(end_date)
            current_run.log_info(f"End date set to last day of month {end_date}")        

        output_dir = Path(workspace.files_path) / "data" / "era5" / "raw"
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
            raise ValueError(f"Variable {variable} not supported")

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
    except Exception as e:
        current_run.log_error(f"An error occurred during the ERA5 extraction: {e}")
        raise


def read_boundaries(boundaries_id: str, filename: str | None = None) -> gpd.GeoDataFrame:
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
    try:
        boundaries_dataset = workspace.get_dataset(boundaries_id)
    except Exception as e:
        raise Exception(f"Dataset {boundaries_id} not found.") from e
    
    ds = boundaries_dataset.latest_version
    if not ds:
        raise FileNotFoundError(f"Dataset {boundaries_id} has no versions available.")

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


def is_valid_ymd(date_str: str) -> bool:
    if date_str:
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    return True # If date_str is empty, consider it valid
    

def to_last_day_previous_month(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    today = datetime.today()

    # Compute the last day of the month for the given date
    next_month_first_day = (dt.replace(day=28) + timedelta(days=4)).replace(day=1)
    last_day_of_month = next_month_first_day - timedelta(days=1)

    # If end_date is the last day of its month and is before today, return it unchanged
    if dt == last_day_of_month and dt < today:
        return date_str

    # Otherwise, return the last day of the previous month for the given date
    first_day_of_month = dt.replace(day=1)
    last_day_prev_month = first_day_of_month - timedelta(days=1)
    return last_day_prev_month.strftime("%Y-%m-%d")
  