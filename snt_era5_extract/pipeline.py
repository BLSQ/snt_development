from __future__ import annotations

import shutil
import time
from datetime import date, datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta
from io import BytesIO
from math import ceil
from pathlib import Path

import geopandas as gpd
from ecmwf.datastores.client import Client
from openhexa.sdk import (
    CustomConnection,
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
from openhexa.toolbox.era5.extract import prepare_requests, retrieve_requests, grib_to_zarr
from openhexa.toolbox.era5.utils import get_variables

CDS_API_URL = "https://cds.climate.copernicus.eu/api"
DATASET_ID = "reanalysis-era5-land"


def _safe_cleanup_era5_store(zarr_store: Path, dst_dir: Path) -> None:
    """Remove Zarr store and all content under dst_dir, then recreate dst_dir.

    Used when duplicate time values are detected so we can retry with a full download.
    Logs actions and catches errors so cleanup failures do not hide the original error.
    """
    if zarr_store.exists():
        try:
            shutil.rmtree(zarr_store)
            current_run.log_info(f"Deleted Zarr store: {zarr_store}")
        except Exception as exc:
            current_run.log_warning(f"Failed to delete Zarr store {zarr_store}: {exc}")

    removed_count = 0
    if dst_dir.exists():
        for entry in dst_dir.iterdir():
            try:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                removed_count += 1
            except Exception as exc:
                current_run.log_warning(f"Could not remove {entry}: {exc}")

    if removed_count:
        current_run.log_info(f"Removed {removed_count} item(s) under {dst_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    time.sleep(1)


@pipeline("snt_era5_extract")
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
    """Download ERA5 products from the Climate Data Store (ecmwf-datastores-client)."""
    root_path = Path(workspace.files_path)

    client = Client(key=cds_connection.key, url=CDS_API_URL)
    current_run.log_info("Successfully connected to the Climate Data Store")

    variable = "total_precipitation"
    current_run.log_info(f"Downloading ERA5 data for variable: {variable}")

    try:
        if not is_valid_ymd(start_date):
            raise ValueError(f"Invalid start date format: {start_date}. Expected format: YYYY-MM-DD")
        if not is_valid_ymd(end_date):
            raise ValueError(f"Invalid end date format: {end_date}. Expected format: YYYY-MM-DD")

        variables = get_variables()
        if variable not in variables:
            raise ValueError(
                f"Variable {variable} not supported. Available: {list(variables.keys())}"
            )

        snt_config_dict = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
        validate_config(snt_config_dict)
        dhis2_formatted_dataset = snt_config_dict["SNT_DATASET_IDENTIFIERS"].get("DHIS2_DATASET_FORMATTED")
        country_code = snt_config_dict["SNT_CONFIG"].get("COUNTRY_CODE")

        boundaries = read_boundaries(dhis2_formatted_dataset, filename=f"{country_code}_shapes.geojson")
        bounds = get_bounds(boundaries)
        area = [int(b) for b in bounds]
        current_run.log_info(f"Using area of interest: {area}")

        if start_date:
            start_date = start_date[0:8] + "01"
        current_run.log_info(f"Start date set to {start_date}")

        if not end_date:
            end_date = (
                datetime.now().astimezone(timezone.utc).replace(day=1) - relativedelta(days=1)  # noqa: UP017
            ).strftime("%Y-%m-%d")
            current_run.log_info(f"End date set to last day of previous month {end_date}")
        else:
            end_date = to_last_day_previous_month(end_date)
            current_run.log_info(f"End date set to last day of month {end_date}")

        output_dir = Path(workspace.files_path) / "data" / "era5" / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)
        dst_dir = output_dir / variable
        dst_dir.mkdir(parents=True, exist_ok=True)
        zarr_store = dst_dir / "data.zarr"
        start_d = date.fromisoformat(start_date)
        end_d = date.fromisoformat(end_date)
        data_var = variables[variable]["short_name"]

        max_attempts = 3
        for attempt in range(max_attempts):
            requests = prepare_requests(
                client=client,
                dataset_id=DATASET_ID,
                start_date=start_d,
                end_date=end_d,
                variable=variable,
                area=area,
                zarr_store=zarr_store,
            )

            if not requests:
                current_run.log_info("No missing dates to download; data already up to date.")
                break

            current_run.log_info(f"Submitting {len(requests)} request(s) to CDS.")
            retrieve_requests(
                client=client,
                dataset_id=DATASET_ID,
                requests=requests,
                dst_dir=dst_dir,
                wait=30,
            )
            try:
                grib_to_zarr(src_dir=dst_dir, zarr_store=zarr_store, data_var=data_var)
            except RuntimeError as e:
                if "Duplicate time values" not in str(e):
                    raise
                if attempt < max_attempts - 1:
                    current_run.log_warning(
                        "Duplicate time values in store detected. Performing safe cleanup and retry."
                    )
                    _safe_cleanup_era5_store(zarr_store, dst_dir)
                    continue
                # Last attempt failed: raise with clear message and state for debugging
                dst_files = list(dst_dir.iterdir()) if dst_dir.exists() else []
                zarr_exists = zarr_store.exists()
                raise RuntimeError(
                    f"Duplicate time values still present after {max_attempts} attempt(s). "
                    f"Zarr exists={zarr_exists}, files in dst_dir={len(dst_files)}. "
                    f"Delete data/era5/raw/{variable}/data.zarr and re-run."
                ) from e
            current_run.log_info(f"Downloaded and stored raw data for variable `{variable}`")
            break
    except Exception as e:
        current_run.log_error(f"An error occurred during the ERA5 extraction: {e}")
        raise


def read_boundaries(boundaries_id: str, filename: str | None = None) -> gpd.GeoDataFrame:
    """Read boundaries geographic file from input dataset.

    Parameters
    ----------
    boundaries_id : str
        Input dataset id containing a SNT shapes file
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


def is_valid_ymd(date_str: str) -> bool:
    """Check if a date string is in the format YYYY-MM-DD.

    Parameters
    ----------
    date_str : str
        Date string to validate.

    Returns
    -------
    bool
        True if the date string is valid or empty, False otherwise.
    """
    if date_str:
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    return True  # If date_str is empty, consider it valid


def to_last_day_previous_month(date_str: str) -> str:
    """Return the last day of the previous month for a given date string.

    Parameters
    ----------
    date_str : str
        Date string in the format YYYY-MM-DD.

    Returns
    -------
    str
        Date string representing the last day of the previous month.
    """
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
