from __future__ import annotations

import shutil
import tempfile
from datetime import date, datetime, timedelta
from io import BytesIO
from math import ceil, floor
from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import xarray as xr
from openhexa.sdk import current_run, workspace
from openhexa.sdk.datasets import DatasetFile
from openhexa.toolbox.era5.cache import Cache
from openhexa.toolbox.era5.dhis2weeks import WeekType, to_dhis2_week
from openhexa.toolbox.era5.extract import Client, grib_to_zarr, prepare_requests, retrieve_requests
from openhexa.toolbox.era5.transform import Period, aggregate_in_space, aggregate_in_time, create_masks
from openhexa.toolbox.era5.utils import get_variables

CDS_API_URL = "https://cds.climate.copernicus.eu/api"
DATASET_ID = "reanalysis-era5-land"
ERA5_VARIABLES = ["total_precipitation"]


def read_boundaries(boundaries_id: str, filename: str) -> gpd.GeoDataFrame:
    boundaries_dataset = workspace.get_dataset(boundaries_id)
    dataset_version = boundaries_dataset.latest_version
    if not dataset_version:
        raise FileNotFoundError(f"Dataset {boundaries_id} has no versions available.")

    dataset_file: DatasetFile | None = None
    for file in dataset_version.files:
        if file.filename == filename and (
            file.filename.endswith(".parquet")
            or file.filename.endswith(".geojson")
            or file.filename.endswith(".gpkg")
        ):
            dataset_file = file

    if dataset_file is None:
        raise FileNotFoundError(f"File {filename} not found in dataset {dataset_version.name}")

    if dataset_file.filename.endswith(".parquet"):
        return gpd.read_parquet(BytesIO(dataset_file.read()))
    return gpd.read_file(BytesIO(dataset_file.read()))


def get_bounds(boundaries: gpd.GeoDataFrame) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = boundaries.total_bounds
    return (
        ceil(ymax + 0.1),
        floor(xmin - 0.1),
        floor(ymin - 0.1),
        ceil(xmax + 0.1),
    )


def is_valid_ymd(date_str: str | None) -> bool:
    if not date_str:
        return True
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def to_last_day_previous_month(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    today = datetime.today()
    next_month_first_day = (dt.replace(day=28) + timedelta(days=4)).replace(day=1)
    last_day_of_month = next_month_first_day - timedelta(days=1)
    if dt == last_day_of_month and dt < today:
        return date_str
    first_day_of_month = dt.replace(day=1)
    return (first_day_of_month - timedelta(days=1)).strftime("%Y-%m-%d")


def get_cds_api_key(cds_connection) -> str:
    key = getattr(cds_connection, "api_key", None) or getattr(cds_connection, "key", None)
    if not key:
        raise RuntimeError("CDS custom connection is missing `api_key` (or `key`).")
    return key


def sync_variable(
    client: Client,
    cache: Cache,
    variable: str,
    start_d: date,
    end_d: date,
    area: list[int],
    raw_dir: Path,
) -> None:
    zarr_store = raw_dir / f"{variable}.zarr"
    if zarr_store.exists():
        current_run.log_warning(f"[{variable}] Removing existing zarr store to force full resync: {zarr_store}")
        shutil.rmtree(zarr_store)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
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
            current_run.log_info(f"[{variable}] No missing dates to download.")
            return

        current_run.log_info(f"[{variable}] Prepared {len(requests)} request(s).")
        retrieve_requests(
            client=client,
            dataset_id=DATASET_ID,
            requests=requests,
            dst_dir=tmp_path,
            cache=cache,
            wait=30,
        )
        short_name = get_variables()[variable]["short_name"]
        grib_to_zarr(src_dir=tmp_path, zarr_store=zarr_store, data_var=short_name)
        current_run.log_info(f"[{variable}] Sync completed.")


def validate_synced_zarr(zarr_store: Path, variable: str) -> None:
    if not zarr_store.exists():
        raise FileNotFoundError(f"[{variable}] Missing zarr store: {zarr_store}")

    ds = xr.open_zarr(zarr_store, consolidated=True, decode_timedelta=False)
    data_var = get_variables()[variable]["short_name"]
    if data_var not in ds.data_vars:
        raise ValueError(f"[{variable}] Variable `{data_var}` not found in zarr store.")

    da = ds[data_var]
    if "time" not in da.dims or da.sizes.get("time", 0) == 0:
        raise ValueError(f"[{variable}] Zarr store has no time values.")

    sample = da.isel(time=slice(0, min(48, da.sizes["time"]))).load()
    non_null_count = int(sample.notnull().sum().item())
    total_count = int(sample.size)
    current_run.log_info(f"[{variable}] Non-null sample points: {non_null_count}/{total_count}")
    if non_null_count == 0:
        raise ValueError(
            f"[{variable}] Synced zarr appears empty (all sampled values are null). "
            "Aborting to avoid publishing invalid aggregates."
        )


def build_daily_snt(zarr_store: Path, boundaries: gpd.GeoDataFrame, variable: str, column_uid: str) -> pl.DataFrame:
    if boundaries.crs and boundaries.crs.to_string() != "EPSG:4326":
        boundaries = boundaries.to_crs("EPSG:4326")

    valid_boundaries = boundaries[boundaries.geometry.notna() & ~boundaries.geometry.is_empty].copy()
    if len(valid_boundaries) == 0:
        raise ValueError("No valid geometries found in boundaries.")

    ds = xr.open_zarr(zarr_store, consolidated=True, decode_timedelta=False)
    masks = create_masks(gdf=valid_boundaries, id_column=column_uid, ds=ds)

    pixel_count = np.asarray(masks.values).reshape(masks.shape[0], -1).sum(axis=1)
    non_empty_mask = pixel_count > 0
    if not np.all(non_empty_mask):
        valid_boundaries = valid_boundaries.loc[non_empty_mask].copy()
        masks = masks.isel(boundary=np.where(non_empty_mask)[0])
        current_run.log_warning(f"Skipping {int((~non_empty_mask).sum())} boundaries with no ERA5 pixel overlap.")
    if len(valid_boundaries) == 0:
        raise ValueError("No boundaries overlap ERA5 pixels.")

    meta = get_variables()[variable]
    data_var = meta["short_name"]
    is_accumulated = bool(meta["accumulated"])

    if is_accumulated:
        daily_mean = aggregate_in_time(
            dataframe=aggregate_in_space(ds=ds, masks=masks, data_var=data_var, agg="mean"),
            period=Period.DAY,
            agg="sum",
        ).rename({"value": "mean"})
        daily_min = aggregate_in_time(
            dataframe=aggregate_in_space(ds=ds, masks=masks, data_var=data_var, agg="min"),
            period=Period.DAY,
            agg="sum",
        ).rename({"value": "min"})
        daily_max = aggregate_in_time(
            dataframe=aggregate_in_space(ds=ds, masks=masks, data_var=data_var, agg="max"),
            period=Period.DAY,
            agg="sum",
        ).rename({"value": "max"})
    else:
        ds_mean = ds.resample(time="1D").mean()
        ds_min = ds.resample(time="1D").min()
        ds_max = ds.resample(time="1D").max()
        daily_mean = aggregate_in_time(
            dataframe=aggregate_in_space(ds=ds_mean, masks=masks, data_var=data_var, agg="mean"),
            period=Period.DAY,
            agg="mean",
        ).rename({"value": "mean"})
        daily_min = aggregate_in_time(
            dataframe=aggregate_in_space(ds=ds_min, masks=masks, data_var=data_var, agg="mean"),
            period=Period.DAY,
            agg="mean",
        ).rename({"value": "min"})
        daily_max = aggregate_in_time(
            dataframe=aggregate_in_space(ds=ds_max, masks=masks, data_var=data_var, agg="mean"),
            period=Period.DAY,
            agg="mean",
        ).rename({"value": "max"})

    daily = (
        daily_mean.join(daily_min, on=["boundary", "period"], how="inner")
        .join(daily_max, on=["boundary", "period"], how="inner")
        .with_columns(
            [
                pl.col("boundary").alias("boundary_id"),
                pl.col("period").str.strptime(pl.Date, "%Y%m%d").alias("date"),
            ]
        )
        .with_columns(
            [
                pl.col("date")
                .map_elements(lambda d: to_dhis2_week(d, WeekType.WEEK), return_dtype=pl.String)
                .alias("week"),
                pl.col("date").dt.strftime("%Y%m").alias("period_month"),
                pl.col("date")
                .map_elements(lambda d: to_dhis2_week(d, WeekType.WEEK_THURSDAY), return_dtype=pl.String)
                .alias("epi_week"),
            ]
        )
    )
    daily = daily.filter(pl.col("mean").is_not_null() & pl.col("date").is_not_null())

    if variable == "2m_temperature":
        daily = daily.with_columns([pl.col("mean") - 273.15, pl.col("min") - 273.15, pl.col("max") - 273.15])
    if variable == "total_precipitation":
        daily = daily.with_columns([pl.col("mean") * 1000, pl.col("min") * 1000, pl.col("max") * 1000])

    daily = daily.select(["boundary_id", "date", "mean", "min", "max", "week", "period_month", "epi_week"])
    if daily.is_empty():
        raise ValueError(f"[{variable}] Daily aggregation produced no rows.")
    if daily.filter(pl.col("mean").is_null()).height > 0:
        raise ValueError(f"[{variable}] Daily aggregation still contains null `mean` values.")

    return daily


def aggregate_daily_snt(daily: pl.DataFrame, key_col: str, sum_aggregation: bool) -> pl.DataFrame:
    agg = pl.sum if sum_aggregation else pl.mean
    out = (
        daily.group_by(["boundary_id", key_col])
        .agg(
            [
                agg("mean").alias("mean"),
                agg("min").alias("min"),
                agg("max").alias("max"),
            ]
        )
        .sort(["boundary_id", key_col])
    )
    if out.is_empty():
        raise ValueError(f"Aggregation by `{key_col}` produced no rows.")
    return out


def apply_snt_formatting(df: pl.DataFrame, aggregation: str) -> pl.DataFrame:
    aggregation_columns = {
        "daily": ["ADM2_ID", "DATE", "MEAN", "MIN", "MAX", "WEEK", "PERIOD", "EPI_WEEK"],
        "weekly": ["ADM2_ID", "WEEK", "MEAN", "MIN", "MAX"],
        "epi_weekly": ["ADM2_ID", "EPI_WEEK", "MEAN", "MIN", "MAX"],
        "monthly": ["ADM2_ID", "PERIOD", "MEAN", "MIN", "MAX"],
    }
    if aggregation not in aggregation_columns:
        raise ValueError(f"Unknown aggregation type: {aggregation}")

    df.columns = aggregation_columns[aggregation]
    if aggregation == "monthly":
        df = df.with_columns(
            [
                pl.col("PERIOD").cast(pl.Utf8).str.slice(0, 4).cast(pl.Int32).alias("YEAR"),
                pl.col("PERIOD").cast(pl.Utf8).str.slice(4, 2).cast(pl.Int32).alias("MONTH"),
            ]
        )
    return df
