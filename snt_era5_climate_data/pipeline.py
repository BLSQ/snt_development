from __future__ import annotations

import shutil
import tempfile
from datetime import date, datetime, timedelta, timezone
from math import ceil, floor
from pathlib import Path

import geopandas as gpd
import polars as pl
import xarray as xr
import numpy as np
from dateutil.relativedelta import relativedelta
from openhexa.sdk import CustomConnection, current_run, parameter, pipeline, workspace
from openhexa.toolbox.era5.cache import Cache
from openhexa.toolbox.era5.dhis2weeks import WeekType, to_dhis2_week
from openhexa.toolbox.era5.extract import Client, grib_to_zarr, prepare_requests, retrieve_requests
from openhexa.toolbox.era5.transform import Period, aggregate_in_space, aggregate_in_time, create_masks
from openhexa.toolbox.era5.utils import get_variables
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    get_file_from_dataset,
    load_configuration_snt,
    pull_scripts_from_repository,
    run_report_notebook,
    save_pipeline_parameters,
    validate_config,
)

CDS_API_URL = "https://cds.climate.copernicus.eu/api"
DATASET_ID = "reanalysis-era5-land"
ERA5_VARIABLES = ["total_precipitation"]
# ERA5_VARIABLES = ["2m_dewpoint_temperature", "2m_temperature", "total_precipitation"]


def get_bounds(boundaries: gpd.GeoDataFrame) -> tuple[int, int, int, int]:
    """Compute CDS request bounds (N, W, S, E) from boundaries extent.

    Returns
    -------
    tuple[int, int, int, int]
        Bounding coordinates formatted for CDS requests.
    """
    xmin, ymin, xmax, ymax = boundaries.total_bounds
    xmin = floor(xmin - 0.1)
    ymin = floor(ymin - 0.1)
    xmax = ceil(xmax + 0.1)
    ymax = ceil(ymax + 0.1)
    return ymax, xmin, ymin, xmax


def is_valid_ymd(date_str: str | None) -> bool:
    """Return True when `date_str` follows YYYY-MM-DD format.

    Returns
    -------
    bool
        True if the date string is valid, otherwise False.
    """
    if not date_str:
        return True
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def to_last_day_previous_month(date_str: str) -> str:
    """Normalize a date to the last day of the previous month.

    Returns
    -------
    str
        Adjusted date in YYYY-MM-DD format.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    today = datetime.today()
    next_month_first_day = (dt.replace(day=28) + timedelta(days=4)).replace(day=1)
    last_day_of_month = next_month_first_day - timedelta(days=1)
    if dt == last_day_of_month and dt < today:
        return date_str
    first_day_of_month = dt.replace(day=1)
    last_day_prev_month = first_day_of_month - timedelta(days=1)
    return last_day_prev_month.strftime("%Y-%m-%d")


def get_cds_api_key(cds_connection: CustomConnection) -> str:
    """Extract CDS API key from the OpenHEXA custom connection.

    Returns
    -------
    str
        CDS API key string.
    """
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
    """Download and store one ERA5 variable into a zarr store."""
    zarr_store = raw_dir / f"{variable}.zarr"
    # Force a clean rebuild for deterministic outputs and avoid stale/corrupted zarr reuse.
    if zarr_store.exists():
        current_run.log_warning(
            f"[{variable}] Removing existing zarr store "
            f"to force full resync: {zarr_store}"
        )
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
        grib_to_zarr(
            src_dir=tmp_path,
            zarr_store=zarr_store,
            data_var=get_variables()[variable]["short_name"],
        )
        current_run.log_info(f"[{variable}] Sync completed.")


def validate_synced_zarr(zarr_store: Path, variable: str) -> None:
    """Validate that synced zarr contains non-null values."""
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


def build_daily_snt(
    zarr_store: Path,
    boundaries: gpd.GeoDataFrame,
    variable: str,
    column_uid: str,
) -> pl.DataFrame:
    """Build daily SNT-ready ERA5 aggregates for a variable.

    Returns
    -------
    pl.DataFrame
        Daily values with boundary id and derived period columns.
    """
    if boundaries.crs and boundaries.crs.to_string() != "EPSG:4326":
        boundaries = boundaries.to_crs("EPSG:4326")

    valid_boundaries = boundaries[boundaries.geometry.notna() & ~boundaries.geometry.is_empty].copy()
    if len(valid_boundaries) == 0:
        raise ValueError("No valid geometries found in boundaries.")

    ds = xr.open_zarr(zarr_store, consolidated=True, decode_timedelta=False)
    masks = create_masks(gdf=valid_boundaries, id_column=column_uid, ds=ds)
    # Keep only boundaries overlapping at least one ERA5 pixel
    pixel_count = np.asarray(masks.values).reshape(masks.shape[0], -1).sum(axis=1)
    non_empty_mask = pixel_count > 0
    if not np.all(non_empty_mask):
        valid_boundaries = valid_boundaries.loc[non_empty_mask].copy()
        masks = masks.isel(boundary=np.where(non_empty_mask)[0])
        skipped = int((~non_empty_mask).sum())
        current_run.log_warning(
            f"Skipping {skipped} boundaries with no ERA5 pixel overlap."
        )
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
    # Avoid turning missing values into artificial zeros in monthly sums.
    daily = daily.filter(pl.col("mean").is_not_null() & pl.col("date").is_not_null())

    if variable == "2m_temperature":
        daily = daily.with_columns([pl.col("mean") - 273.15, pl.col("min") - 273.15, pl.col("max") - 273.15])
    if variable == "total_precipitation":
        daily = daily.with_columns([pl.col("mean") * 1000, pl.col("min") * 1000, pl.col("max") * 1000])

    daily = daily.select(["boundary_id", "date", "mean", "min", "max", "week", "period_month", "epi_week"])
    if daily.is_empty():
        raise ValueError(f"[{variable}] Daily aggregation produced no rows.")

    null_mean_rows = daily.filter(pl.col("mean").is_null()).height
    if null_mean_rows > 0:
        current_run.log_warning(
            f"[{variable}] Daily output still contains {null_mean_rows} "
            "null `mean` rows; dropping them."
        )
        daily = daily.filter(pl.col("mean").is_not_null())

    if daily.is_empty():
        raise ValueError(f"[{variable}] Daily aggregation contains only null `mean` values.")

    return daily


def aggregate_daily_snt(
    daily: pl.DataFrame,
    key_col: str,
    sum_aggregation: bool,
) -> pl.DataFrame:
    """Aggregate daily data by time key using mean or sum semantics.

    Returns
    -------
    pl.DataFrame
        Aggregated values keyed by boundary and selected period column.
    """
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
    """Rename columns to the SNT output schema by aggregation level.

    Returns
    -------
    pl.DataFrame
        Formatted dataframe with expected SNT column names.
    """
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


@pipeline("snt_era5_climate_data")
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
def snt_era5_climate_data(
    start_date: str,
    end_date: str | None,
    cds_connection: CustomConnection,
    pull_scripts: bool,
    run_report_only: bool,
) -> None:
    """Unified ERA5 pipeline for SNT: synchronize raw ERA5 then aggregate to SNT outputs."""
    root_path = Path(workspace.files_path)
    raw_dir = root_path / "data" / "era5" / "raw"
    cache_dir = root_path / "data" / "era5" / "cache"
    output_dir = root_path / "data" / "era5" / "aggregate"
    report_nb = root_path / "pipelines" / "snt_era5_climate_data" / "reporting" / "snt_era5_climate_data_report.ipynb"
    report_out = root_path / "pipelines" / "snt_era5_climate_data" / "reporting" / "outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if pull_scripts:
        current_run.log_info("Pulling ERA5 scripts from repository.")
        try:
            pull_scripts_from_repository(
                pipeline_name="snt_era5_climate_data",
                report_scripts=["snt_era5_climate_data_report.ipynb"],
                code_scripts=[],
            )
        except Exception as e:
            current_run.log_warning(f"Could not pull snt_era5_climate_data scripts: {e}")

    snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
    validate_config(snt_config)
    country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]
    if not run_report_only:
        if not is_valid_ymd(start_date):
            raise ValueError(f"Invalid start date format: {start_date}. Expected YYYY-MM-DD.")
        if not is_valid_ymd(end_date):
            raise ValueError(f"Invalid end date format: {end_date}. Expected YYYY-MM-DD.")

        variables_to_run = ERA5_VARIABLES
        current_run.log_info(f"Variables to process: {variables_to_run}")

        dhis2_formatted_dataset = snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_DATASET_FORMATTED"]
        era5_dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"]["ERA5_DATASET_CLIMATE"]
        boundaries = get_file_from_dataset(
            dataset_id=dhis2_formatted_dataset,
            filename=f"{country_code}_shapes.geojson",
        )
        if not isinstance(boundaries, gpd.GeoDataFrame):
            raise TypeError(
                f"Expected GeoDataFrame for boundaries, got {type(boundaries).__name__} "
                f"from dataset {dhis2_formatted_dataset}."
            )
        area = [int(v) for v in get_bounds(boundaries)]

        if start_date:
            start_date = start_date[0:8] + "01"
        if not end_date:
            end_date = (
                datetime.now().astimezone(timezone.utc).replace(day=1) - relativedelta(days=1)  # noqa: UP017
            ).strftime("%Y-%m-%d")
        else:
            end_date = to_last_day_previous_month(end_date)

        start_d = date.fromisoformat(start_date)
        end_d = date.fromisoformat(end_date)
        current_run.log_info(f"Sync period: {start_d} to {end_d}")

        cds_key = get_cds_api_key(cds_connection=cds_connection)
        client = Client(url=CDS_API_URL, key=cds_key, retry_after=30)
        cache = Cache(database_uri=workspace.database_url, cache_dir=cache_dir)
        current_run.log_info(f"ERA5 cache directory: {cache_dir}")
        get_variables()  # fail fast if toolbox era5 API is not available as expected

        # 1) synchronize raw ERA5
        for variable in variables_to_run:
            sync_variable(
                client=client,
                cache=cache,
                variable=variable,
                start_d=start_d,
                end_d=end_d,
                area=area,
                raw_dir=raw_dir,
            )
            validate_synced_zarr(zarr_store=raw_dir / f"{variable}.zarr", variable=variable)

        # 2) aggregate and export with same outputs as snt_era5_aggregate
        file_paths_to_upload: list[Path] = []
        for variable in variables_to_run:
            current_run.log_info(f"Running SNT aggregation for {variable}")
            daily = build_daily_snt(
                zarr_store=raw_dir / f"{variable}.zarr",
                boundaries=boundaries,
                variable=variable,
                column_uid="ADM2_ID",
            )
            sum_aggregation = variable == "total_precipitation"
            weekly = aggregate_daily_snt(daily=daily, key_col="week", sum_aggregation=sum_aggregation)
            epi_weekly = aggregate_daily_snt(daily=daily, key_col="epi_week", sum_aggregation=sum_aggregation)
            monthly = aggregate_daily_snt(daily=daily, key_col="period_month", sum_aggregation=sum_aggregation)
            if monthly.filter(pl.col("mean").is_null()).height > 0:
                raise ValueError(f"[{variable}] Monthly output contains null `mean` values.")

            dst_dir = output_dir / variable
            dst_dir.mkdir(parents=True, exist_ok=True)

            daily_fname = dst_dir / f"{country_code}_{variable}_daily.parquet"
            weekly_fname = dst_dir / f"{country_code}_{variable}_weekly.parquet"
            epi_weekly_fname = dst_dir / f"{country_code}_{variable}_epi_weekly.parquet"
            monthly_fname = dst_dir / f"{country_code}_{variable}_monthly.parquet"

            apply_snt_formatting(daily, "daily").write_parquet(daily_fname)
            apply_snt_formatting(weekly, "weekly").write_parquet(weekly_fname)
            apply_snt_formatting(epi_weekly, "epi_weekly").write_parquet(epi_weekly_fname)
            apply_snt_formatting(monthly, "monthly").write_parquet(monthly_fname)

            # Keep upload behavior identical to existing aggregate pipeline (monthly only)
            file_paths_to_upload.append(monthly_fname)

        params_file = save_pipeline_parameters(
            pipeline_name="snt_era5_aggregate",
            parameters={
                "run_report_only": run_report_only,
                "start_date": start_date,
                "end_date": end_date,
                "variables": variables_to_run,
                "pull_scripts": pull_scripts,
            },
            output_path=output_dir,
            country_code=country_code,
        )
        file_paths_to_upload.append(params_file)

        add_files_to_dataset(
            dataset_id=era5_dataset_id,
            country_code=country_code,
            file_paths=file_paths_to_upload,
        )
    else:
        current_run.log_info("run_report_only=True: skipping ERA5 climate-data processing and dataset publication.")

    run_report_notebook(
        nb_file=report_nb,
        nb_output_path=report_out,
        country_code=country_code,
    )


if __name__ == "__main__":
    snt_era5_climate_data()
