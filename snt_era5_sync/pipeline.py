from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
from dateutil.relativedelta import relativedelta
from openhexa.sdk import CustomConnection, current_run, parameter, pipeline, workspace
from openhexa.toolbox.era5.cache import Cache
from openhexa.toolbox.era5.extract import Client
from openhexa.toolbox.era5.utils import get_variables
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    load_configuration_snt,
    pull_scripts_from_repository,
    run_report_notebook,
    save_pipeline_parameters,
    validate_config,
)
from era5_sync_utils import (
    CDS_API_URL,
    ERA5_VARIABLES,
    aggregate_daily_snt,
    apply_snt_formatting,
    build_daily_snt,
    get_bounds,
    get_cds_api_key,
    is_valid_ymd,
    read_boundaries,
    sync_variable,
    to_last_day_previous_month,
    validate_synced_zarr,
)


@pipeline("snt_era5_sync")
@parameter("start_date", type=str, name="Start date", help="Start date of extraction period.", default="2018-01-01", required=True)
@parameter("end_date", type=str, name="End date", help="End date of extraction period. Latest available by default.", required=False)
@parameter(
    "cds_connection",
    name="Climate data store",
    type=CustomConnection,
    help="Credentials for connection to the Copernicus Climate Data Store",
    required=True,
)
@parameter("pull_scripts", name="Pull Scripts", help="Pull the latest scripts from the repository", type=bool, default=False, required=False)
def snt_era5_sync(
    start_date: str,
    end_date: str | None,
    cds_connection: CustomConnection,
    pull_scripts: bool,
) -> None:
    """Unified ERA5 pipeline for SNT: synchronize raw ERA5 then aggregate to SNT outputs."""
    root_path = Path(workspace.files_path)
    raw_dir = root_path / "data" / "era5" / "raw"
    cache_dir = root_path / "data" / "era5" / "cache"
    output_dir = root_path / "data" / "era5" / "aggregate"
    report_nb = root_path / "pipelines" / "snt_era5_sync" / "reporting" / "snt_era5_sync_report.ipynb"
    report_out = root_path / "pipelines" / "snt_era5_sync" / "reporting" / "outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if pull_scripts:
        current_run.log_info("Pulling ERA5 scripts from repository.")
        try:
            pull_scripts_from_repository(
                pipeline_name="snt_era5_sync",
                report_scripts=["snt_era5_sync_report.ipynb"],
                code_scripts=[],
            )
        except Exception as e:
            current_run.log_warning(f"Could not pull snt_era5_sync scripts: {e}")

    if not is_valid_ymd(start_date):
        raise ValueError(f"Invalid start date format: {start_date}. Expected YYYY-MM-DD.")
    if not is_valid_ymd(end_date):
        raise ValueError(f"Invalid end date format: {end_date}. Expected YYYY-MM-DD.")

    snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
    validate_config(snt_config)
    country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]

    variables_to_run = ERA5_VARIABLES
    current_run.log_info(f"Variables to process: {variables_to_run}")

    dhis2_formatted_dataset = snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_DATASET_FORMATTED"]
    era5_dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"]["ERA5_DATASET_CLIMATE"]
    boundaries = read_boundaries(dhis2_formatted_dataset, filename=f"{country_code}_shapes.geojson")
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

    run_report_notebook(nb_file=report_nb, nb_output_path=report_out, country_code=country_code)


if __name__ == "__main__":
    snt_era5_sync()
