import tempfile
from datetime import datetime
import zipfile
from io import BytesIO
from pathlib import Path
from shutil import copyfile

import geopandas as gpd
import polars as pl
import pandas as pd
from openhexa.sdk import current_run, parameter, pipeline, workspace
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
from snt_lib.snt_pipeline_utils import (
    pull_scripts_from_repository,
    add_files_to_dataset,
    load_configuration_snt,
    run_report_notebook,
    validate_config,
    save_pipeline_parameters,
)
from openhexa.toolbox.era5.cds import VARIABLES


@pipeline("snt_era5_aggregate")
@parameter(
    "run_report_only",
    name="Run reporting only",
    help="This will only execute the reporting notebook",
    type=bool,
    default=False,
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
def era5_aggregate(run_report_only: bool, pull_scripts: bool):
    """Aggregate ERA5 climate data by applying spatial and temporal aggregation to raw input files.

    Parameters
    ----------
    run_report_only : bool, optional
        If True, only the reporting notebook will be executed, skipping the aggregation steps.
    pull_scripts : bool, optional
        If True, the latest scripts will be pulled from the repository before running the pipeline.
    """
    root_path = Path(workspace.files_path)
    input_dir = root_path / "data" / "era5" / "raw"
    output_dir = root_path / "data" / "era5" / "aggregate"
    snt_pipeline_path = root_path / "pipelines" / "snt_era5_aggregate"

    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_era5_aggregate",
            report_scripts=["snt_era5_aggregate_report.ipynb"],
            code_scripts=[],
        )

    try:
        snt_config_dict = load_configuration_snt(
            config_path=root_path / "configuration" / "SNT_config.json"
        )
        validate_config(snt_config_dict)
        country_code = snt_config_dict["SNT_CONFIG"].get("COUNTRY_CODE")

        if not run_report_only:
            dhis2_formatted_dataset_id = snt_config_dict["SNT_DATASET_IDENTIFIERS"].get(
                "DHIS2_DATASET_FORMATTED"
            )
            era5_dataset_id = snt_config_dict["SNT_DATASET_IDENTIFIERS"].get("ERA5_DATASET_CLIMATE")

            # get boundaries geometries from formatted dataset
            boundaries = read_boundaries(
                dhis2_formatted_dataset_id, filename=f"{country_code}_shapes.geojson"
            )

            # subdirs containing raw data are named after variable names
            subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
            variables = [d.name for d in subdirs if d.name in VARIABLES]

            if not variables:
                raise FileNotFoundError("No variables found in input directory")

            filename_list = []
            for variable in variables:
                current_run.log_info(f"Running aggregations for {variable}.")
                daily = get_daily(
                    input_dir=input_dir / variable,
                    boundaries=boundaries,
                    variable=variable,
                    column_uid="ADM2_ID",
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

                current_run.log_info(
                    f"Applied epi. weekly aggregation to {variable} data ({len(epi_weekly)} rows)"
                )
                monthly = aggregate_per_month(
                    daily=daily,
                    column_uid="boundary_id",
                    sum_aggregation=sum_aggregation,
                )

                current_run.log_info(f"Applied monthly aggregation to {variable} data ({len(monthly)} rows)")
                dst_dir = output_dir / variable
                dst_dir.mkdir(parents=True, exist_ok=True)

                daily_fname = dst_dir / f"{country_code}_{variable}_daily.parquet"
                daily = apply_snt_formatting(daily, aggregation="daily")
                daily.write_parquet(daily_fname)
                # current_run.add_file_output(daily_fname.as_posix())

                weekly_fname = dst_dir / f"{country_code}_{variable}_weekly.parquet"
                weekly = apply_snt_formatting(weekly, aggregation="weekly")
                weekly.write_parquet(weekly_fname)
                # current_run.add_file_output(weekly_fname.as_posix())

                epi_weekly_fname = dst_dir / f"{country_code}_{variable}_epi_weekly.parquet"
                epi_weekly = apply_snt_formatting(epi_weekly, aggregation="epi_weekly")
                epi_weekly.write_parquet(epi_weekly_fname)
                # current_run.add_file_output(epi_weekly_fname.as_posix())

                monthly_fname = dst_dir / f"{country_code}_{variable}_monthly.parquet"
                monthly = apply_snt_formatting(monthly, aggregation="monthly")
                monthly.write_parquet(monthly_fname)
                # current_run.add_file_output(monthly_fname.as_posix())

                # collect filenames for adding to dataset (only monthly for now)
                filename_list.append(
                    # daily_fname,
                    # weekly_fname,
                    # epi_weekly_fname,
                    monthly_fname,
                )

            parameters_file = save_pipeline_parameters(
                pipeline_name="snt_era5_aggregate",
                parameters={"run_report_only": run_report_only, "pull_scripts": pull_scripts},
                output_path=output_dir,
                country_code=country_code,
            )
            filename_list.append(parameters_file)

            add_files_to_dataset(
                dataset_id=era5_dataset_id,
                country_code=country_code,
                file_paths=filename_list,
            )

        run_report_notebook(
            nb_file=snt_pipeline_path / "reporting" / "snt_era5_aggregate_report.ipynb",
            nb_output_path=snt_pipeline_path / "reporting" / "outputs",
            country_code=country_code,
        )

    except Exception as e:
        current_run.log_error(f"Error while running the pipeline: {e}")
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
    # Filter out invalid/empty geometries before aggregation
    # Check for valid geometries (not empty, not null, has area > 0)
    valid_mask = (
        boundaries.geometry.notna() 
        & ~boundaries.geometry.is_empty 
        & (boundaries.geometry.area > 0)
    )
    
    valid_boundaries = boundaries[valid_mask].copy()
    invalid_ids = boundaries[~valid_mask][column_uid].tolist()
    
    if len(invalid_ids) > 0:
        current_run.log_warning(
            f"Skipping {len(invalid_ids)} invalid/empty geometries: {invalid_ids[:10]}{'...' if len(invalid_ids) > 10 else ''}"
        )
    
    if len(valid_boundaries) == 0:
        raise ValueError("No valid geometries found in boundaries. Cannot perform aggregation.")
    
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
        masks = build_masks(valid_boundaries, nrows, ncols, transform)

        var = VARIABLES[variable]["shortname"]

        # Aggregate only for valid boundaries
        daily = aggregate(ds=ds, var=var, masks=masks, boundaries_id=valid_boundaries[column_uid])

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
    
    # Note: Invalid geometries are excluded from aggregation to avoid errors
    # They will not appear in the output. If needed, they can be added back with NA values
    # after formatting, but for now we skip them to prevent crashes.
    
    return daily


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


def apply_snt_formatting(df: pd.DataFrame, aggregation: str = "monthly") -> pd.DataFrame:
    """Apply SNT formatting to the aggregated ERA5 data.

    This function is a placeholder for any future formatting requirements specific to SNT.
    Currently, it does not perform any operations but can be extended as needed.

    Returns
    -------
    pd.DataFrame
        The formatted DataFrame with SNT-specific formatting applied.
    """
    aggregation_columns = {
        "daily": ["ADM2_ID", "DATE", "MEAN", "MIN", "MAX", "WEEK", "PERIOD", "EPI_WEEK"],
        "weekly": ["ADM2_ID", "WEEK", "MEAN", "MIN", "MAX"],
        "epi_weekly": ["ADM2_ID", "EPI_WEEK", "MEAN", "MIN", "MAX"],
        "monthly": ["ADM2_ID", "PERIOD", "MEAN", "MIN", "MAX"],
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
