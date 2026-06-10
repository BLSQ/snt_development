from pathlib import Path

import geopandas as gpd
import polars as pl
import rasterio
from openhexa.sdk import current_run, parameter, pipeline, workspace
from openhexa.toolbox.dhis2.periods import period_from_string
from rasterstats import zonal_stats
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    get_file_from_dataset,
    load_configuration_snt,
    pull_scripts_from_repository,
    run_report_notebook,
    save_pipeline_parameters,
    validate_config,
)
from worlpopclient import WorldPopClient


@pipeline("snt_worldpop_extract")
@parameter(
    "year_start",
    name="Year start",
    help="Start year period for WorldPop population rasters extraction (e.g. 2020)",
    type=int,
    default=None,
    required=True,
)
@parameter(
    "year_end",
    name="Year end",
    help="End year period for WorldPop population rasters extraction (e.g. 2025)",
    type=int,
    default=None,
    required=True,
)
@parameter(
    "run_report_only",
    name="Run reporting only",
    help="This will only execute the reporting notebook",
    type=bool,
    default=False,
)
@parameter(
    "pull_scripts",
    name="Pull Scripts",
    help="Pull the latest scripts from the repository",
    type=bool,
    default=False,
    required=False,
)
def snt_worldpop_extract(year_start: int, year_end: int, run_report_only: bool, pull_scripts: bool) -> None:
    """Write your pipeline orchestration here."""
    # set paths
    snt_root_path = Path(workspace.files_path)
    pipeline_path = snt_root_path / "pipelines" / "snt_worldpop_extract"
    data_path = snt_root_path / "data" / "worldpop"

    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_worldpop_extract",
            report_scripts=["snt_worldpop_extract_report.ipynb"],
            code_scripts=[],
        )

    # get configuration
    snt_config_dict = load_configuration_snt(config_path=snt_root_path / "configuration" / "SNT_config.json")
    validate_config(snt_config_dict)

    # get country identifier for file naming
    country_code = snt_config_dict["SNT_CONFIG"].get("COUNTRY_CODE")

    if run_report_only:
        current_run.log_info("Running report notebook only.")
        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_worldpop_extract_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            country_code=country_code,
        )
        current_run.log_info("Report notebook executed successfully. Ending pipeline run.")
        return

    try:
        validate_periods(year_start, year_end)
    except ValueError as e:
        raise ValueError(f"Invalid period configuration: {e!s}") from e

    parameters_file = save_pipeline_parameters(
        pipeline_name="snt_worldpop_extract",
        parameters={
            "year_start": year_start,
            "year_end": year_end,
            "pull_scripts": pull_scripts,
        },
        output_path=data_path,
        country_code=country_code,
    )

    periods = get_extract_periods(start=str(year_start), end=str(year_end))
    ready_files = []
    for year in periods:
        current_run.log_info(f"Processing WorldPop population data for year {year}.")
        try:
            raster_path = retrieve_population_data(
                country_code=country_code,
                year=year,
                output_path=data_path / "rasters",
                overwrite=False,
            )
        except Exception as e:
            current_run.log_warning(f"{e!s}")
            continue

        pop_agg_path = run_spatial_aggregation(
            tif_file_path=raster_path,
            year=year,
            snt_config=snt_config_dict,
            output_dir=data_path / "aggregations",
        )

        pop_formatted_path = snt_worldpop_format(
            pop_data_path=pop_agg_path,
            year=year,
            country_code=country_code,
            output_dir=data_path / "population",
        )

        if pop_formatted_path:
            ready_files.append(pop_formatted_path)

    # Concatenate processed years into a single file
    files_to_publish = concat_files(ready_files, country_code, data_path / "population")
    if not files_to_publish:
        current_run.log_warning("No population data files processed. Skipping dataset publication.")
        return

    add_files_to_dataset(
        dataset_id=snt_config_dict["SNT_DATASET_IDENTIFIERS"].get("WORLDPOP_DATASET_EXTRACT"),
        country_code=country_code,
        file_paths=files_to_publish + [parameters_file],
    )

    run_report_notebook(
        nb_file=pipeline_path / "reporting" / "snt_worldpop_extract_report.ipynb",
        nb_output_path=pipeline_path / "reporting" / "outputs",
        country_code=country_code,
    )


def retrieve_population_data(
    country_code: str, year: str, output_path: Path, overwrite: bool = False
) -> Path:
    """Retrieve raster population data from worldpop.

    Parameters
    ----------
    country_code : str
        The 3-letter ISO code of the country (e.g.: "COD", "BFA").
    year : str, optional
        The year for which to retrieve the population data. (e.g.: "2020").
    overwrite : bool, optional
        Whether to overwrite existing files. Defaults to False.
    output_path : Path
        The directory where the population data will be saved.

    Returns
    -------
    Path
        The full path to the saved population raster file.

    """
    current_run.log_info("Retrieving population data grid from WorldPop.")
    wpop_client = WorldPopClient()

    # Create output directory (and parents e.g. data/worldpop/) if missing
    output_path.mkdir(parents=True, exist_ok=True)
    country = country_code.upper()

    try:
        pop_file_path = wpop_client.download_data_for_country(
            country_iso3=country,
            year=year,
            output_dir=output_path,
            overwrite=overwrite,
        )
        current_run.log_info(f"Population raster successfully downloaded under: {pop_file_path}.")
        return pop_file_path
    except Exception as e:
        raise Exception(f"Error retrieving WorldPop data for {country} {year}: {e}") from e


def run_spatial_aggregation(tif_file_path: Path, year: str, snt_config: dict, output_dir: Path) -> Path:
    """Run spatial aggregation on the worldpop population data (tif file).

    Returns
    -------
        Path
            The full path to the saved aggregated population data file (parquet).
    """
    if not tif_file_path or not tif_file_path.exists():
        current_run.log_warning(f"Population file not found: {tif_file_path}. Skipping aggregation.")
        return None

    current_run.log_info(f"Running spatial aggregation with WorldPop data {tif_file_path}")

    # Load DHIS2 shapes
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    shapes = get_file_from_dataset(
        dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"].get("DHIS2_DATASET_FORMATTED"),
        filename=f"{country_code}_shapes.geojson",
    )

    # Filter out invalid geometries before zonal_stats (null, empty, or invalid e.g. self-intersecting)
    initial_count = len(shapes)
    shapes = shapes[shapes.geometry.notna()]
    shapes = shapes[~shapes.geometry.is_empty]
    shapes = shapes[shapes.geometry.is_valid]
    filtered_count = len(shapes)
    if initial_count != filtered_count:
        current_run.log_warning(
            f"Filtered out {initial_count - filtered_count} shapes with invalid geometries. "
            f"Processing {filtered_count} valid shapes."
        )
    if len(shapes) == 0:
        raise ValueError("No valid geometries found in shapes file. Cannot compute zonal statistics.")

    # Ensure CRS matches the raster & reproject if necessary
    if shapes.crs is None:
        raise ValueError("Shapes GeoDataFrame must have a defined CRS.")
    with rasterio.open(tif_file_path) as src:
        # Reproject shapes if CRS is different
        if shapes.crs != src.crs:
            current_run.log_info(
                f"The CRS data differs from the provided shapes file. Reprojecting shapes with {src.crs}"
            )
            shapes = shapes.to_crs(src.crs)

        nodata = src.nodata  # No data value

    # get statistics
    current_run.log_info(f"Computing ADM2 spacial aggregation for {len(shapes)} shapes.")
    pop_stats = zonal_stats(
        shapes,
        tif_file_path,
        stats=["sum", "count"],
        nodata=nodata,  # -99999.0
        geojson_out=True,
    )

    # Formats
    result_gdf = gpd.GeoDataFrame.from_features(pop_stats)
    result_gdf = result_gdf.drop(columns=["geometry"])
    result_df = pl.DataFrame(result_gdf).rename({"sum": "population", "count": "pixel_count"})
    result_df = result_df.with_columns(pl.col("population").round(0).cast(pl.Int64))
    result_df.columns = [c.upper() for c in result_df.columns]

    # Log any administrative levels with no population data
    no_data = result_df.filter(pl.col("POPULATION") == 0)
    for row in no_data.iter_rows(named=True):
        current_run.log_warning(
            f"Administrative level 2 : {row['ADM2_NAME']} ({row['ADM2_ID']}) has no population data."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{country_code}_worldpop_agg_{year}.parquet"
    result_df.write_parquet(output_path)
    current_run.log_info(f"Aggregated population data saved under: {output_path}")
    return output_path


def snt_worldpop_format(pop_data_path: Path, year: str, country_code: str, output_dir: Path) -> Path:
    """Format aggregated WorldPop population data for SNT.

    Returns
    -------
        Path
            The full path to the saved formatted population data file (parquet).
    """
    if not pop_data_path or not pop_data_path.exists():
        current_run.log_warning(
            f"No aggregated population data found for {country_code}. Skipping formatting."
        )
        return None

    pop_data = pl.read_parquet(pop_data_path)
    df = pop_data.with_columns(pl.lit(year).alias("YEAR"))
    df = df.select(["YEAR", "ADM1_NAME", "ADM1_ID", "ADM2_NAME", "ADM2_ID", "POPULATION"])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_fpath = output_dir / f"{country_code}_worldpop_population_{year}.parquet"
    df.write_parquet(output_fpath)
    current_run.log_info(f"Formatted population data saved under: {output_fpath}")
    return output_fpath


def get_extract_periods(start: str, end: str) -> list[str]:
    """Generates a list of periods between start and end in YYYYMM format.

    Returns
    -------
    list[str]
        List of periods in YYYYMM format.
    """
    try:
        # Get periods
        start_period = period_from_string(start)
        end_period = period_from_string(end)
        extract_periods = (
            [str(p) for p in start_period.get_range(end_period)]
            if str(start_period) < str(end_period)
            else [str(start_period)]
        )
    except Exception as e:
        raise Exception(f"Error in start/end date configuration: {e!s}") from e
    return extract_periods


def validate_periods(start: int, end: int) -> None:
    """Validate that start and end periods are in the correct format and logical.

    Raises
    ------
    ValueError
        If start or end are not valid integers or if start is greater than end.
    """
    if not (2000 <= start <= 2030):
        raise ValueError(f"Start year {start} is out of range. Must be between 2000 and 2030.")
    if not (2000 <= end <= 2030):
        raise ValueError(f"End year {end} is out of range. Must be between 2000 and 2030.")
    if start > end:
        raise ValueError("Start period must be less than or equal to end period.")


def concat_files(file_paths: list[Path], country_code: str, output_dir: Path) -> list[Path]:
    """Concatenate multiple parquet files into a single file.

    Returns
    -------
        list[Path]
            The full path to the saved concatenated population data file (csv and parquet).
    """
    if not file_paths:
        current_run.log_warning("No files to concatenate.")
        return []

    current_run.log_info(f"Concatenating {len(file_paths)} population data files.")
    dfs = []
    for file_path in file_paths:
        if file_path and file_path.exists():
            dfs.append(pl.read_parquet(file_path))
        else:
            current_run.log_warning(f"File not found and will be skipped in concatenation: {file_path}")

    if not dfs:
        current_run.log_warning("No valid files found for concatenation.")
        return []

    concat_df = pl.concat(dfs)
    concat_df.write_csv(output_dir / f"{country_code}_worldpop_population.csv")
    concat_df.write_parquet(output_dir / f"{country_code}_worldpop_population.parquet")
    current_run.log_info(f"Concatenated population data saved under: {output_dir}")
    return [
        output_dir / f"{country_code}_worldpop_population.csv",
        output_dir / f"{country_code}_worldpop_population.parquet",
    ]


if __name__ == "__main__":
    snt_worldpop_extract()
