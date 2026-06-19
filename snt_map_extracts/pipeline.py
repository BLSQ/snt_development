from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
from malariaAtlasProject.map import MAPExtractorError, MAPRasterExtractor
from malariaAtlasProject.map_utils import (
    load_tiff_bands,
    parse_raster_filename_vars,
)
from openhexa.sdk import current_run, parameter, pipeline, workspace
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
from utils import (
    compute_population_weighted_metric,
    generate_population_table_from_raster,
    get_extract_periods,
    load_raw_population_raster,
)
from worldpopclient import WorldPopClient

# Ticket:
# https://bluesquare.atlassian.net/browse/SNT25-143 (old pipeline)
# https://bluesquare.atlassian.net/browse/SNT25-259 (old pipeline)
# https://bluesquare.atlassian.net/browse/SNT25-284
# https://bluesquare.atlassian.net/browse/SNT25-518 (include periods)


@pipeline("snt_map_extracts")
@parameter(
    code="year_start",
    name="Year start",
    help="Start year of indicators selection (e.g. 2022).",
    type=int,
    default=None,
    required=True,
)
@parameter(
    code="year_end",
    name="Year end",
    help="End year of indicators selection (e.g. 2023).",
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
def snt_map_extracts(year_start: int, year_end: int, run_report_only: bool, pull_scripts: bool) -> None:
    """Main function to get raster data for a dhis2 country."""
    root_path = Path(workspace.files_path)
    pipeline_path = root_path / "pipelines" / "snt_map_extracts"
    pipeline_path.mkdir(parents=True, exist_ok=True)

    if year_start > year_end:
        msg = f"Start period ({year_start}) must be less than or equal to end period ({year_end})."
        current_run.log_warning(msg)
        raise ValueError(msg)

    try:
        validate_worldpop_periods(year_start, year_end)
    except ValueError as e:
        current_run.log_warning(f"Invalid period configuration: {e}")  # pop is optional

    # Define indicators to download
    snt_indicators = {
        "Malaria": {
            "Pf_Parasite_Rate",
            "Pf_Mortality_Rate",
            "Pf_Incidence_Rate",
        },
        "Interventions": {
            "Insecticide_Treated_Net_Access",
            "Insecticide_Treated_Net_Use_Rate",
            "IRS_Coverage",
            "Antimalarial_Effective_Treatment",
        },
    }

    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_map_extracts",
            report_scripts=["snt_map_extracts_report.ipynb"],
            code_scripts=[],
        )

    # Load configuration
    snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
    validate_config(snt_config)
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")

    shapes = retrieve_shapes(snt_config=snt_config)
    if shapes is None:
        current_run.log_error("No valid shapes available. Processing stopped.")
        raise ValueError

    if not run_report_only:
        output_path = root_path / "data" / "map"
        output_path.mkdir(parents=True, exist_ok=True)

        parameters_file = save_pipeline_parameters(
            pipeline_name="snt_map_extracts",
            parameters={
                "year_start": year_start,
                "year_end": year_end,
                "run_report_only": run_report_only,
                "pull_scripts": pull_scripts,
            },
            output_path=output_path,
            country_code=country_code,
        )

        periods = get_extract_periods(start=str(year_start), end=str(year_end))
        files_to_dataset = []
        for year in periods:
            pop_table, pop_raster_path = get_or_download_population_table(
                year=year,
                country_code=country_code,
                shapes=shapes,
                wpop_repo_path=root_path / "data" / "worldpop",
                output_path=root_path / "data" / "map" / "aggregated_populations",
            )

            files_to_dataset += build_map_statistics_table(
                coverage_categories=snt_indicators,
                population_totals=pop_table,
                pop_raster_path=pop_raster_path,
                shapes=shapes,
                target_year=year,
                country_code=country_code,
                output_path=output_path,
            )

        add_files_to_dataset(
            dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"].get("SNT_MAP_EXTRACTS"),
            country_code=country_code,
            file_paths=files_to_dataset + [parameters_file],
        )

    else:
        current_run.log_info("Skipping calculations, running reporting.")

    run_report_notebook(
        nb_file=pipeline_path / "reporting" / "snt_map_extracts_report.ipynb",
        nb_output_path=pipeline_path / "reporting" / "outputs",
        country_code=country_code,
    )

    current_run.log_info("Pipeline completed successfully!")


def get_or_download_population_table(
    year: str, country_code: str, shapes: gpd.GeoDataFrame, wpop_repo_path: Path, output_path: Path
) -> tuple[pl.DataFrame | None, Path | None]:
    """Check if population raster exists for the given year and country, if not, download it.

    Parameters
    ----------
    year : str
        The year for which to retrieve the population data. (e.g.: "2020").
    country_code : str
        The 3-letter ISO code of the country (e.g.: "COD", "BFA").
    shapes : gpd.GeoDataFrame
        GeoDataFrame containing the shapes for zonal statistics.
    wpop_repo_path : Path
        Path to the worldpop pipeline directory where rasters and population tables are stored.
    output_path : Path
        Path to save the generated population table if it needs to be created.

    Returns
    -------
    Tuple[pl.DataFrame | None, Path | None]
        A tuple containing:
        - The population table as a Polars DataFrame if it was generated successfully, otherwise None.
        - The path to the population raster used for generating the table, or None if not retrieved.
    """
    pop_raster_path = list((wpop_repo_path / "rasters").glob(f"{country_code.lower()}_pop_{year}_*.tif"))
    if pop_raster_path:
        current_run.log_info(f"Population raster found for {year}: {pop_raster_path[0]}.")
        pop_raster_path = pop_raster_path[0]

    else:
        current_run.log_info(f"No population raster found for {year}. Attempting to download.")
        pop_raster_path = retrieve_population_data(
            country_code=country_code,
            year=year,
            output_path=wpop_repo_path / "rasters",
            overwrite=False,
        )

    if not pop_raster_path:
        current_run.log_warning(
            f"Population raster could not be retrieved for {year}. Skipping population table generation."
        )
        return None, None

    current_run.log_info(f"Generating population table for {year} from raster: {pop_raster_path}.")
    pop_table = generate_population_table_from_raster(raster_path=pop_raster_path, shapes=shapes)

    if pop_table is None:
        current_run.log_warning(f"Population table could not be generated for {year}.")
        return None, None

    output_path.mkdir(parents=True, exist_ok=True)
    pop_table.write_parquet(output_path / f"{country_code}_worldpop_population_{year}.parquet")
    current_run.log_info(
        f"Population table generated and saved for {year} at "
        f"{output_path / f'{country_code}_worldpop_population_{year}.parquet'}."
    )
    return pop_table, pop_raster_path


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
    output_path.mkdir(parents=True, exist_ok=True)
    country = country_code.upper()

    try:
        pop_file_path = wpop_client.download_data_for_country(
            country_iso3=country,
            year=int(year),
            output_dir=output_path,
            overwrite=overwrite,
        )
        current_run.log_info(f"Population raster successfully downloaded under: {pop_file_path}.")
        return pop_file_path
    except Exception as e:
        raise Exception(f"Error retrieving WorldPop data for {country} {year}: {e}") from e


def retrieve_shapes(snt_config: dict) -> gpd.GeoDataFrame | None:
    """Retrieve and validate shapes for the specified country.

    Parameters
    ----------
    snt_config : dict
        SNT configuration file.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the valid shapes for the specified country.

    """
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_shapes_id = snt_config.get("SNT_DATASET_IDENTIFIERS", {}).get("DHIS2_DATASET_FORMATTED")
    shapes = get_file_from_dataset(dataset_shapes_id, f"{country_code}_shapes.geojson")

    if shapes is None or shapes.shape[0] == 0:
        current_run.log_warning("No shapes found in dataset.")
        return None

    current_run.log_info(f"Shapes loaded from dataset: {dataset_shapes_id}.")

    # Drop None geometries — zonal_stats fails on None.
    invalid_shapes = shapes[shapes.geometry.isna()]
    if len(invalid_shapes) > 0:
        current_run.log_warning(f"Dropping {len(invalid_shapes)} organisation units without geometry.")
    shapes = shapes[shapes.geometry.notna() & shapes.geometry.apply(lambda g: g is not None)]

    # Drop empty geometries (geometry not None but empty)
    empty_shapes = shapes[shapes.geometry.is_empty]
    if len(empty_shapes) > 0:
        current_run.log_warning(f"Dropping {len(empty_shapes)} organisation units with empty geometry.")
    shapes = shapes[~shapes.geometry.is_empty]

    if len(shapes) == 0:
        return None

    return shapes


def build_map_statistics_table(
    coverage_categories: dict,
    population_totals: pl.DataFrame | None,
    pop_raster_path: Path | None,
    shapes: gpd.GeoDataFrame,
    target_year: str,
    country_code: str,
    output_path: Path,
) -> list[Path]:
    """Generate a table of zonal statistics for given coverage indicators and save the results.

    Parameters
    ----------
    coverage_categories : dict
        Dictionary mapping categories to indicator layer names.
    population_totals : pl.DataFrame | None
        DataFrame containing total population values for each shape, or None if not available.
    pop_raster_path : Path
        Path to the selected raster directory.
    shapes : gpd.GeoDataFrame
        GeoDataFrame containing the shapes for zonal statistics.
    target_year : str
        Target year for selecting indicator versions.
    country_code : str
        The 3-letter ISO code of the country (e.g.: "COD", "BFA").
    output_path : Path
        Path to save the output files.

    Returns
    -------
    list[Path]
        List of paths to the generated output files (parquet and csv).
    """
    rasters_path = output_path / "raster_files"
    rasters_path.mkdir(parents=True, exist_ok=True)

    raster_files = retrieve_rasters(
        coverage_categories=coverage_categories,
        target_year=target_year,
        shapes=shapes,
        rasters_path=rasters_path,
    )

    if len(raster_files) == 0:
        current_run.log_warning(
            f"No raster files were downloaded for year {target_year}. Exiting table generation."
        )
        return []

    try:
        map_indicators = compute_zonal_statistics(
            raster_files=raster_files,
            shapes=shapes,
            population_totals=population_totals,
            pop_raster_path=pop_raster_path,
        )
    except Exception as e:
        current_run.log_error(f"Error during aggregation: {e}")
        return []

    if map_indicators.is_empty():
        current_run.log_warning(f"No valid statistics were computed for year {target_year}.")
        return []

    # Save file
    out_dir = output_path / "formatted"
    out_dir.mkdir(parents=True, exist_ok=True)
    file_parquet = out_dir / f"{country_code}_map_data_{target_year}.parquet"
    file_csv = out_dir / f"{country_code}_map_data_{target_year}.csv"
    map_indicators.write_parquet(file_parquet)
    map_indicators.write_csv(file_csv)
    current_run.log_info(f"Output file saved under : {file_csv}")

    return [file_parquet, file_csv]


def retrieve_rasters(
    coverage_categories: dict,
    target_year: str,
    shapes: gpd.GeoDataFrame,
    rasters_path: Path,
) -> list[Path]:
    """Retrieve raster files for specified coverage categories and indicators.

    Returns:
        A list of paths to the downloaded raster files.
    """
    downloaded_rasters = []
    for category, indicators in coverage_categories.items():
        current_run.log_info(f"Processing category: {category}.")
        map_extractor = MAPRasterExtractor(category=category)
        for indicator in indicators:
            try:
                current_run.log_info(f"Downloading raster for indicator: {indicator}.")
                raster_path = map_extractor.download_indicator_raster(
                    indicator=indicator,
                    target_year=target_year,
                    shapes=shapes,
                    output_path=rasters_path,
                    replace_file=False,
                )
                downloaded_rasters.append(raster_path)
            except MAPExtractorError as e:
                current_run.log_error(f"Error downloading raster for {indicator}. Details: {e}")
                continue

    return downloaded_rasters


def compute_zonal_statistics(
    raster_files: list[Path],
    shapes: gpd.GeoDataFrame,
    population_totals: pl.DataFrame | None,
    pop_raster_path: Path | None,
) -> pl.DataFrame:
    """Run zonal statistics aggregations on the downloaded rasters.

    Returns:
        A Polars DataFrame containing the aggregated statistics for each indicator and shape.
    """
    # 1. Load population raster (if available)
    pop_data = pop_transform = pop_crs = pop_nodata = None
    if pop_raster_path is None:
        current_run.log_warning("Population raster file not provided.")
    else:
        pop_data, pop_transform, pop_crs, pop_nodata = load_raw_population_raster(
            raster_path=pop_raster_path,
        )
        # Set nodata to np.nan
        if pop_data is not None:
            pop_data = pop_data.astype(float)
            pop_data[pop_data == pop_nodata] = np.nan

    # 2. Process each raster file
    final_df = pl.DataFrame()
    for raster_file in raster_files:
        file_vars = parse_raster_filename_vars(raster_file)
        coverage_id = (
            f"{file_vars['category']}__{file_vars['version']}_{file_vars['region']}_{file_vars['indicator']}"
        )

        bands = MAPRasterExtractor(category=file_vars["category"]).get_band_names(coverage_id=coverage_id)
        raster_data, raster_transform, raster_crs, raster_nodata = load_tiff_bands(
            raster_file, band_names=bands
        )

        current_run.log_info(f"Computing {raster_file.name} statistics...")
        ref_columns = ["ADM1_NAME", "ADM1_ID", "ADM2_NAME", "ADM2_ID"]
        bands_for_statistics = ["Data", "LCI", "UCI", "GRAY_INDEX"]
        stats_results = []

        # Compute Zonal Statistics per layer
        for band in bands:
            if band in bands_for_statistics:
                current_run.log_info(f"Processing {file_vars['indicator']} band: {band}.")
                zstats = zonal_stats(
                    vectors=shapes,
                    raster=raster_data[band],
                    affine=raster_transform,
                    stats=["mean"],
                    geojson_out=True,
                    nodata=raster_nodata,
                )
                result_gdf = gpd.GeoDataFrame.from_features(zstats).drop(columns=["geometry"])
                metric_var = "MEAN" if band in ["Data", "GRAY_INDEX"] else band
                result_gdf = result_gdf.rename(columns={"mean": metric_var})
                melt_df = result_gdf.melt(
                    id_vars=ref_columns,
                    value_vars=[metric_var],
                    var_name="statistic",
                    value_name="value",
                )

                # Compute population-weighted metric (extra column)
                if pop_data is not None:
                    weighted_metric = compute_population_weighted_metric(
                        metric_data=raster_data[band],
                        metric_transform=raster_transform,
                        metric_crs=raster_crs,
                        metric_nodata=raster_nodata,
                        pop_data=pop_data,
                        pop_transform=pop_transform,
                        pop_crs=pop_crs,
                        population_totals=population_totals,
                        shapes=shapes,
                        indicator=file_vars["indicator"],
                    )
                    if weighted_metric is not None:
                        melt_df = (
                            pl.from_pandas(melt_df)
                            .join(
                                weighted_metric.select(["ADM2_ID", "population_weighted"]),
                                on="ADM2_ID",
                                how="left",
                            )
                            .with_columns(
                                pl.col("population_weighted").cast(pl.Float64, strict=False),
                                pl.col("value").cast(pl.Float64, strict=False),
                            )
                        )
                    else:
                        melt_df = pl.from_pandas(melt_df).with_columns(
                            pl.lit(None).cast(pl.Float64).alias("population_weighted")
                        )

                else:
                    melt_df = pl.from_pandas(melt_df).with_columns(
                        pl.lit(None).cast(pl.Float64).alias("population_weighted")
                    )

                stats_results.append(melt_df)

        # Log missing bands for raster_file
        # NOTE: This is not an error, just info about the available layers per coverage,
        # some of them only have a GRAY_INDEX band
        missing = [s for s in ["Data", "LCI", "UCI"] if s not in bands]
        if bands == ["GRAY_INDEX"]:
            current_run.log_warning(
                f"{file_vars['indicator']} contains only the 'GRAY_INDEX' band; "
                f"no main indicator bands found.",
            )
        elif missing:
            current_run.log_warning(
                f"{file_vars['indicator']} is missing bands: {missing}. Using available band(s): {bands}.",
            )

        if len(stats_results) > 0:
            # Format results, add metadata
            stats = pl.concat(stats_results)
            stats = stats.with_columns(
                [
                    pl.lit(file_vars["category"]).alias("metric_category"),
                    pl.lit(file_vars["indicator"]).alias("metric_name"),
                    pl.lit(file_vars["version"]).alias("version"),
                    pl.lit(int(file_vars["year"])).alias("year"),
                    pl.col("value").cast(pl.Float64, strict=False),
                ]
            )
            final_df = pl.concat([final_df, stats])  # concat final table

    if final_df.shape[0] == 0:
        return pl.DataFrame()  # No valid statistics computed, return empty DataFrame

    # SNT format
    final_df = final_df.rename({col: col.strip().upper() for col in final_df.columns})
    return final_df.with_columns(pl.col("METRIC_NAME").str.strip_chars())


def validate_worldpop_periods(start: int, end: int) -> None:
    """Validate that start and end periods are in the correct format and logical.

    Raises
    ------
    ValueError
        If start or end are not valid integers or if start is greater than end.
    """
    if not (2015 <= start <= 2030):
        raise ValueError(
            f"Start year {start} is out of range for population rasters available in repository (2015-2030)."
            " (see: https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/)"
        )
    if not (2015 <= end <= 2030):
        raise ValueError(
            f"End year {end} is out of range for population rasters available in repository (2015-2030)."
            " (see: https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/)"
        )


if __name__ == "__main__":
    snt_map_extracts()
