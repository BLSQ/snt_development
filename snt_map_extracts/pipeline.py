import logging
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np

# import pandas as pd
import polars as pl
import rasterio
from affine import Affine
from malariaAtlasProject.map import MAPExtractorError, MAPRasterExtractor
from malariaAtlasProject.map_utils import (
    load_tiff_bands,
    parse_raster_filename_vars,
)
from openhexa.sdk import current_run, parameter, pipeline, workspace
from openhexa.toolbox.dhis2.periods import period_from_string
from rasterio.warp import Resampling, reproject
from rasterstats import zonal_stats

# from owslib.wcs import WebCoverageService
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
    type=str,
    default=None,
    required=True,
)
@parameter(
    code="year_end",
    name="Year end",
    help="End year of indicators selection (e.g. 2023).",
    type=str,
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
    # logger = create_file_logger(log_path=pipeline_path / "logs")

    if year_start > year_end:
        current_run.log_warning("Start period must be less than or equal to end period.")
        raise ValueError

    try:
        validate_worldpop_periods(year_start, year_end)
    except ValueError as e:
        current_run.log_warning(f"Invalid period configuration: {e}")

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
    dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("SNT_MAP_EXTRACTS")

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
        for year in periods:
            # check in the worldpop folder for matching raster
            # if it doesn't exist , download and save it in the worldpop folder.
            shapes = retrieve_shapes(snt_config=snt_config)

            if not shapes:
                current_run.log_error("No valid shapes found. Skipping processing.")
                raise ValueError

            pop_raster_path = try_build_population_table(
                year=year,
                country_code=country_code,
                shapes=shapes,
                wpop_pipeline_path=root_path / "pipelines" / "worldpop",
                output_path=pipeline_path / "aggregated_populations",
            )

            # get rasters from Map

            # make table

            make_table(
                coverage_categories=snt_indicators,
                snt_config=snt_config,
                pop_raster_path=pop_raster_path,
                target_year=year,
                output_path=output_path,
            )

        add_files_to_dataset(
            dataset_id=dataset_id,
            country_code=country_code,
            file_paths=[
                output_path / "formatted" / country_code / f"{country_code}_map_data.parquet",
                output_path / "formatted" / country_code / f"{country_code}_map_data.csv",
                parameters_file,
            ],
        )

    else:
        current_run.log_info("Skipping calculations, running only the reporting.")

    run_report_notebook(
        nb_file=pipeline_path / "reporting" / "snt_map_extracts_report.ipynb",
        nb_output_path=pipeline_path / "reporting" / "outputs",
        country_code=country_code,
    )

    current_run.log_info("Pipeline completed successfully!")


def try_build_population_table(
    year: str, country_code: str, shapes: gpd.GeoDataFrame, wpop_pipeline_path: Path, output_path: Path
) -> Path | None:
    """Check if population raster exists for the given year and country, if not, download it.

    Parameters
    ----------
    year : str
        The year for which to retrieve the population data. (e.g.: "2020").
    country_code : str
        The 3-letter ISO code of the country (e.g.: "COD", "BFA").
    shapes : gpd.GeoDataFrame
        GeoDataFrame containing the shapes for zonal statistics.
    wpop_pipeline_path : Path
        Path to the worldpop pipeline directory where rasters and population tables are stored.
    output_path : Path
        Path to save the generated population table if it needs to be created.

    Returns
    -------
    Path | None
        The path to the population raster file if it exists or was downloaded, otherwise None.
    """
    pop_raster_path = list((wpop_pipeline_path / "rasters").glob(f"{country_code}_pop_{year}_*.tif"))
    if pop_raster_path:
        current_run.log_info(f"Population raster found for {year}: {pop_raster_path[0]}.")
        pop_raster_path = pop_raster_path[0]

        # check if the table for that year already exists
        pop_raster_table = list(
            (wpop_pipeline_path / "population").glob(f"{country_code}_worldpop_population_{year}.parquet")
        )
        if pop_raster_table:
            current_run.log_info(f"Population table already exists for {year}: {pop_raster_table[0]}.")
            return pl.read_parquet(pop_raster_table[0])

    else:
        current_run.log_info(f"No population raster found for {year}. Attempting to download.")
        # Here you would implement the logic to download the population raster for the given year and country.
        pop_raster_path = retrieve_population_data(
            country_code=country_code,
            year=year,
            output_path=wpop_pipeline_path / "rasters",
            overwrite=False,
        )

    if not pop_raster_path:
        current_run.log_warning(
            f"Population raster could not be retrieved for {year}. Skipping population table generation."
        )
        return None

    current_run.log_info(f"Generating population table for {year} from raster: {pop_raster_path}.")
    pop_table = generate_population_table_from_raster(raster_path=pop_raster_path, shapes=shapes)
    pop_table.write_parquet(output_path / f"{country_code}_worldpop_population_{year}.parquet")
    current_run.log_info(
        f"Population table generated and saved for {year} at "
        f"{output_path / f'{country_code}_worldpop_population_{year}.parquet'}."
    )
    return pop_table


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


def retrieve_shapes(snt_config: dict) -> gpd.GeoDataFrame | None:
    """Retrieve and validate shapes for the specified country.

    Parameters
    ----------
    snt_config : str
        SNT configuration file.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the valid shapes for the specified country.

    """
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_shapes_id = snt_config.get("SNT_DATASET_IDENTIFIERS", {}).get("DHIS2_DATASET_FORMATTED")
    shapes = get_file_from_dataset(dataset_shapes_id, f"{country_code}_shapes.geojson")
    current_run.log_info(f"Shapes loaded from dataset: {dataset_shapes_id}.")

    # Check shapes: drop rows with null or None geometry (zonal_stats fails on None).
    # Dropped ADM2 will not appear in map_data.parquet; in assemble_results they get NA for map
    # indicators (left join on ADM2_ID). The shapes file in the dataset is not modified.
    invalid_shapes = shapes[shapes.geometry.isna() | shapes.geometry.apply(lambda g: g is None)]
    if len(invalid_shapes) > 0:
        current_run.log_warning(f"Dropping {len(invalid_shapes)} organisation units without geometry.")
    shapes = shapes[shapes.geometry.notna() & shapes.geometry.apply(lambda g: g is not None)]

    # Drop empty geometries so rasterstats/shapely don't get invalid geometries
    empty_shapes = shapes[shapes.geometry.is_empty]
    if len(empty_shapes) > 0:
        current_run.log_warning(f"Dropping {len(empty_shapes)} organisation units with empty geometry.")
    shapes = shapes[~shapes.geometry.is_empty]

    if len(shapes) == 0:
        return None

    return shapes


def generate_population_table_from_raster(raster_path: Path, shapes: gpd.GeoDataFrame) -> pl.DataFrame:
    """Generate a population table from the given raster and shapes.

    Parameters
    ----------
    raster_path : Path
        Path to the population raster file.
    shapes : gpd.GeoDataFrame
        GeoDataFrame containing the shapes for zonal statistics.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame containing the population data for each shape.
    """
    # Load raster data (this is a placeholder, implement as needed)
    with rasterio.open(raster_path) as src:
        pop_data = src.read(1)
        pop_transform = src.transform
        pop_crs = src.crs
        pop_nodata = src.nodata

    # Compute total populations for each shape using zonal statistics
    return compute_total_populations(
        shapes=shapes,
        data=pop_data,
        transform=pop_transform,
        crs=pop_crs,
        nodata=pop_nodata,
    )


def make_table(
    coverage_categories: dict,
    snt_config: str,
    pop_raster_path: Path,
    shapes: gpd.GeoDataFrame,
    target_year: str,
    output_path: Path,
) -> None:
    """Generate a table of zonal statistics for given coverage indicators and save the results.

    Parameters
    ----------
    coverage_categories : dict
        Dictionary mapping categories to indicator layer names.
    snt_config : str
        SNT configuration file.
    pop_raster_path : Path
        Path to the selected raster directory.
    target_year : str
        Target year for selecting indicator versions.
    output_path : Path
        Path to save the output files.
    logger : logging.Logger
        Logger for logging messages.
    """
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    rasters_path = output_path / "raster_files" / country_code
    rasters_path.mkdir(parents=True, exist_ok=True)

    raster_files = retrieve_rasters(
        coverage_categories=coverage_categories,
        target_year=target_year,
        shapes=shapes,
        rasters_path=rasters_path,
    )

    if len(raster_files) == 0:
        current_run.log_warning("No raster files were downloaded. Exiting table generation.")
        return

    run_aggregations(
        raster_files=raster_files,
        shapes=shapes,
        pop_raster_path=pop_raster_path,
        snt_config=snt_config,
        output_path=output_path / "formatted" / country_code,
    )


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


def run_aggregations(
    raster_files: list[Path],
    shapes: gpd.GeoDataFrame,
    pop_raster_path: Path | None,
    snt_config: str,
    output_path: Path,
):
    """Run zonal statistics aggregations on the downloaded rasters."""
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")

    # 1. Load population raster (if available)
    if not pop_raster_path:
        current_run.log_warning("Population raster file not provided.")
        pop_data = None
    else:
        pop_data, pop_transform, pop_crs, pop_nodata = load_raw_population_raster(
            file_pattern=pop_raster_path.name,
            raster_path=pop_raster_path.parent,
        )
        pop_total = compute_total_populations(
            shapes, data=pop_data, transform=pop_transform, crs=pop_crs, nodata=pop_nodata
        )

        # Set nodata to np.nan
        if pop_data is not None:
            pop_data = pop_data.astype(float)
            pop_data[pop_data == pop_nodata] = np.nan

    # 2. Process each raster file
    final_df = pd.DataFrame()
    for raster_file in raster_files:
        file_vars = parse_raster_filename_vars(raster_file)
        coverage_id = (
            f"{file_vars['category']}__{file_vars['version']}_{file_vars['region']}_{file_vars['indicator']}"
        )

        bands = MAPRasterExtractor(category=file_vars["category"], logger=logger).get_band_names(
            coverage_id=coverage_id
        )
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
                        total_population=pop_total,
                        shapes=shapes,
                        indicator=file_vars["indicator"],
                        logger=logger,
                    )
                    if weighted_metric is not None:
                        # We can add population if we need it 'total_population'
                        melt_df = melt_df.merge(
                            weighted_metric[["ADM2_ID", "population_weighted"]],
                            on="ADM2_ID",
                            how="left",
                        )
                    else:
                        melt_df["population_weighted"] = None  # default
                else:
                    melt_df["population_weighted"] = None  # default

                stats_results.append(melt_df)

        # Log missing bands
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
            stats = pd.concat(stats_results, ignore_index=True)
            stats["metric_category"] = file_vars["category"]
            stats["metric_name"] = file_vars["indicator"]
            stats["version"] = file_vars["version"]
            stats["year"] = int(file_vars["year"])
            stats["value"] = pd.to_numeric(stats["value"], errors="coerce")

            # concat final table
            final_df = pd.concat([final_df, stats], ignore_index=True)

    # SNT format
    final_df.columns = [col.strip().upper() for col in final_df.columns]
    final_df["METRIC_NAME"] = final_df["METRIC_NAME"].str.strip()

    # Save Output
    output_path.mkdir(parents=True, exist_ok=True)

    # Save file
    final_df.to_parquet(output_path / f"{country_code}_map_data.parquet", index=False)
    final_df.to_csv(output_path / f"{country_code}_map_data.csv", index=False)
    current_run.log_info(f"Output file saved under : {output_path / f'{country_code}_map_data.csv'}")


def align_raster_to_reference(
    data: np.ndarray,
    crs: str,
    transform: Affine,
    reference_data: np.ndarray,
    reference_crs: str,
    reference_transform: Affine,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Align a metric raster to match a reference raster (CRS and shape).

    Parameters
    ----------
    data : np.ndarray
        2D array of the metric raster.
    crs : rasterio.crs.CRS or str
        CRS of the metric raster.
    transform : Affine
        Affine transform of the metric raster.
    reference_data : np.ndarray
        2D array of the reference raster.
    reference_crs : rasterio.crs.CRS or str
        CRS of the reference raster.
    reference_transform : Affine
        Affine transform of the reference raster.
    resampling : rasterio.enums.Resampling
        Resampling method (default: bilinear).

    Returns
    -------
    np.ndarray
        Metric raster reprojected and resampled to reference grid.
    """
    reference_shape = reference_data.shape
    aligned = np.empty(reference_shape, dtype=data.dtype)

    # Only reproject if CRS or shape/transform differ
    if (crs != reference_crs) or (data.shape != reference_shape):
        reproject(
            source=data,
            destination=aligned,
            src_transform=transform,
            src_crs=crs,
            dst_transform=reference_transform,
            dst_crs=reference_crs,
            resampling=resampling,
        )
    else:
        # Already aligned
        aligned[:] = data

    return aligned


def compute_population_weighted_metric(
    metric_data: np.ndarray,
    metric_transform: Affine,
    metric_crs: str,
    metric_nodata: float,
    pop_data: np.ndarray,
    pop_transform: Affine,
    pop_crs: str,
    total_population: pd.DataFrame,
    shapes: gpd.GeoDataFrame,
    indicator: str,
    logger: logging.Logger,
) -> pd.Series:
    """Compute weighted metric values for given shapes using population data.

    Parameters
    ----------
    metric_data : np.ndarray
        2D array of the metric raster, nodata values set to np.nan.
    metric_transform : Affine
        Affine transform of the metric raster.
    metric_crs : str
        CRS of the metric raster.
    metric_nodata : float
        NoData value of the metric raster.
    pop_data:
        2D array of the population raster, nodata values set to np.nan.
    pop_transform:
        Affine transform of the population raster.
    pop_crs:
        CRS of the population raster.
    total_population:
        DataFrame containing total populations for each shape.
    shapes : gpd.GeoDataFrame
        GeoDataFrame containing the shapes for zonal statistics.
    indicator : str
        Name of the indicator being processed.
    logger : logging.Logger
        Logger for logging messages.

    Returns
    -------
    pd.Series
        Series containing the weighted metric values for each shape or None if population data is unavailable.
    """
    if any(
        x is None
        for x in (shapes, metric_data, metric_transform, metric_crs, pop_data, pop_transform, pop_crs)
    ):
        log_message(
            logger, f"Population-weighted computation skipped for metric: {indicator}.", level="warning"
        )
        return None

    log_message(logger, f"Computing population-weighted for metric: {indicator}.")
    # Align metric raster to population raster (resolution and CRS)
    metric_aligned = align_raster_to_reference(
        data=metric_data,
        crs=metric_crs,
        transform=metric_transform,
        reference_data=pop_data,
        reference_crs=pop_crs,
        reference_transform=pop_transform,
        resampling=Resampling.nearest,  # nearest repeats metric values
    )

    metric_aligned = metric_aligned.astype(float)
    metric_aligned[metric_aligned == metric_nodata] = np.nan

    # Multiply
    weighted_raster = pop_data * metric_aligned
    zstats_w = zonal_stats(
        vectors=shapes,
        raster=weighted_raster,
        affine=pop_transform,
        stats=["sum"],
        geojson_out=True,
        nodata=np.nan,
    )
    result_w = pd.DataFrame(
        [
            {
                "ADM2_ID": f["properties"].get("ADM2_ID"),
                "weighted_sum": f["properties"]["sum"],
            }
            for f in zstats_w
        ]
    )
    result_w["ADM2_ID"] = result_w["ADM2_ID"].astype(str)
    result = result_w.merge(total_population, on="ADM2_ID", how="left")
    result["population_weighted"] = result["weighted_sum"] / result["total_population"]
    return result


def load_raw_population_raster(file_pattern: str, raster_path: Path, logger: logging.Logger) -> tuple:
    """Load raw population raster from the specified path.

    Parameters
    ----------
    file_pattern : str
        Pattern to match the population raster file.
    raster_path : Path
        Path to the population raster file.
    logger : logging.Logger
        Logger for logging messages.

    Returns
    -------
    tuple | None
        The loaded raster dataset or None if loading fails.
    """
    raster_file = list(raster_path.glob(file_pattern))
    if not raster_file:
        current_run.log_warning(f"Population raster not found: {raster_path}.")
        return None, None, None, None

    if len(raster_file) > 1:
        current_run.log_warning(
            f"Expected 1 file but found {len(raster_file)}: {raster_file}. Using first match.",
        )

    try:
        with rasterio.open(raster_file[0]) as src:
            raster = src.read(1)
            transform = src.transform  # affine
            crs = src.crs
            nodata = src.nodata
    except Exception as e:
        current_run.log_warning(f"Could not load population raster {raster_file[0]}. Error: {e}")
        return None, None, None, None

    current_run.log_info(logger, f"Population raster loaded: {raster_file[0]}.")
    return raster, transform, crs, nodata


def compute_total_populations(
    shapes: gpd.GeoDataFrame,
    data: np.ndarray,
    transform: Affine,
    crs: str,
    nodata: float,
) -> pl.DataFrame | None:
    """Compute total populations for given shapes using population data.

    Parameters
    ----------
    shapes : gpd.GeoDataFrame
        GeoDataFrame containing the shapes for zonal statistics.
    data : np.ndarray
        2D array of the population raster.
    transform : Affine
        Affine transform of the population raster.
    crs : str
        CRS of the population raster.
    nodata : float
        NoData value of the population raster.

    Returns
    -------
    pl.DataFrame
        DataFrame with ADM2_ID (Utf8) and total_population (Int64, nullable) columns.
    """
    if any(x is None for x in (shapes, data, crs)):
        return None

    # Ensure CRS matches the raster & reproject if necessary
    if shapes.crs is None:
        raise ValueError("Shapes GeoDataFrame must have a defined CRS.")
    # Reproject shapes if CRS is different (consistent to wpop pipeline calculation check)
    if shapes.crs.to_string() != crs:
        current_run.log_warning(
            f"The CRS data differs from the provided shapes file. Reprojecting shapes with {crs}",
        )
        shapes = shapes.to_crs(crs)

    # get statistics
    current_run.log_info(f"Computing ADM2 spatial aggregation for {len(shapes)} shapes.")
    pop_total = zonal_stats(
        vectors=shapes,
        raster=data,
        affine=transform,
        stats=["sum"],
        geojson_out=True,
        nodata=nodata,
    )
    result = pl.DataFrame(
        [
            {"ADM2_ID": f["properties"].get("ADM2_ID"), "total_population": f["properties"].get("sum")}
            for f in pop_total
        ],
        schema={"ADM2_ID": pl.Utf8, "total_population": pl.Float64},
    )

    return result.with_columns(
        pl.col("total_population").round(0).cast(pl.Int64, strict=False),
    )


def create_file_logger(log_path: Path, level: int = logging.INFO) -> logging.Logger:
    """Create a logger that writes messages to a file.

    Args:
        log_path: Path to the log file.
        level: Logging level (default INFO).

    Returns:
        Configured logger.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"map_extractor_{timestamp}.log"
    logger = logging.getLogger(str(log_file))  # unique name per file
    logger.setLevel(level)

    # Avoid adding multiple handlers if logger already exists
    if not logger.handlers:
        # Ensure parent folder exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Create file handler
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(level)

        # Optional: also log to console
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add handlers
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def get_extract_periods(start: str, end: str) -> list[str]:
    """Generates a list of periods between start and end.

    Returns
    -------
    list[str]
        List of periods as strings (e.g. "2020", "202501").
    """
    try:
        # Get periods
        p1 = period_from_string(start)
        p2 = period_from_string(end)
        periods = [p1] if p1 == p2 else p1.get_range(p2)
        return [str(p) for p in periods]
    except Exception as e:
        raise Exception(f"Error in start/end date configuration: {e!s}") from e


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
