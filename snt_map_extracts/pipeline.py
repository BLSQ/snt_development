from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from datetime import datetime

import logging
from openhexa.sdk import current_run, parameter, pipeline, workspace, File
import rasterio
from rasterio.warp import reproject, Resampling
from affine import Affine
from rasterstats import zonal_stats

# from owslib.wcs import WebCoverageService
from snt_lib.snt_pipeline_utils import (
    pull_scripts_from_repository,
    add_files_to_dataset,
    load_configuration_snt,
    run_report_notebook,
    get_file_from_dataset,
    validate_config,
)
from malariaAtlasProject.map import MAPRasterExtractor
from malariaAtlasProject.map_utils import (
    load_tiff_bands,
    parse_raster_filename_vars,
)

# Ticket:
# https://bluesquare.atlassian.net/browse/SNT25-143
# https://bluesquare.atlassian.net/browse/SNT25-259


@pipeline("snt_map_extracts")
@parameter(
    code="pop_raster_selection",
    name="Population raster selection (.tif)",
    type=File,
    help="Select the population raster (.tif) used for population-weighted calculations.",
    required=False,
    default=None,
)
@parameter(
    code="target_year",
    name="Target Year",
    help=(
        "Target year for indicator selection (e.g. 2022). Defaults to latest if unavailable or not specified."
    ),
    type=str,
    default="2020",  # None -------------------------------------------------------------------------------------------------------------------
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
def snt_map_extracts(
    pop_raster_selection: str, target_year: str, run_report_only: bool, pull_scripts: bool
) -> None:
    """Main function to get raster data for a dhis2 country."""
    root_path = Path(workspace.files_path)
    pipeline_path = root_path / "pipelines" / "snt_map_extracts"
    pipeline_path.mkdir(parents=True, exist_ok=True)
    logger = create_file_logger(log_path=pipeline_path / "logs")

    if pull_scripts:
        log_message(logger, "Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_map_extracts",
            report_scripts=["snt_map_extracts_report.ipynb"],
            code_scripts=[],
        )

    # NOTE: ZIP both names and code into a single named list (labels, indicator)
    try:
        # Load configuration
        snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
        validate_config(snt_config)
        country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
        dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get(
            "SNT_MAP_EXTRACT"
        )  # UPDATE THIS TO "SNT_MAP_EXTRACTS" ----------------------------------------------

        # MAP indicators
        snt_indicators = {
            "Malaria": {
                "Pf_Parasite_Rate",
                "Pf_Mortality_Rate",
            }
        }

        # snt_indicators = {
        #     "Malaria": {
        #         "Pf_Parasite_Rate",
        #         "Pf_Mortality_Rate",
        #         "Pf_Incidence_Rate",
        #     },
        #     "Interventions": {
        #         "Insecticide_Treated_Net_Access",
        #         "Insecticide_Treated_Net_Use_Rate",
        #         "IRS_Coverage",
        #         "Antimalarial_Effective_Treatment",
        #     },
        # }

        if not run_report_only:
            output_path = root_path / "data" / "map"
            output_path.mkdir(parents=True, exist_ok=True)

            # pop_raster_full_path = pop_raster_selection.path
            pop_raster_full_path = Path(
                r"C:\Users\blues\Desktop\Bluesquare\Repositories\snt_development\snt_map_extracts\workspace\data\worldpop\ner_pop_2025_CN_100m_R2025A_v1.tif"
            )
            log_message(logger, f"Population raster selected: {pop_raster_full_path}")
            if not pop_raster_full_path.exists():
                raise FileNotFoundError(f"Population raster file not found: {pop_raster_full_path}")
            if pop_raster_full_path.suffix.lower() != ".tif":
                raise ValueError("Population raster must be a .tif file.")

            make_table(
                coverage_categories=snt_indicators,
                snt_config=snt_config,
                pop_raster_path=pop_raster_full_path,
                target_year=target_year,
                output_path=output_path,
                logger=logger,
            )

            # add_files_to_dataset(
            #     dataset_id=dataset_id,
            #     country_code=country_code,
            #     file_paths=[
            #         output_path / "formatted" / country_code / f"{country_code}_map_data.parquet",
            #         output_path / "formatted" / country_code / f"{country_code}_map_data.csv",
            #     ],
            # )

        else:
            log_message(logger, "Skipping calculations, running only the reporting.")

        # run_report_notebook(
        #     nb_file=pipeline_path / "reporting" / "snt_map_extract_report.ipynb",
        #     nb_output_path=pipeline_path / "reporting" / "outputs",
        #     nb_parameters=None,
        # )

        log_message(logger, "Pipeline completed successfully!")

    except Exception as e:
        log_message(logger, f"Pipeline error: {e}")
        raise e


def make_table(
    coverage_categories: dict,
    snt_config: str,
    pop_raster_path: Path,
    target_year: str,
    output_path: Path,
    logger: logging.Logger,
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
    dataset_shapes_id = snt_config.get("SNT_DATASET_IDENTIFIERS", {}).get("DHIS2_DATASET_FORMATTED")
    shapes = get_file_from_dataset(dataset_shapes_id, f"{country_code}_shapes.geojson")
    log_message(logger, f"Shapes loaded from dataset: {dataset_shapes_id}.")

    # Check shapes
    invalid_shapes = shapes[shapes.geometry.isna()]
    if len(invalid_shapes) > 0:
        log_message(
            logger, f"Dropping {len(invalid_shapes)} organisation units without geometry.", level="warning"
        )
    shapes = shapes[shapes.geometry.notna()]

    if len(shapes) == 0:
        return

    rasters_path = output_path / "raster_files" / country_code
    rasters_path.mkdir(parents=True, exist_ok=True)

    raster_files = retrieve_rasters(
        coverage_categories=coverage_categories,
        target_year=target_year,
        shapes=shapes,
        logger=logger,
        rasters_path=rasters_path,
    )

    if len(raster_files) == 0:
        log_message(logger, "No raster files were downloaded. Exiting table generation.", level="warning")
        return

    run_aggregations(
        raster_files=raster_files,
        shapes=shapes,
        pop_raster_path=pop_raster_path,
        snt_config=snt_config,
        output_path=output_path / "formatted" / country_code,
        logger=logger,
    )


def retrieve_rasters(
    coverage_categories: dict,
    target_year: str,
    shapes: gpd.GeoDataFrame,
    logger: logging.Logger,
    rasters_path: Path,
) -> list[Path]:
    """Retrieve raster files for specified coverage categories and indicators.

    Returns:
        A list of paths to the downloaded raster files.
    """
    downloaded_rasters = []
    for category, indicators in coverage_categories.items():
        log_message(logger, f"Processing category: {category}.")
        map_extractor = MAPRasterExtractor(category=category, logger=logger)
        for indicator in indicators:
            try:
                log_message(logger, f"Downloading raster for indicator: {indicator}.")
                raster_path = map_extractor.download_indicator_raster(
                    indicator=indicator,
                    target_year=target_year,
                    shapes=shapes,
                    output_path=rasters_path,
                    replace_file=False,
                )
                downloaded_rasters.append(raster_path)
            except Exception as e:
                msg = f"Error downloading raster for {indicator}."
                log_message(logger, msg, level="error")
                logger.error(f"{msg} Error: {e}")
                continue

    return downloaded_rasters


def run_aggregations(
    raster_files: list[Path],
    shapes: gpd.GeoDataFrame,
    pop_raster_path: Path | None,
    snt_config: str,
    output_path: Path,
    logger: logging.Logger,
):
    """Run zonal statistics aggregations on the downloaded rasters."""
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")

    # 1. Load population raster (if available)
    if not pop_raster_path:
        log_message(logger, "Population raster file not provided.", level="warning")
    else:
        pop_data, pop_transform, pop_crs, pop_nodata = load_raw_population_raster(
            file_pattern=pop_raster_path.name,
            raster_path=pop_raster_path.parent,
            logger=logger,
        )
        pop_total = compute_total_populations(
            shapes, data=pop_data, transform=pop_transform, crs=pop_crs, nodata=pop_nodata, logger=logger
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

        log_message(logger, f"Computing {raster_file.name} statistics...")
        ref_columns = ["ADM1_NAME", "ADM1_ID", "ADM2_NAME", "ADM2_ID"]
        band_selection = ["Data", "LCI", "UCI"]
        stats_results = []

        # Compute Zonal Statistics per layer
        for band in bands:
            if band in band_selection:
                log_message(logger, f"Processing {file_vars['indicator']} band: {band}.")
                zstats = zonal_stats(
                    vectors=shapes,
                    raster=raster_data[band],
                    affine=raster_transform,
                    stats=["mean"],
                    geojson_out=True,
                    nodata=raster_nodata,
                )
                result_gdf = gpd.GeoDataFrame.from_features(zstats).drop(columns=["geometry"])
                metric_var = "MEAN" if band == "Data" else band
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

        if not all([s in bands for s in band_selection]):
            log_message(
                logger,
                f"One or more required bands missing in {file_vars['indicator']}. "
                f"Expected bands: {band_selection}, found bands: {bands}.",
                level="warning",
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
    log_message(logger, f"Output file saved under : {output_path / f'{country_code}_map_data.csv'}")


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
        log_message(logger, f"No population raster not found: {raster_path}.", level="warning")
        return None, None, None, None

    if len(raster_file) > 1:
        log_message(
            logger,
            f"Expected 1 file but found {len(raster_file)}: {raster_file}. Using first match.",
            level="warning",
        )

    try:
        with rasterio.open(raster_file[0]) as src:
            raster = src.read(1)
            transform = src.transform  # affine
            crs = src.crs
            nodata = src.nodata
        log_message(logger, f"Population raster loaded: {raster_file[0]}.")
        return raster, transform, crs, nodata
    except Exception as e:
        log_message(logger, f"Could not load population raster {raster_file[0]}", level="error")
        logger.error(f"Could not load population raster {raster_file[0]}. Error: {e}")
        return None, None, None, None


def compute_total_populations(
    shapes: gpd.GeoDataFrame,
    data: np.ndarray,
    transform: Affine,
    crs: str,
    nodata: float,
    logger: logging.Logger,
) -> pd.DataFrame:
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
    logger : logging.Logger
        Logger for logging messages.

    Returns
    -------
    pd.Series
        Series containing the total populations for each shape or None if data is unavailable.
    """
    if any(x is None for x in (shapes, data, crs)):
        return None

    # Ensure CRS matches the raster & reproject if necessary
    if shapes.crs is None:
        raise ValueError("Shapes GeoDataFrame must have a defined CRS.")
    # Reproject shapes if CRS is different (consistent to wpop pipeline calculation check)
    if shapes.crs.to_string() != crs:
        log_message(
            logger, f"The CRS data differs from the provided shapes file. Reprojecting shapes with {crs}"
        )
        shapes = shapes.to_crs(crs)

    # get statistics
    log_message(logger, f"Computing ADM2 spacial aggregation for {len(shapes)} shapes.")
    pop_total = zonal_stats(
        vectors=shapes,
        raster=data,
        affine=transform,
        stats=["sum"],
        geojson_out=True,
        nodata=nodata,
    )
    result = pd.DataFrame(
        [
            {"ADM2_ID": f["properties"].get("ADM2_ID"), "total_population": f["properties"]["sum"]}
            for f in pop_total
        ]
    )

    try:
        result["total_population"] = result["total_population"].round(0).astype(int)
    except Exception:
        log_message(
            logger,
            "Could not convert total_population to int, is possible that all results are None.",
            level="warning",
        )
    result["ADM2_ID"] = result["ADM2_ID"].astype(str)

    return result


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


def log_message(logger: logging.Logger, message: str, level: str = "info") -> None:
    """Log a message using self.logger and/or current_run."""
    if not level or not message:
        return

    level = level.lower()
    logger_methods = {
        "info": "info",
        "warning": "warning",
        "error": "error",
        "debug": "debug",
    }
    run_methods = {
        "info": "log_info",
        "warning": "log_warning",
        "error": "log_error",
        "debug": "log_debug",
    }

    if level not in logger_methods:
        raise ValueError(f"Unsupported logging level: {level}")

    # Log to standard logger
    if logger and hasattr(logger, logger_methods[level]):
        getattr(logger, logger_methods[level])(message)

    # Log to OpenHexa current_run
    if "current_run" in globals() and hasattr(current_run, run_methods[level]):
        getattr(current_run, run_methods[level])(message)


if __name__ == "__main__":
    snt_map_extracts()
