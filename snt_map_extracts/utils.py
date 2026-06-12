from openhexa.toolbox.dhis2.periods import period_from_string
from openhexa.sdk import current_run
import geopandas as gpd
import numpy as np
from pathlib import Path
import polars as pl
from rasterstats import zonal_stats
from pyproj import CRS
from affine import Affine
import rasterio
from rasterio.warp import Resampling, reproject


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


def compute_total_populations(
    shapes: gpd.GeoDataFrame,
    data: np.ndarray,
    transform: Affine,
    crs: rasterio.crs.CRS,
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
    crs : rasterio.crs.CRS
        CRS of the population raster.
    nodata : float
        NoData value of the population raster.

    Returns
    -------
    pl.DataFrame
        DataFrame with ADM2_ID (Utf8) and total_population (Int64, nullable) columns.
    """
    if any(x is None for x in (shapes, data, crs)):
        current_run.log_warning("Total population computation skipped due to missing data or shapes.")
        return None

    # Ensure CRS matches the raster & reproject if necessary
    if shapes.crs is None:
        raise ValueError("Shapes GeoDataFrame must have a defined CRS.")

    # Reproject shapes if CRS is different (consistent to wpop pipeline calculation check)
    if shapes.crs != CRS.from_user_input(crs):
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

    return result.with_columns(pl.col("total_population").round(0).cast(pl.Int64, strict=False))


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
    if (crs != reference_crs) or (data.shape != reference_shape) or (transform != reference_transform):
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
    population_totals: pl.DataFrame | None,
    shapes: gpd.GeoDataFrame,
    indicator: str,
) -> pl.DataFrame | None:
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
    population_totals:
        pl.DataFrame containing total populations for each shape.
        If None, an empty column will be added to the final table with null values.
    shapes : gpd.GeoDataFrame
        GeoDataFrame containing the shapes for zonal statistics.
    indicator : str
        Name of the indicator being processed.

    Returns
    -------
    pl.DataFrame | None
        DataFrame with ADM2_ID, weighted_sum, total_population, and population_weighted columns.
    """
    if any(
        x is None
        for x in (shapes, metric_data, metric_transform, metric_crs, pop_data, pop_transform, pop_crs)
    ):
        current_run.log_warning(f"Population-weighted computation skipped for metric: {indicator}.")
        return None

    current_run.log_info(f"Computing population-weighted for metric: {indicator}.")
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
    result_w = pl.DataFrame(
        [
            {
                "ADM2_ID": f["properties"].get("ADM2_ID"),
                "weighted_sum": f["properties"].get("sum"),
            }
            for f in zstats_w
        ]
    ).with_columns(
        pl.col("ADM2_ID").cast(pl.Utf8),
        pl.col("weighted_sum").cast(pl.Float64),
    )

    if population_totals is None or population_totals.shape[0] == 0:
        current_run.log_warning(
            "Population totals not available. Population-weighted metric will set to null values."
        )
        return result_w.with_columns(
            pl.lit(None).cast(pl.Float64).alias("total_population"),
            pl.lit(None).cast(pl.Float64).alias("population_weighted"),
        )

    return result_w.join(population_totals, on="ADM2_ID", how="left").with_columns(
        (pl.col("weighted_sum") / pl.col("total_population")).alias("population_weighted")
    )


def load_raw_population_raster(raster_path: Path) -> tuple:
    """Load raw population raster from the specified path.

    Parameters
    ----------
    raster_path : Path
        Path to the population raster file.

    Returns
    -------
    tuple | None
        The loaded raster dataset or None if loading fails.
    """
    if not (raster_path).exists():
        current_run.log_warning(f"Population raster not found: {raster_path}.")
        return None, None, None, None

    try:
        with rasterio.open(raster_path) as src:
            raster = src.read(1)
            transform = src.transform  # affine
            crs = src.crs
            nodata = src.nodata
    except Exception as e:
        current_run.log_warning(f"Could not load population raster {raster_path}. Error: {e}")
        return None, None, None, None

    return raster, transform, crs, nodata


def generate_population_table_from_raster(raster_path: Path, shapes: gpd.GeoDataFrame) -> pl.DataFrame | None:
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

    if pop_crs is None:
        current_run.log_warning(f"Raster {raster_path} has no CRS defined, skipping population computation.")
        return None

    # Compute total populations for each shape using zonal statistics
    return compute_total_populations(
        shapes=shapes,
        data=pop_data,
        transform=pop_transform,
        crs=pop_crs,
        nodata=pop_nodata,
    )
