from openhexa.toolbox.dhis2.periods import period_from_string
from openhexa.sdk import current_run
import geopandas as gpd
import numpy as np
from pathlib import Path
import polars as pl
from pyproj import CRS
from affine import Affine
import rasterio
from rasterio.warp import Resampling, reproject
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource
from rasterio.transform import array_bounds


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

    Uses exact_extract (fractional pixel-coverage weighting), matching the same engine used
    elsewhere in this pipeline, instead of rasterstats' binary in/out inclusion rule.

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
    xmin, ymin, xmax, ymax = array_bounds(data.shape[0], data.shape[1], transform)
    pop_source = NumPyRasterSource(data, xmin, ymin, xmax, ymax, nodata=nodata, srs_wkt=crs.to_wkt())

    pop_total = exact_extract(
        pop_source,
        shapes,
        ["sum"],
        include_cols=["ADM2_ID"],
        output="pandas",
    ).rename(columns={"sum": "total_population"})

    result = pl.from_pandas(pop_total).with_columns(pl.col("ADM2_ID").cast(pl.Utf8))

    return result.with_columns(pl.col("total_population").round(0).cast(pl.Int64, strict=False))


def compute_population_weighted_metric(
    metric_data: np.ndarray,
    metric_transform: Affine,
    metric_crs: str,
    metric_nodata: float,
    pop_data: np.ndarray,
    pop_transform: Affine,
    pop_crs: str,
    shapes: gpd.GeoDataFrame,
    indicator: str,
    statistic: str = "mean",
) -> pl.DataFrame | None:
    """Compute the population-weighted statistic for given shapes using population data.

    Mirrors sntutils::process_weighted_raster_collection() (per its R source: population is
    reprojected/resampled onto the value raster's grid via terra::resample(method="bilinear"),
    with NA left unfilled through the resample and only replaced by the default weight (0,
    weight_na_as_zero=TRUE) at extraction time).

    For statistic="mean", exact_extract's weighted_mean/weighted_sum ops are used directly.
    For statistic="median", per-pixel values/coverage/weights are pulled instead and combined
    via weighted_median(), since exact_extract has no built-in weighted-median op (confirmed:
    passing weights= alongside ops=["median"] silently ignores the weights raster entirely).

    total_population is derived from the same aligned grid used for the statistic itself, for
    numerator/denominator consistency (population_totals is no longer a separate input).

    Parameters
    ----------
    metric_data : np.ndarray
        2D array of the metric raster.
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
    shapes : gpd.GeoDataFrame
        GeoDataFrame containing the shapes for zonal statistics.
    indicator : str
        Name of the indicator being processed.
    statistic : str
        Either "mean" or "median".

    Returns
    -------
    pl.DataFrame | None
        DataFrame with ADM2_ID, total_population, and population_weighted columns
        (plus weighted_sum for statistic="mean").
    """
    if statistic not in ("mean", "median"):
        raise ValueError(f"Unsupported statistic for population weighting: {statistic!r}")

    if any(
        x is None
        for x in (shapes, metric_data, metric_transform, metric_crs, pop_data, pop_transform, pop_crs)
    ):
        current_run.log_warning(f"Population-weighted computation skipped for metric: {indicator}.")
        return None

    current_run.log_info(f"Computing population-weighted {statistic} for metric: {indicator}.")

    # sntutils resamples population onto the value raster's grid via bilinear interpolation,
    # with NA preserved through the resample (GDAL/terra exclude nodata pixels from the
    # interpolation kernel rather than treating them as 0) and only the *remaining* NA cells
    # replaced by the default weight afterward. Pre-filling NaN with 0 before a bilinear
    # resample would incorrectly drag down interpolated values near nodata boundaries.
    pop_aligned = np.full(metric_data.shape, np.nan, dtype=float)
    reproject(
        source=pop_data,
        destination=pop_aligned,
        src_transform=pop_transform,
        src_crs=pop_crs,
        src_nodata=np.nan,
        dst_transform=metric_transform,
        dst_crs=metric_crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    # default_weight (weight_na_as_zero=TRUE in sntutils): fill any still-missing population
    # cells with 0 only now, after resampling.
    pop_aligned = np.where(np.isnan(pop_aligned), 0.0, pop_aligned)

    xmin, ymin, xmax, ymax = array_bounds(metric_data.shape[0], metric_data.shape[1], metric_transform)
    value_source = NumPyRasterSource(
        metric_data, xmin, ymin, xmax, ymax, nodata=metric_nodata, srs_wkt=metric_crs.to_wkt()
    )
    weight_source = NumPyRasterSource(pop_aligned, xmin, ymin, xmax, ymax, srs_wkt=metric_crs.to_wkt())

    # Total population computed on the same aligned grid used for the statistic itself, so
    # numerator and denominator share identical coverage fractions (unlike the previous
    # implementation, which sourced total_population from a separately-computed, unaligned sum).
    total_population = exact_extract(
        weight_source,
        shapes,
        ["sum"],
        include_cols=["ADM2_ID"],
        output="pandas",
    ).rename(columns={"sum": "total_population"})

    if statistic == "mean":
        weighted = exact_extract(
            value_source,
            shapes,
            ["weighted_mean", "weighted_sum"],
            weights=weight_source,
            include_cols=["ADM2_ID"],
            output="pandas",
        ).rename(columns={"weighted_mean": "population_weighted"})
    else:
        per_pixel = exact_extract(
            value_source,
            shapes,
            ["values", "coverage", "weights"],
            weights=weight_source,
            include_cols=["ADM2_ID"],
            output="pandas",
        )
        records = [
            {
                "ADM2_ID": row["ADM2_ID"],
                "population_weighted": weighted_median(
                    row["values"], np.asarray(row["coverage"]) * np.asarray(row["weights"])
                ),
            }
            for _, row in per_pixel.iterrows()
        ]
        weighted = pl.DataFrame(records).to_pandas()

    result = pl.from_pandas(weighted).join(pl.from_pandas(total_population), on="ADM2_ID", how="left")

    cast_cols = ["ADM2_ID", "total_population", "population_weighted"]
    if "weighted_sum" in result.columns:
        cast_cols.append("weighted_sum")

    return result.with_columns(
        [
            pl.col(c).cast(pl.Utf8) if c == "ADM2_ID" else pl.col(c).cast(pl.Float64, strict=False)
            for c in cast_cols
        ]
    )


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median matching R's matrixStats::weightedMedian() default behavior.

    Replicates the package's default `interpolate=TRUE` path (ties=NULL, which the R
    wrapper maps to interpolate=TRUE): a "smoothed" cumulative distribution is built by
    shifting each sorted element's cumulative weight back by half its own weight, then
    linearly interpolating between the two elements straddling the 0.5 crossing point.
    exact_extract has no built-in weighted-median op, so this operates on the raw
    per-pixel values/weights pulled from it.

    Args:
        values: Raw pixel values (already restricted to pixels with positive weight/coverage).
        weights: Per-pixel weights (e.g. coverage_fraction * population), same length as values.

    Returns:
        The weighted median, or NaN if no valid (finite value, positive weight) pixels remain.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]

    if values.size == 0:
        return float("nan")

    order = np.argsort(values, kind="stable")
    values = values[order]
    w_norm = weights[order] / weights.sum()

    # Shifted cumulative weight: each element's mass is centered on its own interval.
    wcum = np.cumsum(w_norm) - w_norm / 2
    half = int(np.searchsorted(wcum, 0.5))
    half = min(half, values.size - 1)

    if half == 0:
        return float(values[0])

    span = wcum[half] - wcum[half - 1]
    if span == 0:
        return float(values[half])

    frac = (0.5 - wcum[half]) / span
    return float(values[half] + frac * (values[half] - values[half - 1]))


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
