import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from openhexa.sdk import current_run, parameter, pipeline, workspace, File
import rasterio
from rasterio.warp import reproject, Resampling
from affine import Affine
from rasterstats import zonal_stats
from owslib.wcs import WebCoverageService
from snt_lib.snt_pipeline_utils import (
    pull_scripts_from_repository,
    add_files_to_dataset,
    load_configuration_snt,
    run_report_notebook,
    get_file_from_dataset,
    validate_config,
)

# Ticket: https://bluesquare.atlassian.net/browse/SNT25-143


@pipeline("snt_map_extract")
@parameter(
    code="pop_raster_selection",
    name="Population raster selection (.tif)",
    type=File,
    help="Select the population raster (.tif) used for population-weighted calculations.",
    required=False,
    default=None,
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
def snt_map_extract(pop_raster_selection: str, run_report_only: bool, pull_scripts: bool) -> None:
    """Main function to get raster data for a dhis2 country."""
    root_path = Path(workspace.files_path)
    pipeline_path = root_path / "pipelines" / "snt_map_extract"
    pipeline_path.mkdir(parents=True, exist_ok=True)

    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_map_extract",
            report_scripts=["snt_map_extract_report.ipynb"],
            code_scripts=[],
        )

    # NOTE: ZIP both names and code into a single named list (labels, indicator)
    try:
        # Load configuration
        snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
        validate_config(snt_config)
        country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
        dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("SNT_MAP_EXTRACT")

        # get org unit level
        # TODO: move this validation to validate_config() ?
        admin_level_2 = snt_config["SNT_CONFIG"].get("DHIS2_ADMINISTRATION_2")
        match = re.search(r"\d+", admin_level_2)
        if match:
            org_level = int(match.group())
        else:
            raise ValueError(
                f"Invalid DHIS2_ADMINISTRATION_2 "
                f"format expected: 'level_NUMBER_name' received: {admin_level_2}"
            )

        # MAP indicators
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

        if not run_report_only:
            output_path = root_path / "data" / "map"
            output_path.mkdir(parents=True, exist_ok=True)

            if pop_raster_selection is None:
                raster_fname = f"{country_code}_worldpop_ppp_*_UNadj.tif"
                raster_path = root_path / "data" / "worldpop" / "raw"
            else:
                if not pop_raster_selection.name.lower().endswith(".tif"):
                    raise ValueError("Population raster must be a .tif file.")
                raster_fname = pop_raster_selection.name
                raster_path = Path(pop_raster_selection.path).parent

            make_table(
                coverage_indicators=snt_indicators,
                snt_config=snt_config,
                level=org_level,
                raster_path=raster_path,
                raster_fname=raster_fname,
                output_path=output_path,
            )

            add_files_to_dataset(
                dataset_id=dataset_id,
                country_code=country_code,
                file_paths=[
                    output_path / "formatted" / f"{country_code}_map_data.parquet",
                    output_path / "formatted" / f"{country_code}_map_data.csv",
                ],
            )

        else:
            current_run.log_info("Skipping calculations, running only the reporting.")

        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_map_extract_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            nb_parameters=None,
        )

        current_run.log_info("Pipeline completed successfully!")

    except Exception as e:
        current_run.log_error(f"Pipeline error: {e}")
        raise e


def download_raster_data(
    output_path: Path,
    mapping_coverage_indicators: dict,
    minx: float,
    maxx: float,
    miny: float,
    maxy: float,
    geoserver_url: str = "https://data.malariaatlas.org/geoserver/",
):
    """Download raster data for specified coverage indicators and bounding box.

    Parameters
    ----------
    output_path : Path
        The directory where raster files will be saved.
    mapping_coverage_indicators : dict
        Dictionary mapping categories to indicator layer names.
    minx : float
        Minimum longitude of the bounding box.
    maxx : float
        Maximum longitude of the bounding box.
    miny : float
        Minimum latitude of the bounding box.
    maxy : float
        Maximum latitude of the bounding box.
    geoserver_url : str, optional
        The base URL of the GeoServer WCS service (default: "https://data.malariaatlas.org/geoserver/").
    """
    # Example malaria raster layer (adjust this to match exact layer name)
    for category, layers in mapping_coverage_indicators.items():
        url = f"{geoserver_url}{category}/wcs"
        for _, layer_name in layers.items():
            params = {
                "service": "WCS",
                "version": "2.0.1",
                "request": "GetCoverage",
                "coverageId": layer_name,
                "format": "image/tiff",
                "subset": [
                    f"Long({minx},{maxx})",
                    f"Lat({miny},{maxy})",
                ],
            }
            raster_filename = output_path / f"{layer_name}.tif"
            if not raster_filename.exists():
                current_run.log_info(f"Downloading raster for :{layer_name}")
                r = requests.get(url, params=params, stream=True)
                if r.status_code != 200:
                    current_run.log_warning(f"Error downloading raster: {r.status_code}")
                    continue
                with raster_filename.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                current_run.log_info(f"Raster for {layer_name} already downloaded.")


def build_latest_map(category: str) -> dict:
    """Retrieve the latest map coverage IDs for a given category from the malaria atlas WCS service.

    Parameters
    ----------
    category : str
        The category of the map layers (e.g., "Malaria" or "Interventions").

    Returns
    -------
    dict
        A dictionary mapping layer keys to a tuple of (coverage ID, date string).
    """
    url = f"https://data.malariaatlas.org/geoserver/{category}/wcs?service=WCS&request=GetCapabilities"
    wcs = WebCoverageService(url, version="2.0.1", timeout=90)
    latest_map = {}

    for cov_id in wcs.contents:
        parts = cov_id.split("__", 2)
        prefix = parts[0]
        date_str, suffix = parts[1].split("_", 1)
        key = f"{prefix}__{suffix}".replace("Global_", "").replace("Africa_", "")
        # if we already have one, check which date is newer
        if key in latest_map:
            _, existing_date = latest_map[key]
            if date_str > existing_date:
                latest_map[key] = (cov_id, date_str)
        else:
            latest_map[key] = (cov_id, date_str)

    return latest_map


def make_table(
    coverage_indicators: dict,
    snt_config: str,
    level: int,
    raster_path: Path,
    raster_fname: str,
    output_path: Path,
) -> pd.DataFrame:
    """Generate a table of zonal statistics for given coverage indicators and save the results.

    Parameters
    ----------
    coverage_indicators : dict
        Dictionary mapping categories to indicator layer names.
    snt_config : str
        SNT configuration file.
    level : int
        Administrative level for processing.
    raster_path : Path
        Path to the selecte raster directory.
    raster_fname : str
        Filename pattern for the population raster.
    output_path : Path
        Directory where output files will be saved.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the processed zonal statistics.
    """
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    dataset_shapes_id = snt_config.get("SNT_DATASET_IDENTIFIERS", {}).get("DHIS2_DATASET_FORMATTED")
    shapes = get_file_from_dataset(dataset_shapes_id, f"{country_code}_shapes.geojson")
    current_run.log_info(f"Shapes loaded from dataset: {dataset_shapes_id}.")

    invalid_shapes = shapes[shapes.geometry.isna()]
    if len(invalid_shapes) > 0:
        current_run.log_warning(
            f"DHIS2 units with no geometry: {list(invalid_shapes[f'level_{level}_name'].unique())}"
        )
    shapes = shapes[shapes.geometry.notna()]

    found_indicators = filter_available_indicators(coverage_indicators)
    if len(shapes) > 0:
        minx, miny, maxx, maxy = shapes.total_bounds
        rasters_path = output_path / "raster_files"
        rasters_path.mkdir(parents=True, exist_ok=True)

        download_raster_data(rasters_path, found_indicators, minx, maxx, miny, maxy)
        pop_data, pop_transform, pop_crs, pop_nodata = load_raw_population_raster(
            file_pattern=raster_fname,
            raster_path=raster_path,
        )
        pop_total = compute_total_populations(
            shapes, data=pop_data, transform=pop_transform, crs=pop_crs, nodata=pop_nodata
        )

        # Set nodata to np.nan
        if pop_data is not None:
            pop_data = pop_data.astype(float)
            pop_data[pop_data == pop_nodata] = np.nan

        # Step 1: Load Admin Polygons
        final_df = pd.DataFrame()
        for category, layers in found_indicators.items():
            current_run.log_info(f"Processing {category}.")
            for indicator, layer_name in layers.items():
                version = layer_name.split("__")[1]
                metric_columns = ["mean"]

                with rasterio.open(rasters_path / f"{layer_name}.tif") as src:
                    metric = src.read(1)
                    metric_transform = src.transform
                    metric_crs = src.crs
                    metric_nodata = src.nodata

                current_run.log_info(f"Computing {layer_name} statistics...")
                # Compute Zonal Statistics
                zstats = zonal_stats(
                    vectors=shapes,
                    raster=metric,
                    affine=metric_transform,
                    stats=metric_columns,
                    geojson_out=True,
                    nodata=metric_nodata,
                )

                # Format results
                result_gdf = gpd.GeoDataFrame.from_features(zstats)
                result_gdf["metric_category"] = category
                result_gdf["metric_name"] = indicator
                ref_columns = {col for col in result_gdf.columns if col not in metric_columns}

                # Melt to long format (could be several metrics)
                melt_df = result_gdf.melt(
                    id_vars=list(ref_columns),
                    value_vars=metric_columns,
                    var_name="statistic",
                    value_name="value",
                )
                melt_df["version"] = version
                melt_df["period"] = version.split("_")[0]
                melt_df["YEAR"] = melt_df["period"].str[:4].astype(int)
                melt_df["MONTH"] = melt_df["period"].str[4:].astype(int)
                melt_df["value"] = pd.to_numeric(melt_df["value"], errors="coerce")
                # melt_df = melt_df[np.isfinite(melt_df["value"])]
                melt_df = melt_df.drop(columns=["geometry"])

                # Compute population-weighted metric (extra column)
                weighted_metric = compute_population_weighted_metric(
                    metric_data=metric,
                    metric_transform=metric_transform,
                    metric_crs=metric_crs,
                    metric_nodata=metric_nodata,
                    pop_data=pop_data,
                    pop_transform=pop_transform,
                    pop_crs=pop_crs,
                    total_population=pop_total,
                    shapes=shapes,
                    indicator=indicator,
                )

                if weighted_metric is None:
                    melt_df["population_weighted"] = None  # default if not computed
                else:
                    melt_df = melt_df.merge(
                        weighted_metric[["ADM2_ID", "population_weighted"]],
                        on="ADM2_ID",
                        how="left",
                    )

                # merge final table
                final_df = pd.concat([final_df, melt_df], ignore_index=True)

        # Step 5: Save Output
        formatted_path = output_path / "formatted"
        formatted_path.mkdir(parents=True, exist_ok=True)

        # SNT format
        final_df.columns = [col.strip().upper() for col in final_df.columns]
        final_df["METRIC_NAME"] = final_df["METRIC_NAME"].str.strip()

        # Save file
        final_df.to_parquet(formatted_path / f"{country_code}_map_data.parquet", index=False)
        final_df.to_csv(formatted_path / f"{country_code}_map_data.csv", index=False)
        current_run.log_info(f"Output file saved under : {formatted_path / f'{country_code}_map_data.csv'}")

    return final_df


def load_raw_population_raster(file_pattern: str, raster_path: Path) -> tuple:
    """Load raw population raster from the specified path.

    Parameters
    ----------
    file_pattern : str
        Pattern to match the population raster file.
    raster_path : Path
        Path to the population raster file.

    Returns
    -------
    tuple | None
        The loaded raster dataset or None if loading fails.
    """
    raster_file = list(raster_path.glob(file_pattern))
    if not raster_file:
        current_run.log_warning(f"No population raster not found: {raster_path}.")
        return None, None, None, None

    if len(raster_file) > 1:
        current_run.log_warning(
            f"Expected 1 file but found {len(raster_file)}: {raster_file}. Using first match."
        )

    try:
        with rasterio.open(raster_file[0]) as src:
            raster = src.read(1)
            transform = src.transform  # affine
            crs = src.crs
            nodata = src.nodata
        current_run.log_info(f"Population raster loaded: {raster_file[0]}.")
        return raster, transform, crs, nodata
    except Exception as e:
        current_run.log_warning(f"Could not load population raster {raster_file[0]}. Error: {e}")
        return None, None, None, None


def filter_available_indicators(indicators: dict) -> dict:
    """Filter and retrieve available indicator coverage IDs for specified categories.

    Parameters
    ----------
    indicators : dict
        Dictionary mapping categories to lists of indicator names.

    Returns
    -------
    dict
        Dictionary mapping categories to available indicator coverage IDs.
    """
    filtered_indicators = {}
    available_indicators = {}
    for category in indicators:
        available_indicators[category] = build_latest_map(category)

    for category, keys in indicators.items():
        result = {}
        for key in keys:
            full_key = f"{category}__{key}"
            if full_key in available_indicators[category]:
                result[key] = available_indicators[category][full_key][0]
            else:
                current_run.log_warning(
                    f"Coverage indicator {full_key} not found in available indicators for {category}."
                )
        filtered_indicators[category] = result
    return filtered_indicators


def compute_total_populations(
    shapes: gpd.GeoDataFrame,
    data: np.ndarray,
    transform: Affine,
    crs: str,
    nodata: float,
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
        current_run.log_info(
            f"The CRS data differs from the provided shapes file. Reprojecting shapes with {crs}"
        )
        shapes = shapes.to_crs(crs)

    # get statistics
    current_run.log_info(f"Computing ADM2 spacial aggregation for {len(shapes)} shapes.")
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
    result["total_population"] = result["total_population"].round(0).astype(int)
    result["ADM2_ID"] = result["ADM2_ID"].astype(str)
    return result


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
    metric_resolution : tuple
        Resolution of the metric raster.
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

    Returns
    -------
    pd.Series
        Series containing the weighted metric values for each shape or None if population data is unavailable.
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
    result_w = pd.DataFrame(
        [
            {"ADM2_ID": f["properties"].get("ADM2_ID"), "weighted_sum": f["properties"]["sum"]}
            for f in zstats_w
        ]
    )
    result_w["ADM2_ID"] = result_w["ADM2_ID"].astype(str)
    result = result_w.merge(total_population, on="ADM2_ID", how="left")
    result["population_weighted"] = result["weighted_sum"] / result["total_population"]
    return result


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


if __name__ == "__main__":
    snt_map_extract()
