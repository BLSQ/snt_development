import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from openhexa.sdk import (
    current_run,
    parameter,
    pipeline,
    workspace,
)
from rasterstats import zonal_stats
from owslib.wcs import WebCoverageService
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    load_configuration_snt,
    run_report_notebook,
    get_file_from_dataset,
    validate_config,
)


@pipeline("snt_map_extract")
@parameter(
    "run_report_only",
    name="Run reporting only",
    help="This will only execute the reporting notebook",
    type=bool,
    default=False,
)
def snt_map_extract(run_report_only: bool) -> None:
    """Main function to get raster data for a dhis2 country."""
    root_path = Path(workspace.files_path)
    pipeline_path = root_path / "pipelines" / "snt_map_extract"
    pipeline_path.mkdir(parents=True, exist_ok=True)

    # NOTE: ZIP both names and code into a single named list (labels, indicator)
    try:
        # Load configuration
        snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
        validate_config(snt_config)

        country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
        dataset_shapes_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("DHIS2_DATASET_FORMATTED")
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

        # Get shapes
        shapes = get_file_from_dataset(dataset_shapes_id, f"{country_code}_shapes.geojson")
        current_run.log_info(f"Shapes loaded from dataset: {dataset_shapes_id}.")

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

            make_table(
                coverage_indicators=snt_indicators,
                country_code=country_code,
                shapes=shapes,  # type: ignore[reportArgumentType]
                level=org_level,
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


# Example malaria raster layer (adjust this to match exact layer name)
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
    wcs = WebCoverageService(url, version="2.0.1")
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
    country_code: str,
    shapes: gpd.GeoDataFrame,
    level: int,
    output_path: Path,
) -> pd.DataFrame:
    """Generate a table of zonal statistics for given coverage indicators and save the results.

    Parameters
    ----------
    coverage_indicators : dict
        Dictionary mapping categories to indicator layer names.
    country_code : str
        The country code used for naming output files.
    shapes : gpd.GeoDataFrame
        GeoDataFrame containing administrative boundaries.
    level : int
        Administrative level for processing.
    output_path : Path
        Directory where output files will be saved.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the processed zonal statistics.
    """
    found_indicators = filter_available_indicators(coverage_indicators)

    invalid_shapes = shapes[shapes.geometry.isna()]
    if len(invalid_shapes) > 0:
        current_run.log_warning(
            f"DHIS2 units with no geometry: {list(invalid_shapes[f'level_{level}_name'].unique())}"
        )
    shapes = shapes[shapes.geometry.notna()]
    if len(shapes) > 0:
        minx, miny, maxx, maxy = shapes.total_bounds
        rasters_path = output_path / "raster_files"
        rasters_path.mkdir(parents=True, exist_ok=True)

        download_raster_data(rasters_path, found_indicators, minx, maxx, miny, maxy)

        # Step 1: Load Admin Polygons
        final_df = pd.DataFrame()
        for category, layers in found_indicators.items():
            current_run.log_info(f"Processing {category}...")
            for indicator, layer_name in layers.items():
                version = layer_name.split("__")[1]
                zstats = zonal_stats(
                    shapes,
                    rasters_path / f"{layer_name}.tif",
                    stats=["mean"],
                    geojson_out=True,
                )

                # Convert to GeoDataFrame
                result_gdf = gpd.GeoDataFrame.from_features(zstats)

                # Step 4: Reshape to Output Format
                result_gdf["metric_category"] = category
                result_gdf["metric_name"] = indicator
                metric_columns = ["mean"]
                ref_columns = {col for col in result_gdf.columns if col not in metric_columns}

                # Melt to long format
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
                melt_df = melt_df[np.isfinite(melt_df["value"])]
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


if __name__ == "__main__":
    snt_map_extract()
