from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from openhexa.sdk import current_run, parameter, pipeline, workspace
from rasterstats import zonal_stats
from snt_lib.snt_pipeline_utils import (
    pull_scripts_from_repository,
    add_files_to_dataset,
    get_file_from_dataset,
    load_configuration_snt,
    run_report_notebook,
    validate_config,
)
from worlpopclient import WorldPopClient


@pipeline("snt_worldpop_extract")
@parameter(
    "un_adj",
    name="UN adjusted",
    help="Retrieve UN adjusted population",
    type=bool,
    default=True,
    required=False,
)
@parameter(
    "overwrite",
    name="Overwrite",
    help="Overwrite existing population files",
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
def snt_worldpop_extract(un_adj: bool = True, overwrite: bool = False, pull_scripts: bool = False) -> None:
    """Write your pipeline orchestration here."""
    # set paths
    snt_root_path = Path(workspace.files_path)
    pipeline_path = snt_root_path / "pipelines" / "snt_worldpop_extract"
    year = "2020"  # Year data available in WorldPop

    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_worldpop_extract",
            report_scripts=["snt_worldpop_extract_report.ipynb"],
            code_scripts=[],
        )

    try:
        # get configuration
        snt_config_dict = load_configuration_snt(
            config_path=snt_root_path / "configuration" / "SNT_config.json"
        )
        validate_config(snt_config_dict)

        # get country identifier for file naming
        country_code = snt_config_dict["SNT_CONFIG"].get("COUNTRY_CODE")

        # Set output directory
        pop_file_path = retrieve_population_data(
            country_code=country_code,
            output_path=snt_root_path / "data" / "worldpop" / "raw",
            year=year,
            un_adj=un_adj,
            overwrite=overwrite,
        )

        run_spatial_aggregation(
            tif_file_path=pop_file_path,
            snt_config=snt_config_dict,
            output_dir=snt_root_path / "data" / "worldpop" / "population",
            year=year,  # just to add a year column in the output files
        )

        add_files_to_dataset(
            dataset_id=snt_config_dict["SNT_DATASET_IDENTIFIERS"].get("WORLDPOP_DATASET_EXTRACT"),
            country_code=country_code,
            file_paths=[
                snt_root_path / "data" / "worldpop" / "population" / f"{pop_file_path.stem}.csv",
                snt_root_path / "data" / "worldpop" / "population" / f"{pop_file_path.stem}.parquet",
            ],
        )

        # Run report notebook
        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_worldpop_extract_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            nb_parameters=None,
        )
    except Exception as e:
        current_run.log_error(f"An error occurred in the pipeline: {e}")
        raise


def retrieve_population_data(
    country_code: str, output_path: Path, year: str = "2020", un_adj: bool = False, overwrite: bool = False
) -> Path:
    """Retrieve raster population data from worldpop.

    Parameters
    ----------
    country_code : str
        The 3-letter ISO code of the country (e.g., "COD", "BFA").
    output_path : Path
        The directory where the population data will be saved.
    year : str, optional
        The year for which to retrieve the population data. Defaults to "2020".
    un_adj : bool, optional
        Whether to retrieve UN adjusted data. Defaults to False.
    overwrite : bool, optional
        Whether to overwrite existing files. Defaults to False.

    Returns
    -------
    Path
        The path to the saved WorldPop population data file.
    """
    current_run.log_info("Retrieving population data grid from worldpop.")
    wpop_client = WorldPopClient()
    current_run.log_info(f"Downloading data from : {wpop_client.base_url}")

    # Create output directory if it doesn't exist
    Path.mkdir(output_path, exist_ok=True)
    country = country_code.upper()
    un_adj_suffix = "_UN_adj" if un_adj else ""
    filename = f"{country}_worldpop_population_{year}{un_adj_suffix}.tif"

    if not overwrite:
        if (output_path / filename).exists():
            current_run.log_info(f"File {filename} already exists. Skipping download.")
            return output_path / filename
    try:
        current_run.log_info(
            f"Retrieving data for country: {country} - year: {year} - UN adjusted: {un_adj}."
        )
        pop_file_path = wpop_client.download_data_for_country(
            country_iso3=country,
            year=year,
            un_adj=un_adj,
            output_dir=output_path,
            fname=filename,
        )
        current_run.log_info(f"Population data successfully downloaded under : {pop_file_path}.")
        return pop_file_path
    except Exception as e:
        raise Exception(f"Error retrieving WorldPop population data for {country} {year}: {e}") from e


def run_spatial_aggregation(
    tif_file_path: Path, snt_config: dict, output_dir: Path, year: int = 2020
) -> None:
    """Run spatial aggregation on the worldpop population data (tif file)."""
    current_run.log_info(f"Running spatial aggregation with WorldPop data {tif_file_path}")

    if not tif_file_path.exists():
        raise FileNotFoundError(f"WorldPop file not found: {tif_file_path}")

    # Load DHIS2 shapes
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    shapes = get_file_from_dataset(
        dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"].get("DHIS2_DATASET_FORMATTED"),
        filename=f"{country_code}_shapes.geojson",
    )

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
    result_pd = pd.DataFrame(result_gdf)
    result_pd = result_pd.rename(columns={"sum": "population", "count": "pixel_count"})
    result_pd["population"] = result_pd["population"].round(0).astype(int)
    result_pd["YEAR"] = year  # Add year column (reference)
    result_pd.columns = result_pd.columns.str.upper()

    # Log any administrative levels with no population data
    no_data = result_pd[result_pd["POPULATION"] == 0]
    if not no_data.empty:
        for _, row in no_data.iterrows():
            current_run.log_warning(
                f"Administrative level 2 : {row['ADM2_NAME']} ({row['ADM2_ID']}) has no population data."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    result_pd.to_csv(output_dir / f"{tif_file_path.stem}.csv", index=False)
    result_pd.to_parquet(output_dir / f"{tif_file_path.stem}.parquet", index=False)
    current_run.log_info(f"Population data saved to {output_dir / f'{tif_file_path.stem}.csv'}")


if __name__ == "__main__":
    snt_worldpop_extract()
