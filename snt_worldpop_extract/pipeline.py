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
    save_pipeline_parameters,
)
from worlpopclient import WorldPopClient


@pipeline("snt_worldpop_extract")
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
@parameter(
    "year",
    name="Year",
    help="Year of population data to extract from WorldPop",
    type=int,
    default=2020,
    required=False,
)
def snt_worldpop_extract(overwrite: bool = False, pull_scripts: bool = False, year: int = 2020) -> None:
    """Write your pipeline orchestration here."""
    # set paths
    snt_root_path = Path(workspace.files_path)
    pipeline_path = snt_root_path / "pipelines" / "snt_worldpop_extract"
    data_path = snt_root_path / "data" / "worldpop"
    year_str = str(year)  # Convert to string for file naming

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
        retrieve_population_data(
            country_code=country_code,
            output_path=data_path / "raw",
            year=year_str,
            overwrite=overwrite,
        )

        run_spacial_aggregations(
            snt_config=snt_config_dict,
            input_dir=data_path / "raw",
            output_dir=data_path / "aggregations",
            year=year_str,  # just to add a year column in the output files
        )

        snt_worldpop_format(
            snt_config=snt_config_dict,
            year=year,
            input_dir=data_path / "aggregations",
            output_dir=data_path / "population",
        )

        parameters_file = save_pipeline_parameters(
            pipeline_name="snt_worldpop_extract",
            parameters={"overwrite": overwrite, "year": year, "pull_scripts": pull_scripts},
            output_path=data_path,
            country_code=country_code,
        )

        files_to_publish = [
            data_path / "population" / f"{country_code}_worldpop_population.csv",
            data_path / "population" / f"{country_code}_worldpop_population.parquet",
            data_path / "raw" / f"{country_code}_worldpop_ppp_{year}.tif",
        ]
        pop_unadj_tif = data_path / "raw" / f"{country_code}_worldpop_ppp_{year}_UNadj.tif"
        if pop_unadj_tif.exists():
            files_to_publish.append(pop_unadj_tif)
        files_to_publish.append(parameters_file)

        add_files_to_dataset(
            dataset_id=snt_config_dict["SNT_DATASET_IDENTIFIERS"].get("WORLDPOP_DATASET_EXTRACT"),
            country_code=country_code,
            file_paths=files_to_publish,
        )

        # Run report notebook
        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_worldpop_extract_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            country_code=country_code,
        )
    except Exception as e:
        current_run.log_error(f"An error occurred in the pipeline: {e}")
        raise


def retrieve_population_data(
    country_code: str, output_path: Path, year: str = "2020", overwrite: bool = False
) -> None:
    """Retrieve raster population data from worldpop.

    Parameters
    ----------
    country_code : str
        The 3-letter ISO code of the country (e.g., "COD", "BFA").
    output_path : Path
        The directory where the population data will be saved.
    year : str, optional
        The year for which to retrieve the population data. Defaults to "2020".
    overwrite : bool, optional
        Whether to overwrite existing files. Defaults to False.
    """
    current_run.log_info("Retrieving population data grid from worldpop.")
    wpop_client = WorldPopClient()
    current_run.log_info(f"Downloading data from : {wpop_client.base_url}")

    # Create output directory if it doesn't exist
    Path.mkdir(output_path, exist_ok=True)
    country = country_code.upper()
    pop_filename = f"{country}_worldpop_ppp_{year}.tif"
    pop_unadj_filename = f"{country}_worldpop_ppp_{year}_UNadj.tif"
    supports_unadj = int(year) <= 2020
    current_run.log_info(f"Retrieving data for country: {country} - year: {year}")

    try:
        if not overwrite and (output_path / pop_filename).exists():
            current_run.log_info(f"File {pop_filename} already exists. Skipping download")
        else:
            pop_file_path = wpop_client.download_data_for_country(
                country_iso3=country,
                year=year,
                un_adj=False,
                output_dir=output_path,
                fname=pop_filename,
            )
            current_run.log_info(f"Population data successfully downloaded under : {pop_file_path}.")

        if supports_unadj:
            if not overwrite and (output_path / pop_unadj_filename).exists():
                current_run.log_info(f"File {pop_unadj_filename} already exists. Skipping download")
            else:
                pop_file_un_adj_path = wpop_client.download_data_for_country(
                    country_iso3=country,
                    year=year,
                    un_adj=True,
                    output_dir=output_path,
                    fname=pop_unadj_filename,
                )
                current_run.log_info(
                    f"UN adjusted population data successfully downloaded under : {pop_file_un_adj_path}."
                )
        else:
            current_run.log_warning(
                f"UN adjusted WorldPop is not available for year {year}. Continuing with constrained raster only."
            )

    except Exception as e:
        raise Exception(f"Error retrieving WorldPop population data for {country} {year}: {e}") from e


def run_spacial_aggregations(snt_config: dict, input_dir: Path, output_dir: Path, year: str = "2020") -> None:
    """Run spatial aggregations on the worldpop population data (tif file)."""
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    pop_file_path = input_dir / f"{country_code}_worldpop_ppp_{year}.tif"
    pop_file_unadj_path = input_dir / f"{country_code}_worldpop_ppp_{year}_UNadj.tif"

    if pop_file_path.exists():
        run_spatial_aggregation(
            tif_file_path=pop_file_path,
            snt_config=snt_config,
            output_dir=output_dir,
        )
    else:
        current_run.log_warning(f"Population file not found: {pop_file_path}. Skipping aggregation.")

    if pop_file_unadj_path.exists():
        run_spatial_aggregation(
            tif_file_path=pop_file_unadj_path,
            snt_config=snt_config,
            output_dir=output_dir,
        )
    else:
        current_run.log_warning(
            f"UN adjusted population file not found: {pop_file_unadj_path}. Skipping aggregation."
        )


def run_spatial_aggregation(tif_file_path: Path, snt_config: dict, output_dir: Path) -> None:
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
        raise ValueError(
            "No valid geometries found in shapes file. Cannot compute zonal statistics."
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
    current_run.log_info(
        f"Aggregated population data saved under: {output_dir / f'{tif_file_path.stem}.csv'}"
    )


def snt_worldpop_format(snt_config: dict, year: int, input_dir: Path, output_dir: Path) -> None:
    """Format aggregated WorldPop population data for SNT."""
    country_code = snt_config["SNT_CONFIG"].get("COUNTRY_CODE")
    pop_data = pd.read_parquet(input_dir / f"{country_code}_worldpop_ppp_{year}.parquet")
    pop_unadj_path = input_dir / f"{country_code}_worldpop_ppp_{year}_UNadj.parquet"

    if pop_unadj_path.exists():
        pop_unadj_file = pd.read_parquet(pop_unadj_path)
        df = pop_data.merge(
            pop_unadj_file[["ADM2_ID", "POPULATION"]],
            on="ADM2_ID",
            how="left",
            suffixes=("", "_UNADJ"),
        )
    else:
        current_run.log_warning(
            f"UN adjusted aggregation file missing for {country_code} {year}; POPULATION_UNADJ will be left empty."
        )
        df = pop_data.copy()
        df["POPULATION_UNADJ"] = pd.NA
    df["YEAR"] = year  # Add year column (reference)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"{country_code}_worldpop_population.csv", index=False)
    df.to_parquet(output_dir / f"{country_code}_worldpop_population.parquet", index=False)
    current_run.log_info(
        f"Population data saved under: {output_dir / f'{country_code}_worldpop_population.csv'}"
    )


if __name__ == "__main__":
    snt_worldpop_extract()
