import json
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from subprocess import CalledProcessError

import geopandas as gpd
import numpy as np
import pandas as pd
import papermill as pm
import requests
from nbclient.exceptions import CellTimeoutError
from openhexa.sdk import (
    DatasetVersion,
    current_run,
    parameter,
    pipeline,
    workspace,
)
from rasterstats import zonal_stats

# os.environ["PROJ_LIB"] = "/opt/conda/share/proj"
# os.environ["GDAL_DATA"] = "/opt/conda/share/gdal"


@pipeline("snt_map_extract")
@parameter(
    "run_report_only",
    name="Run reporting only",
    help="This will only execute the reporting notebook",
    type=bool,
    default=False,
    required=False,
)
def snt_map_extract(run_report_only: bool) -> None:
    """Main function to get raster data for a dhis2 country."""
    root_path = Path(workspace.files_path)
    pipeline_path = root_path / "pipeline" / "snt_map_extract"

    try:
        # Load configuration
        snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
        validate_config(snt_config)

        country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]
        dataset_shapes_id = snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_DATASET_FORMATTED"]
        dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"]["SNT_MAP_EXTRACT"]

        # get org unit level
        # TODO: move this validation to validate_config()
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

        mapping_coverage_indicators = {
            "Malaria": {
                "Malaria__202206_Global_Pf_Mortality_Count": "Pf_mortality-count",
                "Malaria__202406_Global_Pf_Mortality_Count": "Pf_mortality-count",
                "Malaria__202206_Global_Pf_Mortality_Rate": "Pf_mortality-rate",
                "Malaria__202406_Global_Pf_Mortality_Rate": "Pf_mortality-rate",
                "Malaria__202206_Global_Pf_Incidence_Rate": "Pf_incidence-rate",
                "Malaria__202406_Global_Pf_Incidence_Rate": "Pf_incidence-rate",
                "Malaria__202206_Global_Pf_Incidence_Count": "Pf_incidence-count",
                "Malaria__202406_Global_Pf_Incidence_Count": "Pf_incidence-count",
                "Malaria__202206_Global_Pv_Incidence_Rate": "Pv_incidence-rate",
                "Malaria__202406_Global_Pv_Incidence_Rate": "Pv_incidence-rate",
                "Malaria__202206_Global_Pv_Incidence_Count": "Pv_incidence-count",
                "Malaria__202406_Global_Pv_Incidence_Count": "Pv_incidence-count",
                "Malaria__202206_Global_Pf_Parasite_Rate": "Pf_PR-rate",
                "Malaria__202406_Global_Pf_Parasite_Rate": "Pf_PR-rate",
                "Malaria__202206_Global_Pv_Parasite_Rate": "Pv_PR-rate",
                "Malaria__202406_Global_Pv_Parasite_Rate": "Pv_PR-rate",
            },
            "Interventions": {
                "Interventions__202106_Global_Antimalarial_Effective_Treatment": "Antimalarial_EFT-rate",
                "Interventions__202406_Global_Antimalarial_Effective_Treatment": "Antimalarial_EFT-rate",
                "Interventions__202106_Africa_Insecticide_Treated_Net_Use_Rate": "ITN_use-rate",
                "Interventions__202406_Africa_Insecticide_Treated_Net_Use_Rate": "ITN_use-rate",
                "Interventions__202106_Africa_Insecticide_Treated_Net_Access": "ITN_access-rate",
                "Interventions__202406_Africa_Insecticide_Treated_Net_Access": "ITN_access-rate",
                "Interventions__202106_Africa_IRS_Coverage": "IRS_coverage-rate",
                "Interventions__202106_Africa_Insecticide_Treated_Net_Use": "ITN_use_rate-rate",
                "Interventions__202406_Africa_Insecticide_Treated_Net_Use": "ITN_use_rate-rate",
            },
        }

        if run_report_only:
            output_path = root_path / "data" / "map"
            output_path.mkdir(parents=True, exist_ok=True)

            make_table(
                mapping_coverage_indicators,
                shapes=shapes,
                org_level=org_level,
                output_path=output_path,
            )

            add_files_to_dataset(
                dataset_id=dataset_id,
                country_code=country_code,
                file_paths=[
                    output_path / f"{country_code}_map_data.parquet",
                    output_path / f"{country_code}_map_data.csv",
                ],
            )

        else:
            current_run.log_info("Skipping calculations, running only the reporting.")

        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "SNT_seasonality_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
        )

        current_run.log_info("Pipeline completed successfully!")

    except Exception as e:
        current_run.log_error(f"Pipeline error: {e}")
        raise e


def get_file_from_dataset(dataset_id: str, filename: str) -> gpd.GeoDataFrame:
    """Get a file from a dataset.

    Parameters
    ----------
    dataset_id : str
        The ID of the dataset.
    filename : str
        The name of the file to retrieve.

    Returns
    -------
    gpd.GeoDataFrame
        The GeoDataFrame containing the shapes.
    """
    dataset = workspace.get_dataset(dataset_id)
    if not dataset:
        raise ValueError(f"Dataset with ID {dataset_id} not found.")

    version = dataset.latest_version
    if not version:
        raise ValueError(f"No versions found for dataset {dataset_id}.")

    file_path = version.get_file(filename)
    if not file_path:
        raise ValueError(f"File {filename} not found in dataset {dataset_id}.")

    return gpd.read_file(file_path)


def add_files_to_dataset(
    dataset_id: str,
    country_code: str,
    file_paths: list[str],
) -> bool:
    """Add files to a new dataset version.

    Parameters
    ----------
    dataset_id : str
        The ID of the dataset to which files will be added.
    country_code : str
        The country code used for naming the dataset version.
    file_paths : list[str]
        A list of file paths to be added to the dataset.

    Raises
    ------
    ValueError
        If the dataset ID is not specified in the configuration.

    Returns
    -------
    bool
        True if at least one file was added successfully, False otherwise.
    """
    added_any = False

    for file in file_paths:
        src = Path(file)
        if not src.exists():
            current_run.log_warning(f"File not found: {src}")
            continue

        try:
            # Determine file extension
            ext = src.suffix.lower()
            if ext == ".parquet":
                df = pd.read_parquet(src)
                tmp_suffix = ".parquet"
            elif ext == ".csv":
                df = pd.read_csv(src)
                tmp_suffix = ".csv"
            elif ext == ".geojson":
                gdf = gpd.read_file(src)
                tmp_suffix = ".geojson"
            else:
                current_run.log_warning(f"Unsupported file format: {src.name}")
                continue

            with tempfile.NamedTemporaryFile(suffix=tmp_suffix) as tmp:
                if ext == ".parquet":
                    df.to_parquet(tmp.name)
                elif ext == ".csv":
                    df.to_csv(tmp.name, index=False)
                elif ext == ".geojson":
                    gdf.to_file(tmp.name, driver="GeoJSON")

                if not added_any:
                    new_version = get_new_dataset_version(ds_id=dataset_id, prefix=f"{country_code}_snt")
                    current_run.log_info(f"New dataset version created : {new_version.name}")
                    added_any = True
                new_version.add_file(tmp.name, filename=src.name)
                current_run.log_info(f"File {src.name} added to dataset version : {new_version.name}")
        except Exception as e:
            current_run.log_warning(f"File {src.name} cannot be added : {e}")
            continue

    if not added_any:
        current_run.log_info("No valid files found. Dataset version was not created.")
        return False

    return True


def get_new_dataset_version(ds_id: str, prefix: str = "ds") -> DatasetVersion:
    """Create and return a new dataset version.

    Parameters
    ----------
    ds_id : str
        The ID of the dataset for which a new version will be created.
    prefix : str, optional
        Prefix for the dataset version name (default is "ds").

    Returns
    -------
    DatasetVersion
        The newly created dataset version.

    Raises
    ------
    Exception
        If an error occurs while creating the new dataset version.
    """
    existing_datasets = workspace.list_datasets()
    if ds_id in [eds.slug for eds in existing_datasets]:
        dataset = workspace.get_dataset(ds_id)
    else:
        current_run.log_warning(f"Dataset with ID {ds_id} not found, creating a new one.")
        dataset = workspace.create_dataset(
            name=ds_id.replace("-", "_").upper(), description="SNT Process dataset"
        )

    version_name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    try:
        new_version = dataset.create_version(version_name)
    except Exception as e:
        raise Exception(f"An error occurred while creating the new dataset version: {e}") from e

    return new_version


def run_report_notebook(
    nb_file: Path,
    nb_output_path: Path,
    ready: bool = True,
) -> None:
    """Execute a Jupyter notebook using Papermill.

    Parameters
    ----------
    nb_file : Path
        The full file path to the notebook.
    nb_output_path : Path
        The path to the directory where the output notebook will be saved.
    ready : bool, optional
        Whether the notebook should be executed (default is True).
    """
    if not ready:
        current_run.log_info("Reporting execution skipped.")
        return

    current_run.log_info(f"Executing report notebook: {nb_file}")
    execution_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    nb_output_full_path = nb_output_path / f"{nb_file.stem}_OUTPUT_{execution_timestamp}.ipynb"
    nb_output_path.mkdir(parents=True, exist_ok=True)

    try:
        pm.execute_notebook(input_path=nb_file, output_path=nb_output_full_path)
    except CellTimeoutError as e:
        raise CellTimeoutError(f"Notebook execution timed out: {e}") from e
    except Exception as e:
        raise Exception(f"Error executing the notebook {type(e)}: {e}") from e
    generate_html_report(nb_output_full_path)


def generate_html_report(output_notebook_path: Path, out_format: str = "html") -> None:
    """Generate an HTML report from a Jupyter notebook.

    Parameters
    ----------
    output_notebook_path : Path
        Path to the output notebook file.
    out_format : str
        output extension

    Raises
    ------
    RuntimeError
        If an error occurs during the conversion process.
    """
    if not output_notebook_path.is_file() or output_notebook_path.suffix.lower() != ".ipynb":
        raise RuntimeError(f"Invalid notebook path: {output_notebook_path}")

    report_path = output_notebook_path.with_suffix(".html")
    current_run.log_info(f"Generating HTML report {report_path}")
    cmd = [
        "jupyter",
        "nbconvert",
        f"--to={out_format}",
        str(output_notebook_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except CalledProcessError as e:
        raise CalledProcessError(f"Error converting notebook to HTML (exit {e.returncode}): {e}") from e

    current_run.add_file_output(report_path.as_posix())


def load_configuration_snt(config_path: str) -> dict:
    """Load the SNT configuration from a JSON file.

    Parameters
    ----------
    config_path : str
        Path to the configuration JSON file.

    Returns
    -------
    dict
        The loaded configuration as a dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    ValueError
        If the configuration file contains invalid JSON.
    Exception
        For any other unexpected errors.
    """
    try:
        # Load the JSON file
        with Path.open(config_path, "r") as file:
            config_json = json.load(file)
        current_run.log_info(f"SNT configuration loaded: {config_path}")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: The file {config_path} was not found.") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Error: The file contains invalid JSON {e}") from e
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}") from e

    return config_json


def validate_config(config: dict) -> None:
    """Validate that the critical configuration values are set properly."""
    try:
        snt_config = config["SNT_CONFIG"]
        dataset_ids = config["SNT_DATASET_IDENTIFIERS"]
        definitions = config["DHIS2_DATA_DEFINITIONS"]
    except KeyError as e:
        raise KeyError(f"Missing top-level key in config: {e}") from e

    # Required keys in SNT_CONFIG
    required_snt_keys = [
        "COUNTRY_CODE",
        "DHIS2_ADMINISTRATION_1",
        "DHIS2_ADMINISTRATION_2",
        "ANALYTICS_ORG_UNITS_LEVEL",
        "POPULATION_ORG_UNITS_LEVEL",
    ]
    for key in required_snt_keys:
        if key not in snt_config or snt_config[key] in [None, ""]:
            raise ValueError(f"Missing or empty configuration for: SNT_CONFIG.{key}")

    # Required dataset identifiers
    required_dataset_keys = [
        "DHIS2_DATASET_EXTRACTS",
        "DHIS2_DATASET_FORMATTED",
        "DHIS2_REPORTING_RATE",
        "DHIS2_INCIDENCE",
        "WORLDPOP_DATASET_EXTRACTS",
        "ERA5_DATASET_CLIMATE",
        "SNT_SEASONALITY",
        "SNT_MAP_EXTRACT",
    ]
    for key in required_dataset_keys:
        if key not in dataset_ids or dataset_ids[key] in [None, ""]:
            raise ValueError(f"Missing or empty configuration for: SNT_DATASET_IDENTIFIERS.{key}")

    # Check population indicator
    pop_indicators = definitions.get("POPULATION_INDICATOR_DEFINITIONS", {})
    tot_population = pop_indicators.get("TOT_POPULATION", [])
    if not tot_population:
        raise ValueError("Missing or empty TOT_POPULATION indicator definition.")

    # Check at least one indicator under DHIS2_INDICATOR_DEFINITIONS
    indicator_defs = definitions.get("DHIS2_INDICATOR_DEFINITIONS", {})
    flat_indicators = [val for sublist in indicator_defs.values() for val in sublist]
    if not flat_indicators:
        raise ValueError("No indicators defined under DHIS2_INDICATOR_DEFINITIONS.")


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
        The base URL of the GeoServer WCS service (default is "https://data.malariaatlas.org/geoserver/").
    """
    # Example malaria raster layer (adjust this to match exact layer name)
    for category, layers in mapping_coverage_indicators.items():
        url = f"{geoserver_url}{category}/wcs"
        for layer_name in layers:
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


def make_table(
    mapping_coverage_indicators: dict,
    country_code: str,
    shapes: gpd.GeoDataFrame,
    level: int,
    output_path: Path,
) -> pd.DataFrame:
    """Generate a table of zonal statistics for given coverage indicators and save the results.

    Parameters
    ----------
    mapping_coverage_indicators : dict
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
    invalid_shapes = shapes[shapes.geometry is None]
    if len(invalid_shapes) > 0:
        current_run.log_warning(
            f"DHIS2 units with no geometry: {list(invalid_shapes[f'level_{level}_name'].unique())}"
        )
    shapes = shapes[shapes.geometry is not None]
    if len(shapes) > 0:
        minx, miny, maxx, maxy = shapes.total_bounds
        rasters_path = output_path / "raster_files"
        rasters_path.mkdir(parents=True, exist_ok=True)
        download_raster_data(rasters_path, mapping_coverage_indicators, minx, maxx, miny, maxy)

        # Step 1: Load Admin Polygons
        final_df = pd.DataFrame()
        for category, layers in mapping_coverage_indicators.items():
            current_run.log_info(f"Processing {category}...")
            for layer_name in layers:
                # raster_filename = output_path / "raster_files" / f"{layer_name}.tif"
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
                result_gdf["metric_name"] = mapping_coverage_indicators[category][layer_name]
                metric_columns = ["mean"]
                ref_columns = set(result_gdf.columns).difference(set(metric_columns))

                # Melt to long format
                melt_df = result_gdf.melt(
                    id_vars=ref_columns,
                    value_vars=metric_columns,
                    var_name="statistic",
                    value_name="value",
                )
                melt_df["version"] = version
                melt_df["value"] = melt_df["value"].astype(float)
                melt_df["value"] = pd.to_numeric(melt_df["value"], errors="coerce")
                melt_df = melt_df[np.isfinite(melt_df["value"])]
                final_df = pd.concat([final_df, melt_df], ignore_index=True)

        # Step 5: Save Output
        formatted_path = output_path / "formatted"
        formatted_path.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(formatted_path / f"{country_code}_map_data.parquet", index=False)
        final_df.to_csv(formatted_path / f"{country_code}_map_data.csv", index=False)
        current_run.log_info(f"Output file saved under : {formatted_path / f'{country_code}_map_data.csv'}")

    return final_df


if __name__ == "__main__":
    snt_map_extract()
