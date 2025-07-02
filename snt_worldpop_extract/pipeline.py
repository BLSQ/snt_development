from pathlib import Path
from openhexa.sdk import current_run, pipeline, workspace
from worlpopclient import WorldPopClient
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    load_configuration_snt,
    run_report_notebook,
    validate_config,
)


@pipeline("snt_worldpop_extract")
def snt_worldpop_extract():
    """Write your pipeline orchestration here."""
    # set paths
    snt_root_path = Path(workspace.files_path)
    pipeline_path = snt_root_path / "pipelines" / "snt_worldpop_extract"

    try:
        # get configuration
        snt_config_dict = load_configuration_snt(
            config_path=snt_root_path / "configuration" / "SNT_config.json"
        )

        # Validate configuration
        validate_config(snt_config_dict)

        # get country identifier for file naming
        country_code = snt_config_dict["SNT_CONFIG"].get("COUNTRY_CODE", None)
        if country_code is None:
            raise ValueError("COUNTRY_CODE is not specified in the configuration.")

        # Set output directory
        output_dir = snt_root_path / "data" / "worldpop" / "raw" / "population"
        pop_file_path = retrieve_population_data(country=country_code, output_path=output_dir)
        if pop_file_path is None:
            current_run.log_warning("No population data retrieved.")
            return

        # NOTE: Additional step should be here to produce a dataset file .csv & .parquet (!)

        # NOTE: Lets not push the data to the dataset as is not yet a proper dataset (!)
        # Add files to dataset
        # add_files_to_dataset(
        #     dataset_id=snt_config_dict["SNT_DATASET_IDENTIFIERS"].get("WORLDPOP_DATASET_EXTRACTS", None),
        #     country_code=country_code,
        #     file_paths=[pop_file_path],
        # )

        # Run report notebook
        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "SNT_wpop_population_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            nb_parameters=None,
        )
    except Exception as e:
        current_run.log_error(f"An error occurred in the pipeline: {e}")
        raise


def retrieve_population_data(country: str, output_path: Path) -> Path | None:
    """Retrieve raster population data from worldpop.

    Returns
    -------
    Path
        The path to the saved WorldPop population data file.
    """
    current_run.log_info("Retrieving population data grid from worldpop.")
    wpop_client = WorldPopClient()
    current_run.log_info(f"Connected to WorldPop endpoint : {wpop_client.base_url}")

    try:
        all_datasets = wpop_client.get_datasets_by_country(country_iso3=country)
        last_year = get_latest_population_year(all_datasets)
        current_run.log_info(f"Latest population data for {country} is from {last_year}.")
    except Exception as e:
        current_run.log_error(f"Error retrieving datasets for country {country}: {e}")
        raise

    # Create output directory if it doesn't exist
    Path.mkdir(output_path, exist_ok=True)

    try:
        pop_tif_path = wpop_client.get_population_geotiff(
            country_iso3=country,
            year=last_year,
            output_dir=output_path,
            fname=f"{country}_wpop_population_{last_year}.tif",
        )
        pop_compressed_path = wpop_client.compress_geotiff(
            src_path=pop_tif_path,
            dst_path=pop_tif_path.with_name(f"{country}_wpop_population_{last_year}_compressed.tif"),
        )
        current_run.log_info(f"WorldPop population data saved: {pop_compressed_path}")
        return pop_compressed_path
    except ValueError as e:
        current_run.log_warning(f"No data population retrieved {e}")
        return None
    except Exception as e:
        current_run.log_error(f"Error retrieving WorldPop population data: {e}")
        raise


def get_latest_population_year(datasets: list) -> int:
    """Returns the latest population year from a list of dataset dictionaries.

    Parameters
    ----------
    datasets : list
        List of dictionaries, each containing a 'popyear' key.

    Returns
    -------
    int
        The maximum population year found in the datasets.
    """
    return max([ds["popyear"] for ds in datasets])


if __name__ == "__main__":
    snt_worldpop_extract()
