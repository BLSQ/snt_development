from pathlib import Path

from openhexa.sdk import current_run, parameter, pipeline, workspace
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    dataset_file_exists,
    load_configuration_snt,
    run_notebook,
    run_report_notebook,
    validate_config,
)


@pipeline("snt_dhis2_formatting")
@parameter(
    "run_report_only",
    name="Run reporting only",
    help="This will only execute the reporting notebook",
    type=bool,
    default=False,
    required=False,
)
def snt_dhis2_formatting(run_report_only: bool):
    """Write your pipeline orchestration here.

    Pipeline functions should only call tasks and should never perform IO operations or
    expensive computations.
    """
    # set paths
    snt_root_path = Path(workspace.files_path)
    snt_pipeline_path = snt_root_path / "pipelines" / "snt_dhis2_formatting"
    snt_dhis2_formatted_path = snt_root_path / "data" / "dhis2" / "formatted"
    snt_dhis2_formatted_path.mkdir(parents=True, exist_ok=True)

    try:
        if not run_report_only:
            # Load configuration
            snt_config_dict = load_configuration_snt(
                config_path=snt_root_path / "configuration" / "SNT_config.json"
            )

            # Validate configuration
            validate_config(snt_config_dict)

            # get country identifier for naming
            country_code = snt_config_dict["SNT_CONFIG"].get("COUNTRY_CODE", None)
            if country_code is None:
                current_run.log_warning("COUNTRY_CODE is not specified in the configuration.")

            # format data for SNT
            dhis2_analytics_formatting(
                snt_root_path=snt_root_path, pipeline_root_path=snt_pipeline_path, snt_config=snt_config_dict
            )
            dhis2_population_formatting(
                snt_root_path=snt_root_path, pipeline_root_path=snt_pipeline_path, snt_config=snt_config_dict
            )
            dhis2_shapes_formatting(
                snt_root_path=snt_root_path, pipeline_root_path=snt_pipeline_path, snt_config=snt_config_dict
            )
            dhis2_pyramid_formatting(
                snt_root_path=snt_root_path, pipeline_root_path=snt_pipeline_path, snt_config=snt_config_dict
            )
            dhis2_reporting_rates_formatting(
                snt_root_path=snt_root_path, pipeline_root_path=snt_pipeline_path, snt_config=snt_config_dict
            )

            # add files to a new dataset version
            files_ready = add_files_to_dataset(
                dataset_id=snt_config_dict["SNT_DATASET_IDENTIFIERS"].get("DHIS2_DATASET_FORMATTED", None),
                country_code=country_code,
                file_paths=[
                    snt_dhis2_formatted_path / f"{country_code}_routine.parquet",
                    snt_dhis2_formatted_path / f"{country_code}_routine.csv",
                    snt_dhis2_formatted_path / f"{country_code}_population.parquet",
                    snt_dhis2_formatted_path / f"{country_code}_population.csv",
                    snt_dhis2_formatted_path / f"{country_code}_shapes.geojson",
                    snt_dhis2_formatted_path / f"{country_code}_pyramid.parquet",
                    snt_dhis2_formatted_path / f"{country_code}_pyramid.csv",
                    snt_dhis2_formatted_path / f"{country_code}_reporting.parquet",
                    snt_dhis2_formatted_path / f"{country_code}_reporting.csv",
                ],
            )
        else:
            files_ready = True

        run_report_notebook(
            nb_file=snt_pipeline_path / "reporting" / "SNT_dhis2_indicators_report.ipynb",
            nb_output_path=snt_pipeline_path / "reporting" / "outputs",
            nb_parameters=None,
            ready=files_ready,
        )

    except Exception as e:
        current_run.log_error(f"Error in SNT DHIS2 formatting: {e}")
        raise


def dhis2_analytics_formatting(
    snt_root_path: Path,
    pipeline_root_path: Path,
    snt_config: dict,
) -> None:
    """Format DHIS2 analytics data for SNT."""
    current_run.log_info("Formatting DHIS2 analytics data.")

    # set parameters for notebook
    nb_parameter = {
        "SNT_ROOT_PATH": str(snt_root_path),
    }

    # Check if the reporting rates data file exists
    country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]
    ds_id = snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_DATASET_EXTRACTS"]
    if not dataset_file_exists(ds_id=ds_id, filename=f"{country_code}_dhis2_raw_analytics.parquet"):
        current_run.log_info("File analytics data not found, skipping formatting.")
        return

    try:
        run_notebook(
            nb_path=pipeline_root_path / "code" / "SNT_dhis2_routine_format.ipynb",
            out_nb_path=pipeline_root_path / "papermill_outputs",
            parameters=nb_parameter,
        )
    except Exception as e:
        raise Exception(f"Error in formatting analytics data: {e}") from e


def dhis2_population_formatting(
    snt_root_path: Path,
    pipeline_root_path: Path,
    snt_config: dict,
) -> None:
    """Format DHIS2 population data for SNT."""
    current_run.log_info("Formatting DHIS2 population data.")

    # set parameters for notebook
    nb_parameter = {
        "SNT_ROOT_PATH": str(snt_root_path),
    }

    # Check if the reporting rates data file exists
    country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]
    ds_id = snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_DATASET_EXTRACTS"]
    if not dataset_file_exists(ds_id=ds_id, filename=f"{country_code}_dhis2_raw_population.parquet"):
        current_run.log_info("File population data not found, skipping formatting.")
        return

    try:
        run_notebook(
            nb_path=pipeline_root_path / "code" / "SNT_dhis2_population_format.ipynb",
            out_nb_path=pipeline_root_path / "papermill_outputs",
            parameters=nb_parameter,
        )
    except Exception as e:
        raise Exception(f"Error in formatting population data: {e}") from e


def dhis2_shapes_formatting(
    snt_root_path: Path,
    pipeline_root_path: Path,
    snt_config: dict,
) -> None:
    """Format DHIS2 shapes data for SNT."""
    current_run.log_info("Formatting DHIS2 shapes data.")

    # set parameters for notebook
    nb_parameter = {
        "SNT_ROOT_PATH": str(snt_root_path),
    }

    # Check if the reporting rates data file exists
    country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]
    ds_id = snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_DATASET_EXTRACTS"]
    if not dataset_file_exists(ds_id=ds_id, filename=f"{country_code}_dhis2_raw_shapes.parquet"):
        current_run.log_info("File shapes data not found, skipping formatting.")
        return

    try:
        run_notebook(
            nb_path=pipeline_root_path / "code" / "SNT_dhis2_shapes_format.ipynb",
            out_nb_path=pipeline_root_path / "papermill_outputs",
            parameters=nb_parameter,
        )
    except Exception as e:
        raise Exception(f"Error in formatting shapes data: {e}") from e

    # current_run.log_info(
    #     f"SNT population formatted data saved under: {snt_root_path / 'data' / 'dhis2_formatted'}"
    # )


def dhis2_pyramid_formatting(
    snt_root_path: Path,
    pipeline_root_path: Path,
    snt_config: dict,
) -> None:
    """Format DHIS2 pyramid data for SNT."""
    current_run.log_info("Formatting DHIS2 pyramid data.")

    # set parameters for notebook
    nb_parameter = {
        "SNT_ROOT_PATH": str(snt_root_path),
    }

    # Check if the reporting rates data file exists
    country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]
    ds_id = snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_DATASET_EXTRACTS"]
    if not dataset_file_exists(ds_id=ds_id, filename=f"{country_code}_dhis2_raw_pyramid.parquet"):
        current_run.log_info("File pyramid data not found, skipping formatting.")
        return

    try:
        run_notebook(
            nb_path=pipeline_root_path / "code" / "SNT_dhis2_pyramid_format.ipynb",
            out_nb_path=pipeline_root_path / "papermill_outputs",
            parameters=nb_parameter,
        )
    except Exception as e:
        raise Exception(f"Error in formatting pyramid data: {e}") from e


def dhis2_reporting_rates_formatting(
    snt_root_path: Path,
    pipeline_root_path: Path,
    snt_config: dict,
) -> None:
    """Format DHIS2 reporting data for SNT."""
    current_run.log_info("Formatting DHIS2 reporting rates data.")

    # set parameters for notebook
    nb_parameter = {
        "SNT_ROOT_PATH": str(snt_root_path),
    }

    # Check if the reporting rates data file exists
    country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]
    ds_id = snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_DATASET_EXTRACTS"]
    if not dataset_file_exists(ds_id=ds_id, filename=f"{country_code}_dhis2_raw_reporting.parquet"):
        current_run.log_info("File reporting rates data not found, skipping formatting.")
        return

    try:
        run_notebook(
            nb_path=pipeline_root_path / "code" / "SNT_dhis2_reporting_rates_format.ipynb",
            out_nb_path=pipeline_root_path / "papermill_outputs",
            parameters=nb_parameter,
        )
    except Exception as e:
        raise Exception(f"Error in formatting reporting rates data: {e}") from e


if __name__ == "__main__":
    snt_dhis2_formatting()
