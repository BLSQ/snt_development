import time

from pathlib import Path
from openhexa.sdk import current_run, pipeline, workspace, parameter

from snt_lib.snt_pipeline_utils import (
    pull_scripts_from_repository,
    add_files_to_dataset,
    load_configuration_snt,
    run_notebook,
    run_report_notebook,
    validate_config,
    dataset_file_exists,
    save_pipeline_parameters,
    check_outputs_generated,
)


@pipeline("snt_dhis2_reporting_rate_dataset")
@parameter(
    "run_report_only",
    name="Run reporting notebook only",
    help="This will execute only the reporting notebook. Important: "
    "this uses the outputs of the latest run of the full pipeline! Therefore, be aware that:"
    " if you have not run the full pipeline yet, or if the inputs have changed since the last run, "
    "the report may be outdated or incorrect.",
    type=bool,
    default=False,
    required=False,
)
@parameter(
    "pull_scripts",
    name="Pull notebooks from repository",
    help="Pull the latest notebooks from the GitHub repository. "
    "Note: this will overwrite any local changes to the notebooks!",
    type=bool,
    default=False,
    required=False,
)
def snt_dhis2_reporting_rate_dataset(run_report_only: bool, pull_scripts: bool):
    """Orchestration function. Calls other functions within the pipeline."""
    if pull_scripts:
        current_run.log_info("Pulling pipeline notebooks from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_dhis2_reporting_rate_dataset",
            report_scripts=["snt_dhis2_reporting_rate_dataset_report.ipynb"],
            code_scripts=["snt_dhis2_reporting_rate_dataset.ipynb"],
        )

    # Set paths
    root_path = Path(workspace.files_path)
    pipeline_path = root_path / "pipelines" / "snt_dhis2_reporting_rate_dataset"
    data_path = root_path / "data" / "dhis2" / "reporting_rate"
    data_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load configuration
        snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
        validate_config(snt_config)
    except Exception as e:
        current_run.log_error(f"Loading configuration failed: {e}")
        raise

    country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]
    formatted_ds_id = snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_DATASET_FORMATTED"]
    rr_file = f"{country_code}_reporting.parquet"

    if not run_report_only:
        if not dataset_file_exists(ds_id=formatted_ds_id, filename=rr_file):
            current_run.log_warning(
                f"Reporting rates file {rr_file} was not found in dataset {formatted_ds_id}. "
                "Perhaps the reporting rates were not extracted from DHIS2 "
                f"(see: configuration/SNT_config_{country_code}.json)."
            )
            return

        try:
            validate_reporting_rate_product_uid(snt_config)
        except ValueError as e:
            current_run.log_error(str(e))
            raise

        nb_parameters = {
            "ROOT_PATH": root_path.as_posix(),
        }

        try:
            params_file = save_pipeline_parameters(
                pipeline_name="snt_dhis2_reporting_rate_dataset",
                parameters=nb_parameters,
                output_path=data_path,
                country_code=country_code,
            )
            current_run.log_info(f"Saved pipeline parameters to {params_file}")
        except Exception as e:
            current_run.log_error(f"Saving pipeline parameters failed: {e}")
            raise

        run_start_ts = time.time()
        try:
            run_notebook(
                nb_path=pipeline_path / "code" / "snt_dhis2_reporting_rate_dataset.ipynb",
                out_nb_path=pipeline_path / "papermill_outputs",
                parameters=nb_parameters,
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
                country_code=country_code,
            )
        except Exception as e:
            current_run.log_error(f"Running notebook failed: {e}")
            raise

        expected_outputs = [
            data_path / f"{country_code}_reporting_rate_dataset.parquet",
            data_path / f"{country_code}_reporting_rate_dataset.csv",
        ]
        check_outputs_generated(file_paths=expected_outputs, run_start_ts=run_start_ts)
        add_files_to_dataset(
            dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_REPORTING_RATE"],
            country_code=country_code,
            file_paths=[
                *expected_outputs,
                params_file,
            ],
        )

    else:
        current_run.log_info("Skipping calculations, running only the reporting.")

    try:
        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_dhis2_reporting_rate_dataset_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            country_code=country_code,
        )
    except Exception as e:
        current_run.log_error(f"Running report notebook failed: {e}")
        raise

    current_run.log_info("Pipeline completed successfully!")


def validate_reporting_rate_product_uid(snt_config: dict) -> list[str]:
    """Check that SNT_CONFIG.REPORTING_RATE_PRODUCT_UID lists at least one UID.

    The notebook keeps only the reporting rows whose PRODUCT_UID is in this list, so an empty or
    missing list would leave no data to compute the reporting rate from.

    Parameters
    ----------
    snt_config : dict
        The loaded SNT configuration.

    Returns
    -------
    list[str]
        The configured product UIDs, without blank entries.

    Raises
    ------
    ValueError
        If REPORTING_RATE_PRODUCT_UID is missing, empty, or contains only blank values.
    """
    product_uids = snt_config["SNT_CONFIG"].get("REPORTING_RATE_PRODUCT_UID") or []
    if isinstance(product_uids, str):
        product_uids = [product_uids]
    product_uids = [uid.strip() for uid in product_uids if isinstance(uid, str) and uid.strip()]
    if not product_uids:
        raise ValueError(
            "SNT_CONFIG.REPORTING_RATE_PRODUCT_UID is not set in configuration/SNT_config.json. "
            "List the DHIS2 product UIDs to compute the reporting rate from: the dataset UIDs when "
            "reporting rates are extracted with REPORTING_DATASETS, or both indicator UIDs (actual and "
            "expected reports) when extracted with REPORTING_INDICATORS."
        )
    return product_uids


if __name__ == "__main__":
    snt_dhis2_reporting_rate_dataset()
