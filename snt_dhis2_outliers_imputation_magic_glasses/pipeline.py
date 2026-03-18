from pathlib import Path
import time

from openhexa.sdk import current_run, parameter, pipeline, workspace
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    load_configuration_snt,
    push_data_to_db_table,
    pull_scripts_from_repository,
    run_notebook,
    run_report_notebook,
    save_pipeline_parameters,
    validate_config,
)


@pipeline("snt_dhis2_outliers_imputation_magic_glasses")
@parameter(
    "complete",
    name="Run complete mode",
    help="False (default): Partial mode (MAD15 then MAD10). True: Complete mode (Partial + seasonal detection, can take several hours).",
    type=bool,
    default=False,
    required=False,
)
@parameter(
    "push_db",
    name="Push to Shiny database",
    help="Send the outliers table to the database for the Shiny app.",
    type=bool,
    default=False,
    required=False,
)
@parameter(
    "run_report_only",
    name="Report only",
    help="Run only the reporting notebook (no recomputation).",
    type=bool,
    default=False,
    required=False,
)
@parameter(
    "pull_scripts",
    name="Pull scripts",
    help="Pull the latest scripts from the repository before running.",
    type=bool,
    default=False,
    required=False,
)
def snt_dhis2_outliers_imputation_magic_glasses(
    complete: bool,
    push_db: bool,
    run_report_only: bool,
    pull_scripts: bool,
):
    """Dedicated Magic Glasses outliers pipeline for SNT DHIS2 data."""
    run_mg_complete = bool(complete)
    current_run.log_info(
        f"Selected detection mode: {'complete' if run_mg_complete else 'partial'}"
    )
    if run_mg_complete:
        current_run.log_warning(
            "Complete mode selected: seasonal detection is very slow and can take several hours to run."
        )
    seasonal_workers = 1  # default: sequential execution of seasonal detection

    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_dhis2_outliers_imputation_magic_glasses",
            report_scripts=["snt_dhis2_outliers_imputation_magic_glasses_report.ipynb"],
            code_scripts=["snt_dhis2_outliers_imputation_magic_glasses.ipynb"],
        )

    try:
        current_run.log_info("Starting SNT DHIS2 Magic Glasses outliers pipeline...")

        root_path = Path(workspace.files_path)
        pipeline_path = root_path / "pipelines" / "snt_dhis2_outliers_imputation_magic_glasses"
        data_path = root_path / "data" / "dhis2" / "outliers_imputation"
        code_notebook = (
            pipeline_path
            / "code"
            / "snt_dhis2_outliers_imputation_magic_glasses.ipynb"
        )
        report_notebook = (
            pipeline_path
            / "reporting"
            / "snt_dhis2_outliers_imputation_magic_glasses_report.ipynb"
        )

        pipeline_path.mkdir(parents=True, exist_ok=True)
        data_path.mkdir(parents=True, exist_ok=True)
        current_run.log_info(f"Pipeline path: {pipeline_path}")
        current_run.log_info(f"Data path: {data_path}")

        config_path = root_path / "configuration" / "SNT_config.json"
        snt_config = load_configuration_snt(config_path=config_path)
        validate_config(snt_config)
        country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]

        if not run_report_only:
            # Avoid publishing stale artifacts from previous runs.
            for old_file in data_path.glob(f"{country_code}_routine_outliers*.parquet"):
                old_file.unlink(missing_ok=True)

            input_params = {
                "ROOT_PATH": Path(workspace.files_path).as_posix(),
                "OUTLIERS_METHOD": "MG_COMPLETE" if run_mg_complete else "MG_PARTIAL",
                "RUN_MAGIC_GLASSES_COMPLETE": run_mg_complete,
                "DEVIATION_MAD15": 15,
                "DEVIATION_MAD10": 10,
                "DEVIATION_SEASONAL5": 5,
                "DEVIATION_SEASONAL3": 3,
                "SEASONAL_WORKERS": seasonal_workers,
            }
            run_start_ts = time.time()
            run_notebook(
                nb_path=code_notebook,
                out_nb_path=pipeline_path / "papermill_outputs",
                kernel_name="ir",
                parameters=input_params,
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
                country_code=country_code,
            )

            expected_outputs = [
                data_path / f"{country_code}_routine_outliers_detected.parquet",
                data_path / f"{country_code}_routine_outliers_imputed.parquet",
                data_path / f"{country_code}_routine_outliers_removed.parquet",
            ]
            missing_outputs = [
                path.name
                for path in expected_outputs
                if (not path.exists() or path.stat().st_mtime < run_start_ts)
            ]
            if missing_outputs:
                raise RuntimeError(
                    "Expected output files were not generated during this run: "
                    + ", ".join(missing_outputs)
                )

            parameters_file = save_pipeline_parameters(
                pipeline_name="snt_dhis2_outliers_imputation_magic_glasses",
                parameters=input_params,
                output_path=data_path,
                country_code=country_code,
            )

            add_files_to_dataset(
                dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_OUTLIERS_IMPUTATION"],
                country_code=country_code,
                file_paths=[
                    *data_path.glob(f"{country_code}_routine_outliers*.parquet"),
                    parameters_file,
                ],
            )

            if push_db:
                push_data_to_db_table(
                    table_name=(
                        "outliers_detection_magic_glasses_complete"
                        if run_mg_complete
                        else "outliers_detection_magic_glasses_partial"
                    ),
                    file_path=data_path / f"{country_code}_routine_outliers_detected.parquet",
                )

        else:
            current_run.log_info("Skipping calculations, running only the reporting notebook.")

        run_report_notebook(
            nb_file=report_notebook,
            nb_output_path=pipeline_path / "reporting" / "outputs",
            error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            country_code=country_code,
        )

        current_run.log_info("Magic Glasses pipeline finished successfully.")

    except Exception as e:
        current_run.log_error(f"Notebook execution failed: {e}")
        raise


if __name__ == "__main__":
    snt_dhis2_outliers_imputation_magic_glasses()
