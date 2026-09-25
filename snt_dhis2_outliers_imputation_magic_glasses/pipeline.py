import time
from pathlib import Path

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


@pipeline("snt_dhis2_outliers_imputation_magic_glasses", timeout=28800)
@parameter(
    "mode",
    name="Detection mode",
    help=(
        "Partial: fast (~7 min, MAD15 then MAD10). Complete: Partial + "
        "seasonal detection, can take several hours."
    ),
    type=str,
    default="partial",
    required=False,
    choices=["partial", "complete"],
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
    mode: str,
    push_db: bool,
    run_report_only: bool,
    pull_scripts: bool,
):
    """Dedicated Magic Glasses outliers detection pipeline for SNT DHIS2 data."""
    mode_clean = (mode or "partial").strip().lower()
    if mode_clean not in ("partial", "complete"):
        raise ValueError('mode must be "partial" or "complete".')
    run_mg_complete = mode_clean == "complete"
    current_run.log_info(f"Selected detection mode: {mode_clean}")

    if run_mg_complete:
        current_run.log_warning(
            "Complete mode selected: seasonal detection is very slow and can take several hours to run."
        )

    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_dhis2_outliers_imputation_magic_glasses",
            report_scripts=["snt_dhis2_outliers_imputation_magic_glasses_report.ipynb"],
            code_scripts=["snt_dhis2_outliers_imputation_magic_glasses.ipynb"],
        )

    current_run.log_info("Starting SNT DHIS2 outliers imputation Magic Glasses method pipeline...")

    root_path = Path(workspace.files_path)
    pipeline_path = root_path / "pipelines" / "snt_dhis2_outliers_imputation_magic_glasses"
    data_path = root_path / "data" / "dhis2" / "outliers_imputation"

    pipeline_path.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)
    current_run.log_info(f"Pipeline path: {pipeline_path}")
    current_run.log_info(f"Data path: {data_path}")

    try:
        snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
        validate_config(snt_config)
        country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]
    except Exception as e:
        current_run.log_error(f"Failed to load and validate configuration: {e}")
        raise

    if not run_report_only:
        input_params = {
            "ROOT_PATH": root_path.as_posix(),
            "RUN_MAGIC_GLASSES_COMPLETE": run_mg_complete,
            "DEVIATION_MAD15": 15,
            "DEVIATION_MAD10": 10,
            "DEVIATION_SEASONAL5": 5,
            "DEVIATION_SEASONAL3": 3,
        }
        expected_outputs = [
            data_path / f"{country_code}_routine_outliers_detected.parquet",
            data_path / f"{country_code}_routine_outliers_imputed.parquet",
            data_path / f"{country_code}_routine_outliers_removed.parquet",
        ]

        run_start_ts = time.time()
        try:
            run_notebook(
                nb_path=pipeline_path / "code" / "snt_dhis2_outliers_imputation_magic_glasses.ipynb",
                out_nb_path=pipeline_path / "papermill_outputs",
                kernel_name="ir",
                parameters=input_params,
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
                country_code=country_code,
            )
        except Exception as e:
            current_run.log_error(f"Failed to run outliers imputation notebook: {e}")
            raise

        check_outputs_generated(file_paths=expected_outputs, run_start_ts=run_start_ts)

        try:
            parameters_file = save_pipeline_parameters(
                pipeline_name="snt_dhis2_outliers_imputation_magic_glasses",
                parameters=input_params,
                output_path=data_path,
                country_code=country_code,
            )
        except Exception as e:
            current_run.log_error(f"Failed to save pipeline parameters: {e}")
            raise

        try:
            add_files_to_dataset(
                dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_OUTLIERS_IMPUTATION"],
                country_code=country_code,
                file_paths=[*expected_outputs, parameters_file],
            )
        except Exception as e:
            current_run.log_error(f"Failed to add files to dataset: {e}")
            raise

        if push_db:
            try:
                push_data_to_db_table(
                    table_name="outliers_detected",
                    file_path=data_path / f"{country_code}_routine_outliers_detected.parquet",
                )
            except Exception as e:
                current_run.log_error(f"Failed to push data to DB table: {e}")
                raise

    else:
        current_run.log_info("Skipping outliers calculations, running only the reporting notebook.")

    try:
        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_dhis2_outliers_imputation_magic_glasses_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            country_code=country_code,
        )
    except Exception as e:
        current_run.log_error(f"Failed to run reporting notebook: {e}")
        raise

    current_run.log_info("Pipeline finished successfully.")


def check_outputs_generated(file_paths: list[Path], run_start_ts: float) -> None:
    """Raise if any expected output was not written during the current run.

    Guards against publishing stale files: all outliers imputation pipelines write the same
    filenames, so a leftover file may come from a previous run of another method.

    Parameters
    ----------
    file_paths : list[Path]
        Output files the notebook is expected to produce.
    run_start_ts : float
        Timestamp taken just before the notebook ran; files modified earlier are stale.

    Raises
    ------
    RuntimeError
        If a file is missing or was last modified before ``run_start_ts``.
    """
    missing = [p.name for p in file_paths if not p.exists() or p.stat().st_mtime < run_start_ts]
    if missing:
        msg = f"Expected output files were not generated during this run: {', '.join(missing)}"
        current_run.log_error(msg)
        raise RuntimeError(msg)


if __name__ == "__main__":
    snt_dhis2_outliers_imputation_magic_glasses()
