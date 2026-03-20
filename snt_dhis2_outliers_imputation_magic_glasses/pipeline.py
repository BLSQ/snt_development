from pathlib import Path
import time
import tempfile

from openhexa.sdk import current_run, parameter, pipeline, workspace
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    create_outliers_db_table,
    load_configuration_snt,
    pull_scripts_from_repository,
    run_notebook,
    run_report_notebook,
    save_pipeline_parameters,
    validate_config,
)


def preserve_and_add_files_to_dataset(
    dataset_id: str,
    country_code: str,
    new_files: list[Path],
    method_prefix: str,
):
    """
    Add new files to dataset while preserving existing files from other methods.
    
    Args:
        dataset_id: Dataset identifier
        country_code: Country code
        new_files: List of new file paths to add
        method_prefix: Prefix pattern to identify files from this method (e.g., "mean", "median", "magic_glasses")
    """
    try:
        dataset = workspace.get_dataset(dataset_id)
        latest_version = dataset.latest_version
        existing_files = latest_version.list_files()
        
        # Filter out files from this method but keep others
        preserved_files = []
        for file_obj in existing_files:
            filename = file_obj.name
            
            # Determine if this file belongs to the current method
            is_current_method = False
            if method_prefix == "magic_glasses":
                # Magic Glasses files: flagged_outliers_magic_glasses.parquet, outlier_magic_glasses_*.parquet
                is_current_method = (
                    filename == f"{country_code}_flagged_outliers_magic_glasses.parquet" or
                    filename.startswith(f"{country_code}_outlier_magic_glasses_")
                )
            else:
                # Other methods: routine_outliers-{method}*.parquet
                is_current_method = filename.startswith(f"{country_code}_routine_outliers-{method_prefix}")
            
            # Preserve files from other methods
            if not is_current_method:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
                        tmp_path = Path(tmp_file.name)
                        file_obj.download(tmp_path)
                        preserved_files.append(tmp_path)
                        current_run.log_info(f"Preserving existing file: {filename}")
                except Exception as e:
                    current_run.log_warning(f"Could not preserve file {filename}: {e}")
        
        # Combine preserved files with new files
        all_files = preserved_files + new_files
        current_run.log_info(f"Adding {len(new_files)} new files and preserving {len(preserved_files)} existing files")
    except Exception as e:
        current_run.log_warning(f"Could not preserve existing files, adding only new files: {e}")
        all_files = new_files
    
    add_files_to_dataset(
        dataset_id=dataset_id,
        country_code=country_code,
        file_paths=all_files,
    )


@pipeline("snt_dhis2_outliers_imputation_magic_glasses")
@parameter(
    "mode",
    name="Detection mode",
    help="Partial: fast (~7 min, MAD15 then MAD10). Complete: Partial + seasonal detection, can take several hours.",
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
    run_mg_partial = True
    run_mg_complete = mode_clean == "complete"
    current_run.log_info(f"Selected detection mode: {mode_clean}")
    current_run.log_info(
        f"Flags => RUN_MAGIC_GLASSES_PARTIAL={run_mg_partial}, RUN_MAGIC_GLASSES_COMPLETE={run_mg_complete}"
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
        data_path = root_path / "data" / "dhis2" / "outliers_detection"

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
            for old_file in data_path.glob(f"{country_code}_flagged_outliers_magic_glasses.parquet"):
                old_file.unlink(missing_ok=True)
            for old_file in data_path.glob(f"{country_code}_outlier_magic_glasses_*.parquet"):
                old_file.unlink(missing_ok=True)

            input_params = {
                "ROOT_PATH": Path(workspace.files_path).as_posix(),
                "RUN_MAGIC_GLASSES_PARTIAL": run_mg_partial,
                "RUN_MAGIC_GLASSES_COMPLETE": run_mg_complete,
                "DEVIATION_MAD15": 15,
                "DEVIATION_MAD10": 10,
                "DEVIATION_SEASONAL5": 5,
                "DEVIATION_SEASONAL3": 3,
                "SEASONAL_WORKERS": seasonal_workers,
            }
            run_start_ts = time.time()
            run_notebook(
                nb_path=pipeline_path / "code" / "snt_dhis2_outliers_imputation_magic_glasses.ipynb",
                out_nb_path=pipeline_path / "papermill_outputs",
                kernel_name="ir",
                parameters=input_params,
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
                country_code=country_code,
            )

            partial_file = data_path / f"{country_code}_outlier_magic_glasses_partial.parquet"
            complete_file = data_path / f"{country_code}_outlier_magic_glasses_complete.parquet"
            if not partial_file.exists() or partial_file.stat().st_mtime < run_start_ts:
                raise RuntimeError(
                    "Partial output file was not generated during this run."
                )
            if run_mg_complete and (
                not complete_file.exists() or complete_file.stat().st_mtime < run_start_ts
            ):
                raise RuntimeError(
                    "Complete mode selected but complete seasonal output was not generated during this run."
                )

            parameters_file = save_pipeline_parameters(
                pipeline_name="snt_dhis2_outliers_imputation_magic_glasses",
                parameters=input_params,
                output_path=data_path,
                country_code=country_code,
            )

            # Get new files for Magic Glasses
            mg_files = list(data_path.glob(f"{country_code}_flagged_outliers_magic_glasses.parquet"))
            mg_files.extend(data_path.glob(f"{country_code}_outlier_magic_glasses_*.parquet"))
            new_files = [*mg_files, parameters_file]
            
            # Preserve existing files from other methods and add new ones
            dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_OUTLIERS_IMPUTATION"]
            preserve_and_add_files_to_dataset(
                dataset_id=dataset_id,
                country_code=country_code,
                new_files=new_files,
                method_prefix="magic_glasses",
            )

            if push_db:
                try:
                    create_outliers_db_table(country_code=country_code, data_path=data_path)
                except Exception as e:
                    current_run.log_warning(
                        f"MG files were produced but DB push failed with current utility: {e}"
                    )

        else:
            current_run.log_info("Skipping calculations, running only the reporting notebook.")

        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_dhis2_outliers_imputation_magic_glasses_report.ipynb",
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
