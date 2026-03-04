from pathlib import Path

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


@pipeline("snt_dhis2_outliers_imputation_magic_glasses")
@parameter(
    "mode",
    name="Detection mode",
    help="Partial = fast (~7 min, MAD15 then MAD10). Complete = same + seasonal detection; very slow, can take several hours.",
    type=str,
    default="partial",
    required=False,
    choices=["partial", "complete"],
)
@parameter(
    "dev_subset",
    name="Limit to first 2 regions (debug)",
    help="Restrict data to 2 ADM1 for faster testing.",
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
    name="Update scripts",
    help="Pull the latest scripts from the repository before running.",
    type=bool,
    default=False,
    required=False,
)
def snt_dhis2_outliers_imputation_magic_glasses(
    mode: str,
    dev_subset: bool,
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
    if run_mg_complete:
        current_run.log_warning(
            "Complete mode selected: seasonal detection is very slow and can take several hours to run."
        )
    dev_subset_adm1_n = 2
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
            input_params = {
                "ROOT_PATH": Path(workspace.files_path).as_posix(),
                "RUN_MAGIC_GLASSES_PARTIAL": run_mg_partial,
                "RUN_MAGIC_GLASSES_COMPLETE": run_mg_complete,
                "DEVIATION_MAD15": 15,
                "DEVIATION_MAD10": 10,
                "DEVIATION_SEASONAL5": 5,
                "DEVIATION_SEASONAL3": 3,
                "SEASONAL_WORKERS": seasonal_workers,
                "DEV_SUBSET": dev_subset,
                "DEV_SUBSET_ADM1_N": dev_subset_adm1_n,
            }
            run_notebook(
                nb_path=pipeline_path / "code" / "snt_dhis2_outliers_imputation_magic_glasses.ipynb",
                out_nb_path=pipeline_path / "papermill_outputs",
                kernel_name="ir",
                parameters=input_params,
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
                country_code=country_code,
            )

            parameters_file = save_pipeline_parameters(
                pipeline_name="snt_dhis2_outliers_imputation_magic_glasses",
                parameters=input_params,
                output_path=data_path,
                country_code=country_code,
            )

            mg_files = [
                *data_path.glob(f"{country_code}_flagged_outliers_magic_glasses.parquet"),
                *data_path.glob(f"{country_code}_outlier_magic_glasses_*.parquet"),
                parameters_file,
            ]
            add_files_to_dataset(
                dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_OUTLIERS_IMPUTATION"],
                country_code=country_code,
                file_paths=mg_files,
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
