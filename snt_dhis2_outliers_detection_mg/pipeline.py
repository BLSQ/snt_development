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


@pipeline("snt_dhis2_outliers_detection_mg")
@parameter(
    "run_mg_partial",
    name="Run magic glasses partial method (up to MAD10)",
    help="Identifies outliers based on MAD15 and removes them, then identifies outliers based on MAD10",
    type=bool,
    default=True,
    required=False,
)
@parameter(
    "run_mg_complete",
    name="Run magic glasses complete method (up to seasonal3)",
    help="Picks up from magic glasses partial, and then applies seasonal 5 and seasonal 3",
    type=bool,
    default=False,
    required=False,
)
@parameter(
    "deviation_mad15",
    name="MAD deviation first pass",
    help="Number of MAD used for the first MG partial pass (default: 15).",
    type=int,
    default=15,
    required=False,
)
@parameter(
    "deviation_mad10",
    name="MAD deviation second pass",
    help="Number of MAD used for the second MG partial pass (default: 10).",
    type=int,
    default=10,
    required=False,
)
@parameter(
    "deviation_seasonal5",
    name="Seasonal deviation first pass",
    help="Deviation threshold for first seasonal pass in MG complete (default: 5).",
    type=int,
    default=5,
    required=False,
)
@parameter(
    "deviation_seasonal3",
    name="Seasonal deviation second pass",
    help="Deviation threshold for second seasonal pass in MG complete (default: 3).",
    type=int,
    default=3,
    required=False,
)
@parameter(
    "seasonal_workers",
    name="Seasonal workers",
    help="Number of workers for seasonal outlier detection when MG complete is enabled.",
    type=int,
    default=1,
    required=False,
)
@parameter(
    "dev_subset",
    name="Use dev subset",
    help="If enabled, run on a subset of ADM1 values to speed up debugging.",
    type=bool,
    default=False,
    required=False,
)
@parameter(
    "dev_subset_adm1_n",
    name="Dev subset ADM1 count",
    help="Number of ADM1 values to keep when dev subset is enabled.",
    type=int,
    default=2,
    required=False,
)
@parameter(
    "push_db",
    name="Push outliers table to DB",
    help="Push outliers table to DB for the Shiny app.",
    type=bool,
    default=False,
    required=False,
)
@parameter(
    "run_report_only",
    name="Run reporting only",
    help="This will only execute the reporting notebook",
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
def snt_dhis2_outliers_detection_mg(
    run_mg_partial: bool,
    run_mg_complete: bool,
    deviation_mad15: int,
    deviation_mad10: int,
    deviation_seasonal5: int,
    deviation_seasonal3: int,
    seasonal_workers: int,
    dev_subset: bool,
    dev_subset_adm1_n: int,
    push_db: bool,
    run_report_only: bool,
    pull_scripts: bool,
):
    """Dedicated Magic Glasses outliers detection pipeline for SNT DHIS2 data."""
    if not run_mg_partial and not run_mg_complete:
        raise ValueError("At least one MG mode must be enabled: run_mg_partial or run_mg_complete.")
    if seasonal_workers < 1:
        raise ValueError("seasonal_workers must be >= 1.")
    if dev_subset_adm1_n < 1:
        raise ValueError("dev_subset_adm1_n must be >= 1.")

    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_dhis2_outliers_detection_mg",
            report_scripts=["snt_dhis2_outliers_detection_mg_report.ipynb"],
            code_scripts=["snt_dhis2_outliers_detection_mg.ipynb"],
        )

    try:
        current_run.log_info("Starting SNT DHIS2 dedicated Magic Glasses outliers pipeline...")

        root_path = Path(workspace.files_path)
        pipeline_path = root_path / "pipelines" / "snt_dhis2_outliers_detection_mg"
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
                "DEVIATION_MAD15": deviation_mad15,
                "DEVIATION_MAD10": deviation_mad10,
                "DEVIATION_SEASONAL5": deviation_seasonal5,
                "DEVIATION_SEASONAL3": deviation_seasonal3,
                "SEASONAL_WORKERS": seasonal_workers,
                "DEV_SUBSET": dev_subset,
                "DEV_SUBSET_ADM1_N": dev_subset_adm1_n,
            }
            run_notebook(
                nb_path=pipeline_path / "code" / "snt_dhis2_outliers_detection_mg.ipynb",
                out_nb_path=pipeline_path / "papermill_outputs",
                kernel_name="ir",
                parameters=input_params,
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
                country_code=country_code,
            )

            parameters_file = save_pipeline_parameters(
                pipeline_name="snt_dhis2_outliers_detection_mg",
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
            nb_file=pipeline_path / "reporting" / "snt_dhis2_outliers_detection_mg_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            country_code=country_code,
        )

        current_run.log_info("Dedicated MG pipeline finished successfully.")

    except Exception as e:
        current_run.log_error(f"Notebook execution failed: {e}")
        raise


if __name__ == "__main__":
    snt_dhis2_outliers_detection_mg()
