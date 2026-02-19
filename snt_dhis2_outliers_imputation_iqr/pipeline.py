from pathlib import Path

from openhexa.sdk import current_run, parameter, pipeline, workspace
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    load_configuration_snt,
    pull_scripts_from_repository,
    run_notebook,
    run_report_notebook,
    validate_config,
    create_outliers_db_table,
)


@pipeline("snt_dhis2_outliers_imputation_iqr")
@parameter(
    "deviation_iqr",
    name="IQR multiplier",
    help="IQR multiplier (default is 1.5)",
    type=float,
    default=1.5,
    required=False,
)
@parameter(
    "push_db",
    name="Push outliers table to DB",
    help="Push outliers table to DB",
    type=bool,
    default=True,
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
def snt_dhis2_outliers_imputation_iqr(
    deviation_iqr: float,
    push_db: bool,
    run_report_only: bool,
    pull_scripts: bool,
):
    """Outliers imputation pipeline IQR method for SNT DHIS2 data."""
    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_dhis2_outliers_imputation_iqr",
            report_scripts=["snt_dhis2_outliers_imputation_iqr_report.ipynb"],
            code_scripts=["snt_dhis2_outliers_imputation_iqr.ipynb"],
        )

    try:
        current_run.log_info("Starting SNT DHIS2 outliers imputation IQR method pipeline...")

        root_path = Path(workspace.files_path)
        pipeline_path = root_path / "pipelines" / "snt_dhis2_outliers_imputation_iqr"
        data_path = root_path / "data" / "dhis2" / "outliers_imputation"

        pipeline_path.mkdir(parents=True, exist_ok=True)
        data_path.mkdir(parents=True, exist_ok=True)
        current_run.log_info(f"Pipeline path: {pipeline_path}")
        current_run.log_info(f"Data path: {data_path}")

        config_path = root_path / "configuration" / "SNT_config.json"
        snt_config = load_configuration_snt(config_path=config_path)
        validate_config(snt_config)
        country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]

        if not run_report_only:
            run_notebook(
                nb_path=pipeline_path / "code" / "snt_dhis2_outliers_imputation_iqr.ipynb",
                out_nb_path=pipeline_path / "papermill_outputs",
                kernel_name="ir",
                parameters={
                    "ROOT_PATH": Path(workspace.files_path).as_posix(),
                    "DEVIATION_IQR": deviation_iqr,
                },
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            )

            iqr_files = list(data_path.glob(f"{country_code}_routine_outliers-iqr*.parquet"))
            if iqr_files:
                add_files_to_dataset(
                    dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_OUTLIERS_IMPUTATION"],
                    country_code=country_code,
                    file_paths=iqr_files,
                )

            if push_db:
                create_outliers_db_table(country_code=country_code, data_path=data_path)

        else:
            current_run.log_info("Skipping outliers calculations, running only the reporting notebook.")

        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_dhis2_outliers_imputation_iqr_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
        )

        current_run.log_info("Pipeline finished successfully.")

    except Exception as e:
        current_run.log_error(f"Notebook execution failed: {e}")
        raise


if __name__ == "__main__":
    snt_dhis2_outliers_imputation_iqr()
