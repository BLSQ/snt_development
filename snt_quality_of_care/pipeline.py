from pathlib import Path

from openhexa.sdk import current_run, parameter, pipeline, workspace
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    load_configuration_snt,
    pull_scripts_from_repository,
    run_notebook,
    run_report_notebook,
    save_pipeline_parameters,
    validate_config,
)


@pipeline("snt_quality_of_care")
@parameter(
    "outlier_imputation_method",
    name="Outlier imputation method",
    help="Which imputed routine data to use (filename: {country}_routine_outliers-{method}_imputed.parquet)",
    choices=["mean", "median", "iqr", "trend"],
    type=str,
    default="mean",
    required=True,
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
    name="Pull scripts",
    help="Pull the latest scripts from the repository",
    type=bool,
    default=False,
    required=False,
)
def snt_quality_of_care(
    outlier_imputation_method: str,
    run_report_only: bool,
    pull_scripts: bool,
):
    """Quality of care indicators from DHIS2 routine data (testing rate, treatment rate, case fatality, etc.)."""
    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_quality_of_care",
            report_scripts=["snt_quality_of_care_report.ipynb"],
            code_scripts=["snt_quality_of_care.ipynb"],
        )

    try:
        current_run.log_info("Starting SNT Quality of Care pipeline...")
        root_path = Path(workspace.files_path)
        pipeline_path = root_path / "pipelines" / "snt_quality_of_care"
        data_path = root_path / "data" / "dhis2" / "quality_of_care"
        (pipeline_path / "reporting" / "outputs").mkdir(parents=True, exist_ok=True)
        data_path.mkdir(parents=True, exist_ok=True)

        config_path = root_path / "configuration" / "SNT_config.json"
        snt_config = load_configuration_snt(config_path=config_path)
        validate_config(snt_config)
        country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]

        if not run_report_only:
            notebook_params = {
                "ROOT_PATH": root_path.as_posix(),
                "OUTLIER_IMPUTATION_METHOD": outlier_imputation_method,
            }
            run_notebook(
                nb_path=pipeline_path / "code" / "snt_quality_of_care.ipynb",
                out_nb_path=pipeline_path / "papermill_outputs",
                kernel_name="python3",
                parameters=notebook_params,
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
                country_code=country_code,
            )

            parameters_file = save_pipeline_parameters(
                pipeline_name="snt_quality_of_care",
                parameters=notebook_params,
                output_path=data_path,
                country_code=country_code,
            )

            dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"].get("DHIS2_QUALITY_OF_CARE")
            if dataset_id:
                add_files_to_dataset(
                    dataset_id=dataset_id,
                    country_code=country_code,
                    file_paths=[
                        data_path / f"{country_code}_quality_of_care.parquet",
                        data_path / f"{country_code}_quality_of_care.csv",
                        data_path / f"{country_code}_quality_of_care.xlsx",
                        parameters_file,
                    ],
                )
        else:
            current_run.log_info("Skipping calculations, running only the reporting notebook.")

        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_quality_of_care_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            country_code=country_code,
        )

        current_run.log_info("Pipeline finished successfully.")
    except Exception as e:
        current_run.log_error(f"Quality of care pipeline failed: {e}")
        raise


if __name__ == "__main__":
    snt_quality_of_care()
