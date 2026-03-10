from pathlib import Path

from openhexa.sdk import current_run, pipeline, workspace, parameter
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    load_configuration_snt,
    validate_config,
    run_report_notebook,
    run_notebook,
    pull_scripts_from_repository,
    save_pipeline_parameters,
)


@pipeline("snt_dhis2_quality_of_care")
@parameter(
    "outlier_imputation_method",
    name="Outlier imputation method",
    help="Choose which outlier detection/imputation method to use.",
    type=str,
    choices=["mean", "median", "iqr", "trend"],
    default="mean",
    required=True,
)
@parameter(
    "data_action",
    name="Data action",
    help="Choose whether to use imputed data (outliers replaced) or removed data (outliers removed).",
    type=str,
    choices=["imputed", "removed"],
    default="imputed",
    required=True,
)
@parameter(
    "run_report_only",
    name="Run reporting only",
    help="Skip computations and execute only the reporting notebook.",
    type=bool,
    default=False,
    required=False,
)
@parameter(
    "pull_scripts",
    name="Pull scripts",
    help="Pull the latest pipeline scripts from the repository.",
    type=bool,
    default=False,
    required=False,
)
def snt_dhis2_quality_of_care(
    outlier_imputation_method: str,
    data_action: str,
    run_report_only: bool,
    pull_scripts: bool,
):
    """Compute quality-of-care indicators from outliers-imputed DHIS2 routine data."""
    try:
        current_run.log_info("Starting SNT Quality of Care pipeline...")
        root_path = Path(workspace.files_path)
        pipeline_path = root_path / "pipelines" / "snt_dhis2_quality_of_care"
        data_path = root_path / "data" / "dhis2" / "quality_of_care"
        pipeline_path.mkdir(parents=True, exist_ok=True)
        data_path.mkdir(parents=True, exist_ok=True)

        if pull_scripts:
            current_run.log_info("Pulling pipeline scripts from repository.")
            pull_scripts_from_repository(
                pipeline_name="snt_dhis2_quality_of_care",
                report_scripts=["snt_dhis2_quality_of_care_report.ipynb"],
                code_scripts=["snt_dhis2_quality_of_care.ipynb"],
            )

        snt_config = load_configuration_snt(config_path=root_path / "configuration" / "SNT_config.json")
        validate_config(snt_config)
        country_code = snt_config["SNT_CONFIG"]["COUNTRY_CODE"]

        nb_parameters = {
            "outlier_imputation_method": outlier_imputation_method,
            "data_action": data_action,
        }

        parameters_file = save_pipeline_parameters(
            pipeline_name="snt_dhis2_quality_of_care",
            parameters=nb_parameters,
            output_path=data_path,
            country_code=country_code,
        )

        if not run_report_only:
            run_notebook(
                nb_path=pipeline_path / "code" / "snt_dhis2_quality_of_care.ipynb",
                out_nb_path=pipeline_path / "papermill_outputs",
                kernel_name="ir",
                parameters=nb_parameters,
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
                country_code=country_code,
            )

            add_files_to_dataset(
                dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_QUALITY_OF_CARE"],
                country_code=country_code,
                file_paths=[
                    data_path / f"{country_code}_quality_of_care_{outlier_imputation_method}_{data_action}.parquet",
                    data_path / f"{country_code}_quality_of_care_{outlier_imputation_method}_{data_action}.csv",
                    data_path / f"{country_code}_quality_of_care_{outlier_imputation_method}_{data_action}.xlsx",
                    parameters_file,
                ],
            )
        else:
            current_run.log_info("Skipping computations, running only reporting notebook.")

        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_dhis2_quality_of_care_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            country_code=country_code,
        )

        current_run.log_info("Quality of Care pipeline finished successfully.")
    except Exception as e:
        current_run.log_error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    snt_dhis2_quality_of_care()
