from pathlib import Path
import tempfile

from openhexa.sdk import current_run, parameter, pipeline, workspace
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    load_configuration_snt,
    pull_scripts_from_repository,
    run_notebook,
    run_report_notebook,
    save_pipeline_parameters,
    validate_config,
    create_outliers_db_table,
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


@pipeline("snt_dhis2_outliers_imputation_median")
@parameter(
    "deviation_median",
    name="Number of MAD around the median",
    help="Number of MAD around the median (default is 3)",
    type=int,
    default=3,
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
def snt_dhis2_outliers_imputation_median(
    deviation_median: int,
    push_db: bool,
    run_report_only: bool,
    pull_scripts: bool,
):
    """Outliers imputation pipeline Median method (median ± k*MAD) for SNT DHIS2 data."""
    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_dhis2_outliers_imputation_median",
            report_scripts=["snt_dhis2_outliers_imputation_median_report.ipynb"],
            code_scripts=["snt_dhis2_outliers_imputation_median.ipynb"],
        )

    try:
        current_run.log_info("Starting SNT DHIS2 outliers imputation Median method pipeline...")

        root_path = Path(workspace.files_path)
        pipeline_path = root_path / "pipelines" / "snt_dhis2_outliers_imputation_median"
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
            input_params = {
                "ROOT_PATH": Path(workspace.files_path).as_posix(),
                "DEVIATION_MEDIAN": deviation_median,
            }
            run_notebook(
                nb_path=pipeline_path / "code" / "snt_dhis2_outliers_imputation_median.ipynb",
                out_nb_path=pipeline_path / "papermill_outputs",
                kernel_name="ir",
                parameters=input_params,
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
                country_code=country_code,
            )

            parameters_file = save_pipeline_parameters(
                pipeline_name="snt_dhis2_outliers_imputation_median",
                parameters=input_params,
                output_path=data_path,
                country_code=country_code,
            )

            median_files = list(data_path.glob(f"{country_code}_routine_outliers-median*.parquet"))
            new_files = [*median_files, parameters_file]
            
            # Preserve existing files from other methods and add new ones
            dataset_id = snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_OUTLIERS_IMPUTATION"]
            preserve_and_add_files_to_dataset(
                dataset_id=dataset_id,
                country_code=country_code,
                new_files=new_files,
                method_prefix="median",
            )

            if push_db:
                create_outliers_db_table(country_code=country_code, data_path=data_path)

        else:
            current_run.log_info("Skipping outliers calculations, running only the reporting notebook.")

        run_report_notebook(
            nb_file=pipeline_path / "reporting" / "snt_dhis2_outliers_imputation_median_report.ipynb",
            nb_output_path=pipeline_path / "reporting" / "outputs",
            error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            country_code=country_code,
        )

        current_run.log_info("Pipeline finished successfully.")

    except Exception as e:
        current_run.log_error(f"Notebook execution failed: {e}")
        raise


if __name__ == "__main__":
    snt_dhis2_outliers_imputation_median()
