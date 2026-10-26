import time
from pathlib import Path
from openhexa.sdk import current_run, pipeline, parameter, workspace
from snt_lib.snt_pipeline_utils import (
    pull_scripts_from_repository,
    add_files_to_dataset,
    load_configuration_snt,
    run_notebook,
    run_report_notebook,
    validate_config,
    save_pipeline_parameters,
    check_outputs_generated,
)


@pipeline("snt_dhs_indicators")
@parameter(
    "run_reports_only",
    name="Run reportings only",
    help="Only execute the report notebooks on existing outputs; nothing is recomputed or published",
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
def dhs_indicators(run_reports_only: bool, pull_scripts: bool) -> None:
    """Compute, report and publish the DHS indicators.

    Each indicator (bednets, careseeking, mortality, prevalence, vaccination) runs its computation
    notebook, checks that its output files were written during this run, then runs its report notebook.
    The outputs and the parameters file are published to the `DHS_INDICATORS` dataset as one new
    version, only when the computation notebooks ran.

    Args:
        run_reports_only (bool): If True, only the report notebooks run; no outputs are computed or published.
        pull_scripts (bool): If True, pull the latest notebooks and utils scripts from the repository first.
    """
    data_source = "DHS"
    admin_level = "ADM2"

    # general paths
    snt_root_path = Path(workspace.files_path)
    data_output_path = snt_root_path / "data" / "dhs" / "indicators"
    pipeline_path = snt_root_path / "pipelines" / "snt_dhs_indicators"

    if pull_scripts:
        current_run.log_info("Pulling pipeline scripts from repository.")
        pull_scripts_from_repository(
            pipeline_name="snt_dhs_indicators",
            report_scripts=[
                "snt_dhs_bednets_report.ipynb",
                "snt_dhs_careseeking_report.ipynb",
                "snt_dhs_mortality_report.ipynb",
                "snt_dhs_prevalence_report.ipynb",
                "snt_dhs_vaccination_report.ipynb",
            ],
            code_scripts=[
                "snt_dhs_bednets_computation.ipynb",
                "snt_dhs_careseeking_computation.ipynb",
                "snt_dhs_mortality_computation.ipynb",
                "snt_dhs_prevalence_computation.ipynb",
                "snt_dhs_vaccination_computation.ipynb",
            ],
            extra_util_scripts=["snt_dhs_indicator_tables.r"],
        )

    try:
        # Load configuration
        snt_config_dict = load_configuration_snt(snt_root_path / "configuration" / "SNT_config.json")
        validate_config(snt_config_dict)
    except Exception as e:
        current_run.log_error(f"Error loading or validating configuration: {e}")
        raise

    # get country identifier for naming
    country_code = snt_config_dict["SNT_CONFIG"].get("COUNTRY_CODE")

    # bednet files
    file_paths_bednets = [
        data_output_path / "bednets" / f"{country_code}_{data_source}_{admin_level}_PCT_ITN_ACCESS.parquet",
        data_output_path / "bednets" / f"{country_code}_{data_source}_{admin_level}_PCT_ITN_ACCESS.csv",
        data_output_path / "bednets" / f"{country_code}_{data_source}_{admin_level}_PCT_ITN_USE.parquet",
        data_output_path / "bednets" / f"{country_code}_{data_source}_{admin_level}_PCT_ITN_USE.csv",
    ]

    run_dhs_indicator_notebooks(
        pipeline_root_path=pipeline_path,
        computation_notebook_name="snt_dhs_bednets_computation.ipynb",
        reporting_notebook_name="snt_dhs_bednets_report.ipynb",
        run_report_only=run_reports_only,
        country_code=country_code,
        expected_output_files=file_paths_bednets,
    )

    # careseeking files
    file_paths_careseeking = [
        data_output_path
        / "careseeking"
        / f"{country_code}_{data_source}_{admin_level}_PCT_CARESEEKING_SAMPLE_AVERAGE.parquet",
        data_output_path
        / "careseeking"
        / f"{country_code}_{data_source}_{admin_level}_PCT_CARESEEKING_SAMPLE_AVERAGE.csv",
        data_output_path / "careseeking" / f"{country_code}_{data_source}_{admin_level}_PCT_NO_CARE.parquet",
        data_output_path / "careseeking" / f"{country_code}_{data_source}_{admin_level}_PCT_NO_CARE.csv",
        data_output_path
        / "careseeking"
        / f"{country_code}_{data_source}_{admin_level}_PCT_PUBLIC_CARE.parquet",
        data_output_path / "careseeking" / f"{country_code}_{data_source}_{admin_level}_PCT_PUBLIC_CARE.csv",
        data_output_path
        / "careseeking"
        / f"{country_code}_{data_source}_{admin_level}_PCT_PRIVATE_CARE.parquet",
        data_output_path / "careseeking" / f"{country_code}_{data_source}_{admin_level}_PCT_PRIVATE_CARE.csv",
    ]

    run_dhs_indicator_notebooks(
        pipeline_root_path=pipeline_path,
        computation_notebook_name="snt_dhs_careseeking_computation.ipynb",
        reporting_notebook_name="snt_dhs_careseeking_report.ipynb",
        run_report_only=run_reports_only,
        country_code=country_code,
        expected_output_files=file_paths_careseeking,
    )

    # mortality files
    file_paths_mortality = [
        data_output_path / "mortality" / f"{country_code}_{data_source}_{admin_level}_U5MR_PERMIL.parquet",
        data_output_path / "mortality" / f"{country_code}_{data_source}_{admin_level}_U5MR_PERMIL.csv",
    ]
    run_dhs_indicator_notebooks(
        pipeline_root_path=pipeline_path,
        computation_notebook_name="snt_dhs_mortality_computation.ipynb",
        reporting_notebook_name="snt_dhs_mortality_report.ipynb",
        run_report_only=run_reports_only,
        country_code=country_code,
        expected_output_files=file_paths_mortality,
    )

    # prevalence files
    file_paths_prevalence = [
        data_output_path
        / "prevalence"
        / f"{country_code}_{data_source}_{admin_level}_PCT_U5_PREV_RDT.parquet",
        data_output_path / "prevalence" / f"{country_code}_{data_source}_{admin_level}_PCT_U5_PREV_RDT.csv",
    ]
    run_dhs_indicator_notebooks(
        pipeline_root_path=pipeline_path,
        computation_notebook_name="snt_dhs_prevalence_computation.ipynb",
        reporting_notebook_name="snt_dhs_prevalence_report.ipynb",
        run_report_only=run_reports_only,
        country_code=country_code,
        expected_output_files=file_paths_prevalence,
    )

    # DTP* vaccination files
    file_paths_vaccination = [
        data_output_path / "vaccination" / f"{country_code}_{data_source}_{admin_level}_PCT_DTP1.parquet",
        data_output_path / "vaccination" / f"{country_code}_{data_source}_{admin_level}_PCT_DTP1.csv",
        data_output_path / "vaccination" / f"{country_code}_{data_source}_{admin_level}_PCT_DTP2.parquet",
        data_output_path / "vaccination" / f"{country_code}_{data_source}_{admin_level}_PCT_DTP2.csv",
        data_output_path / "vaccination" / f"{country_code}_{data_source}_{admin_level}_PCT_DTP3.parquet",
        data_output_path / "vaccination" / f"{country_code}_{data_source}_{admin_level}_PCT_DTP3.csv",
        data_output_path
        / "vaccination"
        / f"{country_code}_{data_source}_{admin_level}_PCT_DROPOUT_DTP.parquet",
        data_output_path / "vaccination" / f"{country_code}_{data_source}_{admin_level}_PCT_DROPOUT_DTP.csv",
    ]
    run_dhs_indicator_notebooks(
        pipeline_root_path=pipeline_path,
        computation_notebook_name="snt_dhs_vaccination_computation.ipynb",
        reporting_notebook_name="snt_dhs_vaccination_report.ipynb",
        run_report_only=run_reports_only,
        country_code=country_code,
        expected_output_files=file_paths_vaccination,
    )

    if run_reports_only:
        current_run.log_info("Reports only: no outputs computed, dataset not updated.")
        return

    parameters_file = save_pipeline_parameters(
        pipeline_name="snt_dhs_indicators",
        parameters={"run_reports_only": run_reports_only, "pull_scripts": pull_scripts},
        output_path=data_output_path,
        country_code=country_code,
    )

    # add files to a new dataset version
    add_files_to_dataset(
        dataset_id=snt_config_dict["SNT_DATASET_IDENTIFIERS"].get("DHS_INDICATORS", None),
        country_code=country_code,
        file_paths=[
            *file_paths_bednets,
            *file_paths_careseeking,
            *file_paths_mortality,
            *file_paths_prevalence,
            *file_paths_vaccination,
            parameters_file,
        ],
    )


def run_dhs_indicator_notebooks(
    pipeline_root_path: Path,
    computation_notebook_name: str,
    reporting_notebook_name: str,
    expected_output_files: list[Path],
    run_report_only: bool = False,
    country_code: str | None = None,
) -> None:
    """Execute the computation notebook, check its outputs, then run the reporting notebook.

    Args:
        pipeline_root_path (Path): Directory of the pipeline.
        computation_notebook_name (str): Filename of the computation notebook.
        reporting_notebook_name (str): Filename of the reporting notebook.
        expected_output_files (list[Path]): Output files the computation notebook must write during this run;
            checked after it runs (skipped when `run_report_only` is True).
        run_report_only (bool): If True, only the reporting notebook will be executed.
        country_code (str | None): Country code; a country-specific notebook variant
            (`{notebook}_{country_code}.ipynb`) runs instead of the generic one when it exists.
    """
    computation_notebook_path = pipeline_root_path / "code" / computation_notebook_name
    papermill_folder_path = pipeline_root_path / "papermill_outputs"
    reporting_folder_path = pipeline_root_path / "reporting"

    if not run_report_only:
        run_start_ts = time.time()
        try:
            run_notebook(
                nb_path=computation_notebook_path,
                out_nb_path=papermill_folder_path,
                parameters={},
                error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
                country_code=country_code,
            )
        except Exception as e:
            current_run.log_error(f"Error running computation notebook '{computation_notebook_name}': {e}")
            raise
        check_outputs_generated(file_paths=expected_output_files, run_start_ts=run_start_ts)

    try:
        run_report_notebook(
            nb_file=reporting_folder_path / reporting_notebook_name,
            nb_output_path=reporting_folder_path / "outputs",
            error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"},
            country_code=country_code,
        )
    except Exception as e:
        current_run.log_error(f"Error running reporting notebook '{reporting_notebook_name}': {e}")
        raise


if __name__ == "__main__":
    dhs_indicators()
