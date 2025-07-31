from pathlib import Path
from openhexa.sdk import current_run, pipeline, workspace
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    load_configuration_snt,
    run_notebook,
    run_report_notebook,
    validate_config,
)


@pipeline("snt_dhs_indicators")
def dhs_indicators():
    """Pipeline to compute and report on DHS indicators."""
    data_source = "DHS"
    admin_level = "ADM1"

    # general paths
    snt_root_path = Path(workspace.files_path)
    data_output_path = snt_root_path / "data" / "dhs" / "indicators"
    pipeline_path = snt_root_path / "pipelines" / "snt_dhs_indicators"

    try:
        # Load configuration
        snt_config_dict = load_configuration_snt(
            config_path=Path(snt_root_path, "configuration", "SNT_config.json")
        )

        # Validate configuration
        validate_config(snt_config_dict)

        # get country identifier for naming
        country_code = snt_config_dict["SNT_CONFIG"].get("COUNTRY_CODE")

        run_dhs_indicator_notebooks(
            pipeline_root_path=pipeline_path,
            computation_notebook_name="snt_dhs_bednets_computation.ipynb",
            reporting_notebook_name="snt_dhs_bednets_report.ipynb",
        )

        run_dhs_indicator_notebooks(
            pipeline_root_path=pipeline_path,
            computation_notebook_name="snt_dhs_careseeking_computation.ipynb",
            reporting_notebook_name="snt_dhs_careseeking_report.ipynb",
        )

        run_dhs_indicator_notebooks(
            pipeline_root_path=pipeline_path,
            computation_notebook_name="snt_dhs_mortality_computation.ipynb",
            reporting_notebook_name="snt_dhs_mortality_report.ipynb",
        )

        run_dhs_indicator_notebooks(
            pipeline_root_path=pipeline_path,
            computation_notebook_name="snt_dhs_prevalence_computation.ipynb",
            reporting_notebook_name="snt_dhs_prevalence_report.ipynb",
        )

        run_dhs_indicator_notebooks(
            pipeline_root_path=pipeline_path,
            computation_notebook_name="snt_dhs_vaccination_computation.ipynb",
            reporting_notebook_name="snt_dhs_vaccination_report.ipynb",
        )

        # add files to a new dataset version
        add_files_to_dataset(
            dataset_id=snt_config_dict["SNT_DATASET_IDENTIFIERS"].get("DHS_INDICATORS", None),
            country_code=country_code,
            file_paths=[
                # bednet access files
                data_output_path
                / "bednets"
                / f"{country_code}_{data_source}_{admin_level}_PCT_ITN_ACCESS.parquet",
                data_output_path
                / "bednets"
                / f"{country_code}_{data_source}_{admin_level}_PCT_ITN_ACCESS.csv",
                # bednet use files
                data_output_path
                / "bednets"
                / f"{country_code}_{data_source}_{admin_level}_PCT_ITN_USE.parquet",
                data_output_path / "bednets" / f"{country_code}_{data_source}_{admin_level}_PCT_ITN_USE.csv",
                # careseeking summary files
                data_output_path
                / "careseeking"
                / f"{country_code}_{data_source}_{admin_level}_PCT_CARESEEKING_SAMPLE_AVERAGE.parquet",
                data_output_path
                / "careseeking"
                / f"{country_code}_{data_source}_{admin_level}_PCT_CARESEEKING_SAMPLE_AVERAGE.csv",
                # no care files
                data_output_path
                / "careseeking"
                / f"{country_code}_{data_source}_{admin_level}_PCT_NO_CARE.parquet",
                data_output_path
                / "careseeking"
                / f"{country_code}_{data_source}_{admin_level}_PCT_NO_CARE.csv",
                # public care files
                data_output_path
                / "careseeking"
                / f"{country_code}_{data_source}_{admin_level}_PCT_PUBLIC_CARE.parquet",
                data_output_path
                / "careseeking"
                / f"{country_code}_{data_source}_{admin_level}_PCT_PUBLIC_CARE.csv",
                # private care files
                data_output_path
                / "careseeking"
                / f"{country_code}_{data_source}_{admin_level}_PCT_PRIVATE_CARE.parquet",
                data_output_path
                / "careseeking"
                / f"{country_code}_{data_source}_{admin_level}_PCT_PRIVATE_CARE.csv",
                # mortality files
                data_output_path
                / "mortality"
                / f"{country_code}_{data_source}_{admin_level}_U5MR_PERMIL.parquet",
                data_output_path
                / "mortality"
                / f"{country_code}_{data_source}_{admin_level}_U5MR_PERMIL.csv",
                # prevalence files
                data_output_path
                / "prevalence"
                / f"{country_code}_{data_source}_{admin_level}_PCT_U5_PREV_RDT.parquet",
                data_output_path
                / "prevalence"
                / f"{country_code}_{data_source}_{admin_level}_PCT_U5_PREV_RDT.csv",
                # DTP1 vaccination files
                data_output_path
                / "vaccination"
                / f"{country_code}_{data_source}_{admin_level}_PCT_DTP1.parquet",
                data_output_path / "vaccination" / f"{country_code}_{data_source}_{admin_level}_PCT_DTP1.csv",
                # DTP2 vaccination files
                data_output_path
                / "vaccination"
                / f"{country_code}_{data_source}_{admin_level}_PCT_DTP2.parquet",
                data_output_path / "vaccination" / f"{country_code}_{data_source}_{admin_level}_PCT_DTP2.csv",
                # DTP3 vaccination files
                data_output_path
                / "vaccination"
                / f"{country_code}_{data_source}_{admin_level}_PCT_DTP3.parquet",
                data_output_path / "vaccination" / f"{country_code}_{data_source}_{admin_level}_PCT_DTP3.csv",
                # DTP attrition/dropout files
                data_output_path
                / "vaccination"
                / f"{country_code}_{data_source}_{admin_level}_PCT_DROPOUT_DTP.parquet",
                data_output_path
                / "vaccination"
                / f"{country_code}_{data_source}_{admin_level}_PCT_DROPOUT_DTP.csv",
            ],
        )

    except Exception as e:
        current_run.log_error(f"Error occurred while executing the pipeline: {e}")
        raise


def run_dhs_indicator_notebooks(
    pipeline_root_path: Path, computation_notebook_name: str, reporting_notebook_name: str
) -> None:
    """Execute the computation notebook and generate a report using the reporting notebook.

    Args:
        pipeline_root_path (Path): Directory of the pipeline.
        computation_notebook_name (str): Filename of the computation notebook.
        reporting_notebook_name (str): Filename of the reporting notebook.

    """
    computation_notebook_path = pipeline_root_path / "code" / computation_notebook_name
    papermill_folder_path = pipeline_root_path / "papermill_outputs"
    reporting_folder_path = pipeline_root_path / "reporting"

    try:
        run_notebook(
            nb_path=computation_notebook_path,
            out_nb_path=papermill_folder_path,
            parameters=None,
        )
    except Exception as e:
        raise Exception(f"Error running computation notebook '{computation_notebook_name}': {e}") from e

    try:
        run_report_notebook(
            nb_file=reporting_folder_path / reporting_notebook_name,
            nb_output_path=reporting_folder_path / "outputs",
            nb_parameters=None,
        )
    except Exception as e:
        raise Exception(f"Error running reporting notebook '{reporting_notebook_name}': {e}") from e


if __name__ == "__main__":
    dhs_indicators()
