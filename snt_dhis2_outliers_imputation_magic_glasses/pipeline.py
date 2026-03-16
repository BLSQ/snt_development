from pathlib import Path
import time
import math

import pandas as pd
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


def _materialize_standard_outputs_from_legacy(
    legacy_file: Path, output_dir: Path, country_code: str, use_complete_flag: bool
) -> None:
    """Build standard outliers outputs from legacy MG flagged table."""
    df = pd.read_parquet(legacy_file)
    flag_col = (
        "OUTLIER_MAGIC_GLASSES_COMPLETE" if use_complete_flag else "OUTLIER_MAGIC_GLASSES_PARTIAL"
    )
    if flag_col not in df.columns:
        raise RuntimeError(f"Legacy file does not contain expected flag column: {flag_col}")

    fixed_cols = ["PERIOD", "YEAR", "MONTH", "ADM1_ID", "ADM2_ID", "OU_ID"]
    required_cols = [*fixed_cols, "INDICATOR", "VALUE", flag_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise RuntimeError(f"Legacy file missing required columns: {', '.join(missing)}")

    # 1) detected subset
    detected = df[df[flag_col] == True].copy()  # noqa: E712
    detected.to_parquet(output_dir / f"{country_code}_routine_outliers_detected.parquet", index=False)

    # 2) removed (set outliers to NA)
    removed_long = df.copy()
    removed_long.loc[removed_long[flag_col] == True, "VALUE"] = pd.NA  # noqa: E712

    # 3) imputed (same approach as other pipelines: centered moving average n=3)
    imputed_long = df.copy()
    imputed_long["TO_IMPUTE"] = imputed_long["VALUE"].where(imputed_long[flag_col] != True, pd.NA)  # noqa: E712
    sort_cols = ["ADM1_ID", "ADM2_ID", "OU_ID", "INDICATOR", "PERIOD", "YEAR", "MONTH"]
    imputed_long = imputed_long.sort_values(sort_cols)

    group_cols = ["ADM1_ID", "ADM2_ID", "OU_ID", "INDICATOR"]
    moving_avg = (
        imputed_long.groupby(group_cols)["TO_IMPUTE"]
        .apply(lambda s: s.rolling(window=3, center=True, min_periods=3).mean())
        .reset_index(level=group_cols, drop=True)
    )
    imputed_long["MOVING_AVG"] = moving_avg.apply(
        lambda x: pd.NA if pd.isna(x) else float(math.ceil(x))
    )
    imputed_long["VALUE"] = imputed_long["TO_IMPUTE"].where(imputed_long["TO_IMPUTE"].notna(), imputed_long["MOVING_AVG"])

    def _to_wide(dt: pd.DataFrame) -> pd.DataFrame:
        wide = dt.pivot_table(
            index=fixed_cols,
            columns="INDICATOR",
            values="VALUE",
            aggfunc="first",
        ).reset_index()
        wide.columns.name = None
        return wide

    _to_wide(imputed_long).to_parquet(output_dir / f"{country_code}_routine_outliers_imputed.parquet", index=False)
    _to_wide(removed_long).to_parquet(output_dir / f"{country_code}_routine_outliers_removed.parquet", index=False)


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
            # Avoid publishing stale artifacts from previous runs.
            for old_file in data_path.glob(f"{country_code}_routine_outliers*.parquet"):
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

            expected_outputs = [
                data_path / f"{country_code}_routine_outliers_detected.parquet",
                data_path / f"{country_code}_routine_outliers_imputed.parquet",
                data_path / f"{country_code}_routine_outliers_removed.parquet",
            ]
            missing_outputs = [
                path.name
                for path in expected_outputs
                if (not path.exists() or path.stat().st_mtime < run_start_ts)
            ]
            if missing_outputs:
                # Fallback for legacy MG notebooks still writing legacy file names.
                legacy_data_path = root_path / "data" / "dhis2" / "outliers_detection"
                legacy_candidates = [
                    data_path / f"{country_code}_flagged_outliers_magic_glasses.parquet",
                    legacy_data_path / f"{country_code}_flagged_outliers_magic_glasses.parquet",
                ]
                legacy_file = next(
                    (p for p in legacy_candidates if p.exists() and p.stat().st_mtime >= run_start_ts),
                    None,
                )

                if legacy_file is not None:
                    current_run.log_warning(
                        "Legacy MG output detected. Rebuilding standard outputs from "
                        f"{legacy_file}."
                    )
                    _materialize_standard_outputs_from_legacy(
                        legacy_file=legacy_file,
                        output_dir=data_path,
                        country_code=country_code,
                        use_complete_flag=run_mg_complete,
                    )
                    missing_outputs = [
                        path.name
                        for path in expected_outputs
                        if (not path.exists() or path.stat().st_mtime < run_start_ts)
                    ]

                if missing_outputs:
                    # Log available artifacts for quicker debugging.
                    available_new = sorted(p.name for p in data_path.glob(f"{country_code}_*.parquet"))
                    available_legacy = (
                        sorted(p.name for p in legacy_data_path.glob(f"{country_code}_*.parquet"))
                        if legacy_data_path.exists()
                        else []
                    )
                    current_run.log_warning(
                        "Available parquet files in outliers_imputation: "
                        + (", ".join(available_new) if available_new else "none")
                    )
                    current_run.log_warning(
                        "Available parquet files in outliers_detection: "
                        + (", ".join(available_legacy) if available_legacy else "none")
                    )
                    raise RuntimeError(
                        "Expected output files were not generated during this run: "
                        + ", ".join(missing_outputs)
                    )

            parameters_file = save_pipeline_parameters(
                pipeline_name="snt_dhis2_outliers_imputation_magic_glasses",
                parameters=input_params,
                output_path=data_path,
                country_code=country_code,
            )

            add_files_to_dataset(
                dataset_id=snt_config["SNT_DATASET_IDENTIFIERS"]["DHIS2_OUTLIERS_IMPUTATION"],
                country_code=country_code,
                file_paths=[
                    *data_path.glob(f"{country_code}_routine_outliers*.parquet"),
                    parameters_file,
                ],
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
