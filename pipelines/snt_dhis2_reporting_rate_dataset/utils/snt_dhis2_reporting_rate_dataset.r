# Load base utils
# Same bootstrap pattern as reporting_rate_dataelement / formatting review:
# `load_dataset_file()`, `load_snt_config()`, `get_setup_variables()` + paths_to_check.
source(file.path("~/workspace/code", "snt_utils.r"))


#' Get Setup Variables for SNT Workspace
#' Initializes workspace paths, loads R packages, and imports OpenHEXA SDK.
#'
#' @param SNT_ROOT_PATH Character. Root path of the SNT workspace. Default: '~/workspace'
#' @param packages Character vector. R packages to install and load.
#' @return List with `paths_to_check` (CONFIG_PATH, UPLOADS_PATH, DATA_PATH) and the
#'   same three paths at the top level for backward compatibility (`setup$CONFIG_PATH`, …).
#'
#' @export
get_setup_variables <- function(
    SNT_ROOT_PATH = "~/workspace",
    packages = c(
        "arrow", "dplyr", "tidyr", "ggplot2",
        "stringr", "stringi", "jsonlite", "httr", "glue", "reticulate"
    )
) {
    paths_to_check <- list(
        CONFIG_PATH  = file.path(SNT_ROOT_PATH, "configuration"),
        UPLOADS_PATH = file.path(SNT_ROOT_PATH, "uploads"),
        DATA_PATH    = file.path(SNT_ROOT_PATH, "data")
    )
    setup_variable <- c(
        list(paths_to_check = paths_to_check),
        paths_to_check
    )

    install_and_load(packages)

    if (Sys.getenv("PROJ_LIB", "") == "") {
        Sys.setenv(PROJ_LIB = "/opt/conda/share/proj")
    }
    if (Sys.getenv("GDAL_DATA", "") == "") {
        Sys.setenv(GDAL_DATA = "/opt/conda/share/gdal")
    }
    Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")
    reticulate::py_config()$python
    assign("openhexa", reticulate::import("openhexa.sdk"), envir = .GlobalEnv)

    return(setup_variable)
}


#' Load SNT Configuration File
#' Reads and parses a JSON configuration file.
#' @param snt_config_path Character. Path to the configuration JSON file.
#' @return List containing parsed configuration.
#'
#' @export
load_snt_config <- function(snt_config_path) {
    config_json <- tryCatch(
        { jsonlite::fromJSON(snt_config_path) },
        error = function(e) {
            msg <- glue::glue("[ERROR] Error while loading configuration: {snt_config_path}")
            cat(msg)
            stop(msg)
        }
    )
    log_msg(paste0("SNT configuration loaded from: ", snt_config_path))
    return(config_json)
}


#' Fail if Papermill did not inject `ROUTINE_FILE` and `DATASET_ID`.
#'
#' Kept as a named entry point so older notebooks that call this before other
#' setup keep working after utils refactors.
assert_papermill_reporting_rate_dataset_params <- function() {
    required <- c("ROUTINE_FILE", "DATASET_ID")
    missing <- required[!vapply(required, exists, logical(1), inherits = TRUE)]
    if (length(missing) > 0) {
        stop(
            "[ERROR] Missing pipeline parameters (Papermill): ",
            paste(missing, collapse = ", "),
            ". Expected only ROUTINE_FILE and DATASET_ID from `snt_dhis2_reporting_rate_dataset`."
        )
    }
}


#' Build globals used in the dataset reporting-rate notebook from `SNT_config.json`.
#'
#' Calls `assert_papermill_reporting_rate_dataset_params()` first (redundant if the
#' notebook already called it).
parse_reporting_rate_dataset_snt_settings <- function(config_json) {
    assert_papermill_reporting_rate_dataset_params()

    list(
        COUNTRY_CODE = config_json$SNT_CONFIG$COUNTRY_CODE,
        ADMIN_1 = toupper(config_json$SNT_CONFIG$DHIS2_ADMINISTRATION_1),
        ADMIN_2 = toupper(config_json$SNT_CONFIG$DHIS2_ADMINISTRATION_2),
        REPORTING_RATE_PRODUCT_ID = config_json$SNT_CONFIG$REPORTING_RATE_PRODUCT_UID,
        fixed_cols_rr = c("YEAR", "MONTH", "ADM2_ID", "REPORTING_RATE")
    )
}


#' Load Dataset File from OpenHEXA
#' Retrieves the latest version of a file from an OpenHEXA dataset.
#'
#' @param dataset_id Character. OpenHEXA dataset identifier.
#' @param filename Character. Name of file to load.
#' @return Dataframe containing the loaded data.
#'
#' @export
load_dataset_file <- function(dataset_id, filename) {
    data <- tryCatch(
        { get_latest_dataset_file_in_memory(dataset_id, filename) },
        error = function(e) {
            msg <- glue::glue("[ERROR] Error while loading {filename} file: {conditionMessage(e)}")
            log_msg(msg, "error")
            stop(msg)
        }
    )
    msg <- glue::glue("{filename} data loaded from dataset: {dataset_id} dataframe dimensions: [{paste(dim(data), collapse = ', ')}]")
    log_msg(msg)
    return(data)
}


#' Write CSV + Parquet under `<DATA_PATH>/dhis2/reporting_rate/`.
write_reporting_rate_dataset_outputs <- function(reporting_rate_tbl, snt_environment, country_code) {
    output_dir <- file.path(snt_environment$DATA_PATH, "dhis2", "reporting_rate")
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    base <- paste0(country_code, "_reporting_rate_dataset")
    csv_path <- file.path(output_dir, paste0(base, ".csv"))
    pq_path <- file.path(output_dir, paste0(base, ".parquet"))
    utils::write.csv(reporting_rate_tbl, csv_path, row.names = FALSE)
    log_msg(glue::glue("Exported: {csv_path}"))
    arrow::write_parquet(reporting_rate_tbl, pq_path)
    log_msg(glue::glue("Exported: {pq_path}"))
    invisible(list(csv_path = csv_path, parquet_path = pq_path))
}
