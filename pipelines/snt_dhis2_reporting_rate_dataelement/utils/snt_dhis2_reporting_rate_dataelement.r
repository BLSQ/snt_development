# Load base utils
# Bootstrap matches `snt_dhis2_population_transformation`: fixed-path `source()` of this
# file, `snt_environment <- get_setup_variables()`, then `load_snt_config()`.
# Helpers: `load_dataset_file()`, optional `paths_to_check` in the setup list.
source(file.path("~/workspace/code", "snt_utils.r"))


#' Get Setup Variables for SNT Workspace
#' Initializes workspace paths, loads R packages, and imports OpenHEXA SDK.
#'
#' @param SNT_ROOT_PATH Character. Root path of the SNT workspace. Default: '~/workspace'
#' @param packages Character vector. R packages to install and load.
#' @return List with `paths_to_check` plus `CONFIG_PATH`, `UPLOADS_PATH`, `DATA_PATH`
#'   (use as `snt_environment$CONFIG_PATH`, same pattern as population transformation).
#'
#' @export
get_setup_variables <- function(
    SNT_ROOT_PATH = "~/workspace",
    packages = c(
        "arrow", "rlang", "dplyr", "tidyr", "lubridate", "ggplot2",
        "stringr", "stringi", "jsonlite", "httr", "glue", "reticulate", "zoo"
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

    configure_conda_r_spatial_env()
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


#' Conda-friendly defaults for PROJ/GDAL (used when reading spatial data).
configure_conda_r_spatial_env <- function() {
    if (Sys.getenv("PROJ_LIB", "") == "") {
        Sys.setenv(PROJ_LIB = "/opt/conda/share/proj")
    }
    if (Sys.getenv("GDAL_DATA", "") == "") {
        Sys.setenv(GDAL_DATA = "/opt/conda/share/gdal")
    }
}


#' Standard aggregated indicator codes present in formatted routine extracts.
standard_dhis2_indicator_codes_for_dataelement <- function() {
    c("CONF", "PRES", "SUSP", "TEST")
}


#' Fail if Papermill did not inject `ROUTINE_FILE` and `DATASET_ID`.
#'
#' Kept as a named entry point so older notebooks that call this before
#' `parse_dataelement_snt_settings()` keep working after utils refactors.
assert_papermill_dataelement_params <- function() {
    required_pm <- c("ROUTINE_FILE", "DATASET_ID")
    missing_pm <- required_pm[!vapply(required_pm, exists, logical(1), inherits = TRUE)]
    if (length(missing_pm) > 0) {
        stop(
            "[ERROR] Missing pipeline parameters (Papermill): ",
            paste(missing_pm, collapse = ", "),
            ". Expected only ROUTINE_FILE and DATASET_ID from `snt_dhis2_reporting_rate_dataelement`."
        )
    }
}


#' Normalize optional `SNT_CONFIG$REPORTING_RATE_DATAELEMENT` list from JSON.
#'
#' When absent, uses the same defaults as the historical OpenHEXA parameters
#' (denominator `ROUTINE_ACTIVE_FACILITIES`, unweighted, activity CONF/PRES/SUSP,
#' volume CONF/PRES).
#'
#' Also calls `assert_papermill_dataelement_params()` (redundant if the notebook
#' already called it).
parse_dataelement_snt_settings <- function(config_json) {
    assert_papermill_dataelement_params()

    rc <- config_json$SNT_CONFIG$REPORTING_RATE_DATAELEMENT
    if (is.null(rc) || length(rc) == 0) {
        rc <- list()
    }

    denom <- rc$DATAELEMENT_METHOD_DENOMINATOR
    denom_ch <- if (is.null(denom)) "" else as.character(denom)[[1]]
    if (!nzchar(denom_ch) || is.na(denom_ch)) {
        denom <- "ROUTINE_ACTIVE_FACILITIES"
    } else {
        denom <- denom_ch
    }

    use_w <- rc$USE_WEIGHTED_REPORTING_RATES
    if (is.null(use_w)) {
        use_w <- FALSE
    } else {
        use_w <- isTRUE(use_w)
    }

    act <- rc$ACTIVITY_INDICATORS
    if (is.null(act)) {
        act <- c("CONF", "PRES", "SUSP")
    }
    act <- as.character(unlist(act, use.names = FALSE))

    vol <- rc$VOLUME_ACTIVITY_INDICATORS
    if (is.null(vol)) {
        vol <- c("CONF", "PRES")
    }
    vol <- as.character(unlist(vol, use.names = FALSE))

    list(
        COUNTRY_CODE = config_json$SNT_CONFIG$COUNTRY_CODE,
        ADMIN_1 = toupper(config_json$SNT_CONFIG$DHIS2_ADMINISTRATION_1),
        ADMIN_2 = toupper(config_json$SNT_CONFIG$DHIS2_ADMINISTRATION_2),
        DHIS2_INDICATORS = standard_dhis2_indicator_codes_for_dataelement(),
        DATAELEMENT_METHOD_DENOMINATOR = denom,
        USE_WEIGHTED_REPORTING_RATES = use_w,
        ACTIVITY_INDICATORS = act,
        VOLUME_ACTIVITY_INDICATORS = vol,
        fixed_cols = c("PERIOD", "YEAR", "MONTH", "ADM1_ID", "ADM2_ID", "OU_ID"),
        fixed_cols_rr = c("YEAR", "MONTH", "ADM2_ID", "REPORTING_RATE")
    )
}


stopifnot_nonempty_activity_indicators <- function(activity_indicators) {
    if (!length(activity_indicators)) {
        stop("[ERROR] No activity indicators selected; choose at least one (e.g. CONF).")
    }
}


validate_indicator_columns_in_routine <- function(
    dhis2_routine,
    activity_indicators,
    volume_activity_indicators
) {
    if (!all(activity_indicators %in% names(dhis2_routine))) {
        log_msg(
            glue::glue(
                "Warning: one or more activity indicators are missing from `dhis2_routine`: ",
                "{paste(activity_indicators, collapse = ', ')}"
            ),
            "warning"
        )
    }
    if (!all(volume_activity_indicators %in% names(dhis2_routine))) {
        msg <- glue::glue(
            "[ERROR] Volume activity indicator(s) not present in routine data: ",
            "{paste(volume_activity_indicators, collapse = ', ')}"
        )
        log_msg(msg, "error")
        stop(msg)
    }
}


#' YYYYMM sequence covering the routine period range (inclusive by month).
monthly_period_vector_from_routine <- function(dhis2_routine) {
    period_start <- min(dhis2_routine$PERIOD, na.rm = TRUE)
    period_end <- max(dhis2_routine$PERIOD, na.rm = TRUE)
    pv <- format(
        seq(lubridate::ym(period_start), lubridate::ym(period_end), by = "month"),
        "%Y%m"
    )
    list(
        PERIOD_START = period_start,
        PERIOD_END = period_end,
        period_vector = pv
    )
}


#' Write CSV + Parquet under `<DATA_PATH>/dhis2/reporting_rate/`.
write_reporting_rate_dataelement_outputs <- function(reporting_rate_tbl, snt_environment, country_code) {
    output_dir <- file.path(snt_environment$DATA_PATH, "dhis2", "reporting_rate")
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    base <- paste0(country_code, "_reporting_rate_dataelement")
    csv_path <- file.path(output_dir, paste0(base, ".csv"))
    pq_path <- file.path(output_dir, paste0(base, ".parquet"))
    utils::write.csv(reporting_rate_tbl, csv_path, row.names = FALSE)
    log_msg(glue::glue("Exported: {csv_path}"))
    arrow::write_parquet(reporting_rate_tbl, pq_path)
    log_msg(glue::glue("Exported: {pq_path}"))
    invisible(list(csv_path = csv_path, parquet_path = pq_path))
}


build_facility_master_dataelement <- function(
    dhis2_pyramid_formatted,
    period_vector,
    config_json,
    ADMIN_1,
    ADMIN_2
) {
    dhis2_pyramid_formatted %>%
        dplyr::rename(
            OU_ID = glue::glue("LEVEL_{config_json$SNT_CONFIG$ANALYTICS_ORG_UNITS_LEVEL}_ID"),
            OU_NAME = glue::glue("LEVEL_{config_json$SNT_CONFIG$ANALYTICS_ORG_UNITS_LEVEL}_NAME"),
            ADM2_ID = stringr::str_replace(ADMIN_2, "NAME", "ID"),
            ADM2_NAME = dplyr::all_of(ADMIN_2),
            ADM1_ID = stringr::str_replace(ADMIN_1, "NAME", "ID"),
            ADM1_NAME = dplyr::all_of(ADMIN_1)
        ) %>%
        dplyr::select(ADM1_ID, ADM1_NAME, ADM2_ID, ADM2_NAME, OU_ID, OU_NAME, OPENING_DATE, CLOSED_DATE) %>%
        dplyr::distinct() %>%
        tidyr::crossing(PERIOD = period_vector) %>%
        dplyr::mutate(PERIOD = as.numeric(PERIOD))
}
