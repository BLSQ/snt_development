# Main helpers for median outliers imputation pipeline.
bootstrap_outliers_context <- function(
    root_path = "~/workspace",
    required_packages = c(
        "data.table", "arrow", "tidyverse", "jsonlite", "DBI", "RPostgres",
        "reticulate", "glue", "zoo"
    ),
    load_openhexa = TRUE
) {
    code_path <- file.path(root_path, "code")
    config_path <- file.path(root_path, "configuration")
    data_path <- file.path(root_path, "data")
    output_dir <- file.path(data_path, "dhis2", "outliers_imputation")
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

    source(file.path(code_path, "snt_utils.r"))
    install_and_load(required_packages)

    Sys.setenv(PROJ_LIB = "/opt/conda/share/proj")
    Sys.setenv(GDAL_DATA = "/opt/conda/share/gdal")
    Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")

    openhexa <- NULL
    if (load_openhexa) {
        openhexa <- reticulate::import("openhexa.sdk")
    }

    config_json <- tryCatch(
        {
            jsonlite::fromJSON(file.path(config_path, "SNT_config.json"))
        },
        error = function(e) {
            msg <- glue::glue("[ERROR] Error while loading configuration {conditionMessage(e)}")
            log_msg(msg)
            stop(msg)
        }
    )

    return(list(
        ROOT_PATH = root_path,
        CODE_PATH = code_path,
        CONFIG_PATH = config_path,
        DATA_PATH = data_path,
        OUTPUT_DIR = output_dir,
        openhexa = openhexa,
        config_json = config_json
    ))
}

load_routine_data <- function(dataset_name, country_code, required_indicators = NULL, cast_year_month = TRUE) {
    dhis2_routine <- tryCatch(
        {
            get_latest_dataset_file_in_memory(dataset_name, paste0(country_code, "_routine.parquet"))
        },
        error = function(e) {
            msg <- glue::glue("[ERROR] Error while loading DHIS2 routine data file for {country_code} : {conditionMessage(e)}")
            log_msg(msg)
            stop(msg)
        }
    )

    log_msg(glue::glue("DHIS2 routine data loaded from dataset : {dataset_name}"))
    log_msg(glue::glue("DHIS2 routine data loaded has dimensions: {nrow(dhis2_routine)} rows, {ncol(dhis2_routine)} columns."))

    if (cast_year_month && all(c("YEAR", "MONTH") %in% colnames(dhis2_routine))) {
        dhis2_routine[c("YEAR", "MONTH")] <- lapply(dhis2_routine[c("YEAR", "MONTH")], as.integer)
    }

    if (!is.null(required_indicators)) {
        missing_indicators <- setdiff(required_indicators, colnames(dhis2_routine))
        if (length(missing_indicators) > 0) {
            msg <- paste("[ERROR] Missing indicator column(s) in routine data:", paste(missing_indicators, collapse = ", "))
            log_msg(msg)
            stop(msg)
        }
    }

    dhis2_routine
}

impute_outliers_dt <- function(dt, outlier_col) {
    dt <- data.table::as.data.table(dt)
    data.table::setorder(dt, ADM1_ID, ADM2_ID, OU_ID, INDICATOR, PERIOD, YEAR, MONTH)
    dt[, TO_IMPUTE := data.table::fifelse(get(outlier_col) == TRUE, NA_real_, VALUE)]
    dt[, MEDIAN_IMPUTED := data.table::frollapply(
        TO_IMPUTE,
        n = 3,
        FUN = function(x) ceiling(median(x, na.rm = TRUE)),
        align = "center"
    ), by = .(ADM1_ID, ADM2_ID, OU_ID, INDICATOR)]
    dt[, VALUE_IMPUTED := data.table::fifelse(is.na(TO_IMPUTE), MEDIAN_IMPUTED, TO_IMPUTE)]
    dt[, c("TO_IMPUTE", "MEDIAN_IMPUTED") := NULL]
    return(as.data.frame(data.table::copy(dt)))
}

format_routine_data_selection <- function(
    df,
    outlier_column,
    DHIS2_INDICATORS,
    fixed_cols,
    pyramid_names,
    remove = FALSE
) {
    if (remove) {
        df <- df %>% dplyr::filter(!.data[[outlier_column]])
    }
    target_cols <- c(
        "PERIOD", "YEAR", "MONTH", "ADM1_NAME", "ADM1_ID",
        "ADM2_NAME", "ADM2_ID", "OU_ID", "OU_NAME", DHIS2_INDICATORS
    )
    output <- df %>%
        dplyr::select(-VALUE) %>%
        dplyr::rename(VALUE = VALUE_IMPUTED) %>%
        dplyr::select(dplyr::all_of(fixed_cols), INDICATOR, VALUE) %>%
        dplyr::mutate(VALUE = ifelse(is.nan(VALUE), NA_real_, VALUE)) %>%
        tidyr::pivot_wider(names_from = "INDICATOR", values_from = "VALUE") %>%
        dplyr::left_join(pyramid_names, by = c("ADM1_ID", "ADM2_ID", "OU_ID"))
    return(output %>% dplyr::select(dplyr::all_of(intersect(target_cols, names(output)))))
}

