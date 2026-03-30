# Shared helpers for snt_dhis2_incidence notebooks.

resolve_routine_filename <- function(routine_choice) {
    if (routine_choice == "raw") return("_routine.parquet")
    is_removed <- FALSE
    if (routine_choice == "raw_without_outliers") is_removed <- TRUE
    removed_status <- if (is_removed) "removed" else "imputed"
    return(glue::glue("_routine_outliers_{removed_status}.parquet"))
}

infer_reporting_routine_choice <- function(reporting_parameters) {
    if (is.null(reporting_parameters)) return(NULL)
    if (length(names(reporting_parameters)) == 0) return(NULL)

    if ("ROUTINE_FILE" %in% names(reporting_parameters)) {
        routine_file <- as.character(reporting_parameters$ROUTINE_FILE[[1]])
        if (grepl("_routine_outliers_removed\\.parquet$", routine_file)) return("raw_without_outliers")
        if (grepl("_routine_outliers_imputed\\.parquet$", routine_file)) return("imputed")
        if (grepl("_routine\\.parquet$", routine_file)) return("raw")
    }

    if ("REPORTING_RATE_METHOD" %in% names(reporting_parameters)) return("raw")
    return(NULL)
}

save_yearly_incidence <- function(yearly_incidence, data_path, file_extension, write_function, country_code = NULL) {
    if (is.null(country_code) && exists("COUNTRY_CODE")) {
        country_code <- get("COUNTRY_CODE")
    }
    if (is.null(country_code)) {
        stop("country_code is required to export yearly incidence.")
    }

    file_name <- paste0(country_code, "_incidence", file_extension)
    file_path <- file.path(data_path, file_name)
    output_dir <- dirname(file_path)

    if (!dir.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE)
    }

    write_function(yearly_incidence, file_path)
    log_msg(paste0("Exporting : ", file_path))
}
