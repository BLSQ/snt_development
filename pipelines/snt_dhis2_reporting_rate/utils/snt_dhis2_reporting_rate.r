# Shared helpers for snt_dhis2_reporting_rate notebooks.

inspect_reporting_rate <- function(data_tibble) {
    tibble_name_full <- deparse(substitute(data_tibble))
    method <- stringr::str_extract(tibble_name_full, "(?<=reporting_rate_).*")

    values_greater_than_1 <- sum(data_tibble$REPORTING_RATE > 1, na.rm = TRUE)
    total_values <- length(data_tibble$REPORTING_RATE)

    if (total_values > 0) {
        proportion <- values_greater_than_1 / total_values * 100
        min_rate <- min(data_tibble$REPORTING_RATE, na.rm = TRUE)
        max_rate <- max(data_tibble$REPORTING_RATE, na.rm = TRUE)
    } else {
        proportion <- 0
        min_rate <- NA
        max_rate <- NA
    }

    clarification <- if (proportion == 0) NULL else " (there are more reports than expected)"

    log_msg(
        paste0(
            "🔍 For reporting rate method : `", method, "`, the values of REPORTING_RATE range from ", round(min_rate, 2),
            " to ", round(max_rate, 2),
            ", and ", round(proportion, 2), " % of values are >1", clarification, "."
        )
    )

    hist(data_tibble$REPORTING_RATE, breaks = 50)
}

is_aire_l5 <- function(x) {
    stringr::str_detect(x, stringr::regex("^\\s*aire[^a-zA-Z]?", ignore_case = TRUE))
}

is_hospital_l4 <- function(x) {
    stringr::str_detect(x, stringr::regex("^(hd|chr|chu|hgr)", ignore_case = TRUE))
}

snt_write_csv <- function(x, output_data_path, method, country_code = NULL) {
    if (is.null(country_code) && exists("COUNTRY_CODE")) {
        country_code <- get("COUNTRY_CODE")
    }
    if (is.null(country_code)) {
        stop("country_code is required to export reporting rate csv.")
    }

    full_directory_path <- file.path(output_data_path, "reporting_rate")
    if (!dir.exists(full_directory_path)) {
        dir.create(full_directory_path, recursive = TRUE)
    }

    file_path <- file.path(full_directory_path, paste0(country_code, "_reporting_rate_", method, ".csv"))
    readr::write_csv(x, file_path)
    log_msg(paste0("Exported : ", file_path))
}

snt_write_parquet <- function(x, output_data_path, method, country_code = NULL) {
    if (is.null(country_code) && exists("COUNTRY_CODE")) {
        country_code <- get("COUNTRY_CODE")
    }
    if (is.null(country_code)) {
        stop("country_code is required to export reporting rate parquet.")
    }

    full_directory_path <- file.path(output_data_path, "reporting_rate")
    if (!dir.exists(full_directory_path)) {
        dir.create(full_directory_path, recursive = TRUE)
    }

    file_path <- file.path(full_directory_path, paste0(country_code, "_reporting_rate_", method, ".parquet"))
    arrow::write_parquet(x, file_path)
    log_msg(paste0("Exported : ", file_path))
}

printdim <- function(df, name = deparse(substitute(df))) {
    cat("Dimensions of", name, ":", nrow(df), "rows x", ncol(df), "columns\n\n")
}
