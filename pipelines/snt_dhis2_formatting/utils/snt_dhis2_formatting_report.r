# Shared helpers for snt_dhis2_formatting reporting notebook.

printdim <- function(df, name = deparse(substitute(df))) {
    cat("Dimensions of", name, ":", nrow(df), "rows x", ncol(df), "columns\n\n")
}

load_routine_data_report <- function(dataset_name, country_code) {
    routine_data <- tryCatch(
        {
            get_latest_dataset_file_in_memory(dataset_name, paste0(country_code, "_routine.parquet"))
        },
        error = function(e) {
            msg <- paste0(
                "[WARNING] Error while loading DHIS2 Routine data for: ",
                country_code,
                " the report cannot be executed. [ERROR DETAILS] ",
                conditionMessage(e)
            )
            stop(msg)
        }
    )

    log_msg(glue::glue(
        "DHIS2 routine file loaded from dataset: {dataset_name}. Dimensions: {nrow(routine_data)} rows, {ncol(routine_data)} columns."
    ))
    routine_data
}

detect_mad_outliers <- function(data_long, deviation = 15, outlier_column = "mad_flag") {
    data_long %>%
        dplyr::group_by(OU, indicator, YEAR) %>%
        dplyr::mutate(
            median_val = median(value, na.rm = TRUE),
            mad_val = mad(value, na.rm = TRUE),
            "{outlier_column}" := value > (median_val + deviation * mad_val) | value < (median_val - deviation * mad_val)
        ) %>%
        dplyr::ungroup()
}

create_dynamic_labels <- function(breaks) {
    fmt <- function(x) {
        format(x / 1000, big.mark = "'", scientific = FALSE, trim = TRUE)
    }

    c(
        paste0("< ", fmt(breaks[1]), "k"),
        paste0(fmt(breaks[-length(breaks)]), " - ", fmt(breaks[-1]), "k"),
        paste0("> ", fmt(breaks[length(breaks)]), "k")
    )
}
