compute_rain_proportion <- function(admin_id, block_duration, row_data, annual_data, admin_col, year_column) {
    if (is.na(block_duration) || is.infinite(block_duration)) {
        return(NA_real_)
    }

    sum_col <- paste("RAINFALL_SUM", block_duration, "MTH_FW", sep = "_")
    if (!sum_col %in% names(row_data)) {
        return(NA_real_)
    }

    admin_row_data <- row_data[get(admin_col) == admin_id]
    admin_annual_data <- annual_data[get(admin_col) == admin_id]
    if (nrow(admin_row_data) == 0 || nrow(admin_annual_data) == 0) {
        return(NA_real_)
    }

    yearly_max_block <- admin_row_data[
        !is.na(get(sum_col)),
        .(max_block_sum = if (.N > 0L) max(get(sum_col), na.rm = TRUE) else NA_real_),
        by = year_column
    ]

    yearly_max_block <- yearly_max_block[is.finite(max_block_sum)]
    if (nrow(yearly_max_block) == 0) {
        return(NA_real_)
    }

    merged <- merge(yearly_max_block, admin_annual_data, by = year_column)
    merged <- merged[ANNUAL_TOTAL > 0]
    if (nrow(merged) == 0) {
        return(NA_real_)
    }

    merged[, prop := max_block_sum / ANNUAL_TOTAL]
    mean(merged$prop, na.rm = TRUE)
}


compute_start_month <- function(admin_id, block_duration, row_data, admin_col, year_column, month_column) {
    if (is.na(block_duration) || is.infinite(block_duration)) {
        return(NA_integer_)
    }

    seasonality_row_col <- paste("RAINFALL", block_duration, "MTH_ROW_SEASONALITY", sep = "_")
    if (!seasonality_row_col %in% names(row_data)) {
        return(NA_integer_)
    }

    admin_row_data <- row_data[get(admin_col) == admin_id]
    if (nrow(admin_row_data) == 0) {
        return(NA_integer_)
    }

    first_seasonal_months_by_year <- admin_row_data[
        get(seasonality_row_col) == 1,
        .(first_month = min(get(month_column))),
        by = year_column
    ]$first_month

    if (length(first_seasonal_months_by_year) == 0) {
        return(NA_integer_)
    }

    month_counts <- table(first_seasonal_months_by_year)
    as.integer(names(month_counts)[which.max(month_counts)])
}
