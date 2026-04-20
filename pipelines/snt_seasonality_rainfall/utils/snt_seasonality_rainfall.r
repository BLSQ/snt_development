# Helpers for rainfall seasonality code notebook.


# Compute average proportion of annual rainfall falling in the selected block.
rainfall_compute_rain_proportion_for_admin <- function(
    admin_id,
    block_duration,
    row_data,
    annual_data,
    admin_col,
    year_column,
    indicator_name = "RAINFALL"
) {
    if (is.na(block_duration) || is.infinite(block_duration)) {
        return(NA_real_)
    }

    sum_col_candidates <- c(
        paste(indicator_name, "SUM", block_duration, "MTH_FW", sep = "_"),
        paste(indicator_name, block_duration, "MTH_FW", sep = "_")
    )
    sum_col <- sum_col_candidates[sum_col_candidates %in% names(row_data)][1]
    if (is.na(sum_col) || !nzchar(sum_col)) {
        return(NA_real_)
    }

    admin_row_data <- row_data[get(admin_col) == admin_id]
    admin_annual_data <- annual_data[get(admin_col) == admin_id]
    if (nrow(admin_row_data) == 0L || nrow(admin_annual_data) == 0L) {
        return(NA_real_)
    }

    yearly_max_block <- admin_row_data[
        !is.na(get(sum_col)),
        .(max_block_sum = if (.N > 0L) max(get(sum_col), na.rm = TRUE) else NA_real_),
        by = year_column
    ]
    yearly_max_block <- yearly_max_block[is.finite(max_block_sum)]
    if (nrow(yearly_max_block) == 0L) {
        return(NA_real_)
    }

    merged <- merge(yearly_max_block, admin_annual_data, by = year_column)
    merged <- merged[ANNUAL_TOTAL > 0]
    if (nrow(merged) == 0L) {
        return(NA_real_)
    }

    merged[, prop := max_block_sum / ANNUAL_TOTAL]
    mean(merged$prop, na.rm = TRUE)
}


# Compute the most frequent first onset month across years for one admin unit.
rainfall_compute_start_month_for_admin <- function(
    admin_id,
    block_duration,
    row_data,
    admin_col,
    year_column,
    month_column,
    indicator_name = "RAINFALL"
) {
    if (is.na(block_duration) || is.infinite(block_duration)) {
        return(NA_integer_)
    }

    seasonality_row_col <- paste(indicator_name, block_duration, "MTH_ROW_SEASONALITY", sep = "_")
    if (!seasonality_row_col %in% names(row_data)) {
        return(NA_integer_)
    }

    admin_row_data <- row_data[get(admin_col) == admin_id]
    if (nrow(admin_row_data) == 0L) {
        return(NA_integer_)
    }

    first_seasonal_months_by_year <- admin_row_data[
        get(seasonality_row_col) == 1,
        .(first_month = min(get(month_column))),
        by = year_column
    ]$first_month

    if (length(first_seasonal_months_by_year) == 0L) {
        return(NA_integer_)
    }

    month_counts <- table(first_seasonal_months_by_year)
    as.integer(names(month_counts)[which.max(month_counts)])
}


# Add rain proportion and seasonal start month columns to wide seasonality output.
rainfall_add_wide_metrics <- function(
    seasonality_wide_dt,
    row_seasonality_dt,
    imputed_dt,
    admin_id_col,
    year_col,
    month_col,
    seasonality_col,
    season_duration_col,
    season_start_month_col,
    rain_proportion_col,
    imputed_col,
    indicator_name = "RAINFALL"
) {
    annual_totals_dt <- imputed_dt[
        ,
        .(ANNUAL_TOTAL = sum(get(imputed_col), na.rm = TRUE)),
        by = c(admin_id_col, year_col)
    ]

    seasonality_wide_dt[, (rain_proportion_col) := mapply(
        rainfall_compute_rain_proportion_for_admin,
        admin_id = get(admin_id_col),
        block_duration = get(season_duration_col),
        MoreArgs = list(
            row_data = row_seasonality_dt,
            annual_data = annual_totals_dt,
            admin_col = admin_id_col,
            year_column = year_col,
            indicator_name = indicator_name
        )
    )]

    seasonality_wide_dt[, (season_start_month_col) := mapply(
        rainfall_compute_start_month_for_admin,
        admin_id = get(admin_id_col),
        block_duration = get(season_duration_col),
        MoreArgs = list(
            row_data = row_seasonality_dt,
            admin_col = admin_id_col,
            year_column = year_col,
            month_column = month_col,
            indicator_name = indicator_name
        )
    )]

    seasonality_wide_dt[
        get(seasonality_col) == 0 | is.na(get(seasonality_col)),
        c(rain_proportion_col, season_start_month_col) := .(NA_real_, NA_integer_)
    ]

    seasonality_wide_dt
}
