#' Compute mean seasonal block proportion of annual cases.
#'
#' For each year and admin unit, computes the maximum forward block sum for the
#' selected duration, divides by annual total cases, and returns the mean ratio.
#'
#' @param admin_id Target administrative unit id.
#' @param block_duration Seasonal block duration in months.
#' @param row_data Row-level seasonality computations.
#' @param annual_data Annual totals by admin/year.
#' @param admin_col Column name for administrative unit id.
#' @param year_column Column name for year.
#' @return Numeric proportion (or NA when not computable).
compute_cases_proportion <- function(admin_id, block_duration, row_data, annual_data, admin_col, year_column) {
    if (is.na(block_duration) || is.infinite(block_duration)) {
        return(NA_real_)
    }

    sum_col <- paste("CASES_SUM", block_duration, "MTH_FW", sep = "_")
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


#' Compute dominant seasonal start month for an admin unit.
#'
#' Uses row-level seasonality flags for a selected block duration and returns
#' the most frequent month marked as seasonal.
#'
#' @param admin_id Target administrative unit id.
#' @param block_duration Seasonal block duration in months.
#' @param row_data Row-level seasonality computations.
#' @param admin_col Column name for administrative unit id.
#' @param month_column Column name for month.
#' @return Integer month (1-12) or NA when not computable.
compute_start_month <- function(admin_id, block_duration, row_data, admin_col, month_column) {
    if (is.na(block_duration) || is.infinite(block_duration)) {
        return(NA_integer_)
    }

    seasonality_row_col <- paste("CASES", block_duration, "MTH_ROW_SEASONALITY", sep = "_")
    if (!seasonality_row_col %in% names(row_data)) {
        return(NA_integer_)
    }

    admin_row_data <- row_data[get(admin_col) == admin_id]
    if (nrow(admin_row_data) == 0) {
        return(NA_integer_)
    }

    seasonal_months <- admin_row_data[get(seasonality_row_col) == 1, get(month_column)]
    if (length(seasonal_months) == 0) {
        return(NA_integer_)
    }

    month_counts <- table(seasonal_months)
    as.integer(names(month_counts)[which.max(month_counts)])
}


#' Add analyst-facing wide metrics to seasonality output.
#'
#' Computes CASES_PROPORTION and SEASONAL_BLOCK_START_MONTH, then enforces NA
#' for non-seasonal units in both derived columns.
#'
#' @param seasonality_wide_dt Wide seasonality table (by admin id).
#' @param row_seasonality_dt Row-level seasonality table.
#' @param imputed_dt Imputed input table used for annual totals.
#' @param admin_id_col Admin id column name.
#' @param year_col Year column name.
#' @param month_col Month column name.
#' @param seasonality_col Seasonality binary column name.
#' @param season_duration_col Block duration column name.
#' @param season_start_month_col Start month output column name.
#' @param cases_proportion_col Cases proportion output column name.
#' @param imputed_col Imputed values column name.
#' @return Updated data.table.
cases_add_wide_metrics <- function(
    seasonality_wide_dt,
    row_seasonality_dt,
    imputed_dt,
    admin_id_col,
    year_col,
    month_col,
    seasonality_col,
    season_duration_col,
    season_start_month_col,
    cases_proportion_col,
    imputed_col
) {
    annual_totals_dt <- imputed_dt[
        ,
        .(ANNUAL_TOTAL = sum(get(imputed_col), na.rm = TRUE)),
        by = c(admin_id_col, year_col)
    ]

    seasonality_wide_dt[, (cases_proportion_col) := mapply(
        compute_cases_proportion,
        admin_id = get(admin_id_col),
        block_duration = get(season_duration_col),
        MoreArgs = list(
            row_data = row_seasonality_dt,
            annual_data = annual_totals_dt,
            admin_col = admin_id_col,
            year_column = year_col
        )
    )]

    seasonality_wide_dt[, (season_start_month_col) := mapply(
        compute_start_month,
        admin_id = get(admin_id_col),
        block_duration = get(season_duration_col),
        MoreArgs = list(
            row_data = row_seasonality_dt,
            admin_col = admin_id_col,
            month_column = month_col
        )
    )]

    seasonality_wide_dt[
        get(seasonality_col) == 0 | is.na(get(seasonality_col)),
        c(cases_proportion_col, season_start_month_col) := .(NA_real_, NA_integer_)
    ]

    seasonality_wide_dt
}
