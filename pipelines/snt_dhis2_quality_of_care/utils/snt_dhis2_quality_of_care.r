# ================================================
# Title: Main helpers for the Quality of Care pipeline
# Description: Routine data typing, district-year aggregation, output saving and yearly maps,
#   sourced by the pipeline notebook and the reporting notebook.
# Dependencies: data.table, sf, dplyr, ggplot2, arrow, glue
# Requires: code/snt_utils.r, loaded by the notebooks before this file.
# ================================================


#' Validate the Quality of Care Data Action Parameter
#'
#' Checks that the routine data choice is one of the outliers pipelines' outputs
#' (`imputed` or `removed`). A NULL or empty value falls back to `imputed`; any
#' other value stops execution with an `[ERROR]` message.
#'
#' @param data_action Character. Routine data choice, `imputed` or `removed`.
#' @return Character. The validated data action.
#'
#' @export
validate_quality_of_care_action <- function(data_action) {
    if (is.null(data_action) || !nzchar(data_action)) {
        return("imputed")
    }
    allowed_actions <- c("imputed", "removed")
    if (!(data_action %in% allowed_actions)) {
        stop(glue::glue("[ERROR] Invalid data_action `{data_action}`. Allowed: {paste(allowed_actions, collapse = ', ')}"))
    }
    data_action
}


#' Normalize Column Types of the Quality of Care Routine Data
#'
#' Converts the routine data to a data.table (by reference) and casts the
#' available indicator columns to numeric, treating empty strings and `-` as NA.
#' Also casts `YEAR` to integer and `ADM2_ID` to character.
#'
#' @param routine Data frame. Routine data loaded from the outliers dataset.
#' @param indicator_cols Character vector. Indicator columns to cast to numeric;
#'   columns absent from `routine` are skipped.
#' @return data.table. The routine data with normalized column types.
#'
#' @export
normalize_qoc_routine_types <- function(routine, indicator_cols) {
    data.table::setDT(routine)
    available_cols <- intersect(indicator_cols, names(routine))

    for (col in available_cols) {
        col_vals <- as.character(routine[[col]])
        col_vals[is.na(col_vals) | col_vals == "" | col_vals == "-"] <- NA_character_
        routine[, (col) := as.numeric(col_vals)]
    }

    routine[, YEAR := as.integer(YEAR)]
    routine[, ADM2_ID := as.character(ADM2_ID)]
    routine
}


#' Aggregate Quality of Care Indicators by District and Year
#'
#' Sums the available indicator columns by `ADM2_ID` and `YEAR`, ignoring NA
#' values. When none of the indicator columns are present, returns the unique
#' district-year pairs only.
#'
#' @param routine data.table. Routine data with normalized types (see
#'   `normalize_qoc_routine_types()`).
#' @param indicator_cols Character vector. Indicator columns to sum; must match
#'   the vector used in `normalize_qoc_routine_types()`.
#' @return data.table. One row per `ADM2_ID` and `YEAR` with summed indicators.
#'
#' @export
aggregate_qoc_district_year <- function(routine, indicator_cols) {
    available_cols <- intersect(indicator_cols, names(routine))

    if (length(available_cols) > 0) {
        routine[, lapply(.SD, function(x) sum(x, na.rm = TRUE)), .SDcols = available_cols, by = .(ADM2_ID, YEAR)]
    } else {
        unique(routine[, .(ADM2_ID, YEAR)])
    }
}


#' Attach District Names to the Quality of Care Table
#'
#' Left-joins `ADM2_NAME` from the shapes onto the quality-of-care table by
#' `ADM2_ID`. The table is returned unchanged if the shapes lack either column.
#'
#' @param qoc_dt data.table. District-year quality-of-care indicators.
#' @param shapes_sf sf. District shapes with `ADM2_ID` and `ADM2_NAME` columns.
#' @return data.table. The quality-of-care table, with `ADM2_NAME` when available.
#'
#' @export
attach_quality_of_care_shapes <- function(qoc_dt, shapes_sf) {
    shapes_dt <- data.table::as.data.table(sf::st_drop_geometry(shapes_sf))
    if ("ADM2_ID" %in% names(shapes_dt) && "ADM2_NAME" %in% names(shapes_dt)) {
        shapes_dt[, ADM2_ID := as.character(ADM2_ID)]
        qoc_dt <- merge(qoc_dt, unique(shapes_dt[, .(ADM2_ID, ADM2_NAME)]), by = "ADM2_ID", all.x = TRUE)
    }
    qoc_dt
}


#' Save the District-Year Quality of Care Outputs
#'
#' Writes the quality-of-care table as parquet and as its csv twin, named
#' `{country_code}_quality_of_care_district_year_{data_action}`, and logs the
#' saved paths.
#'
#' @param qoc_dt data.table. District-year quality-of-care indicators.
#' @param output_data_path Character. Directory where the files are written.
#' @param country_code Character. Country code used as the filename prefix.
#' @param data_action Character. Routine data choice used as the filename suffix.
#' @return Named list with the `parquet` and `csv` output file paths.
#'
#' @export
save_quality_of_care_outputs <- function(qoc_dt, output_data_path, country_code, data_action) {
    out_district_parquet <- file.path(output_data_path, glue::glue("{country_code}_quality_of_care_district_year_{data_action}.parquet"))
    out_district_csv <- file.path(output_data_path, glue::glue("{country_code}_quality_of_care_district_year_{data_action}.csv"))

    arrow::write_parquet(qoc_dt, out_district_parquet)
    data.table::fwrite(qoc_dt, out_district_csv)
    log_msg(glue::glue("Saved outputs: {out_district_parquet}, {out_district_csv}"))

    list(parquet = out_district_parquet, csv = out_district_csv)
}


#' Generate and Save Yearly District Maps of Quality of Care Indicators
#'
#' For each indicator present in the table and each year, joins the values to
#' the district shapes and saves a PNG map named `{indicator}_{year}.png`
#' (`allout_{year}.png` for non-malaria outpatients). Rates are binned into
#' fixed classes; absolute values into quantile classes. A map that fails is
#' logged as a `[WARNING]` and skipped.
#'
#' @param qoc_dt data.table. District-year quality-of-care indicators.
#' @param shapes_sf sf. District shapes with an `ADM2_ID` column.
#' @param figures_path Character. Directory where the PNG maps are written.
#' @return Invisibly, TRUE. Called for its side effects.
#'
#' @export
save_quality_of_care_maps <- function(qoc_dt, shapes_sf, figures_path) {
    shapes_sf$ADM2_ID <- as.character(shapes_sf$ADM2_ID)
    qoc_dt$ADM2_ID <- as.character(qoc_dt$ADM2_ID)

    plot_yearly_map <- function(df, sf_shapes, value_col, title_prefix, filename_prefix, is_rate = TRUE) {
        if (!(value_col %in% names(df))) return(invisible(NULL))
        sf_shapes_local <- sf_shapes
        years <- sort(unique(df$YEAR))

        for (yr in years) {
            tryCatch(
                {
                    df_y <- df[YEAR == yr]
                    if (nrow(df_y) == 0) next
                    df_y$ADM2_ID <- as.character(df_y$ADM2_ID)
                    map_df <- dplyr::left_join(sf_shapes_local, df_y, by = "ADM2_ID")
                    if (!(value_col %in% names(map_df))) next

                    vals <- map_df[[value_col]]
                    finite_vals <- vals[is.finite(vals) & !is.na(vals)]
                    if (length(finite_vals) == 0) next

                    if (is_rate) {
                        cat_vals <- cut(vals, breaks = c(-Inf, 0, 0.2, 0.4, 0.6, 0.8, 1.0, Inf), labels = c("<0", "0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", ">1.0"), include.lowest = TRUE)
                        fill_palette <- "YlOrRd"
                    } else {
                        if (length(finite_vals) > 4) {
                            br <- unique(as.numeric(quantile(finite_vals, probs = seq(0, 1, 0.2), na.rm = TRUE)))
                            if (length(br) < 2) {
                                cat_vals <- as.factor(rep("all", nrow(map_df)))
                            } else {
                                cat_vals <- cut(vals, breaks = br, include.lowest = TRUE)
                            }
                        } else {
                            cat_vals <- as.factor(vals)
                        }
                        fill_palette <- "Blues"
                    }

                    map_df <- dplyr::mutate(map_df, cat = as.factor(cat_vals))
                    p <- ggplot2::ggplot(map_df) +
                        ggplot2::geom_sf(ggplot2::aes(fill = cat), color = "grey60", size = 0.1) +
                        ggplot2::scale_fill_brewer(palette = fill_palette, na.value = "white", drop = FALSE) +
                        ggplot2::theme_void() +
                        ggplot2::labs(title = paste0(title_prefix, " - ", yr), fill = value_col, caption = "Source: SNT DHIS2 outliers-imputed routine data") +
                        ggplot2::theme(legend.position = "bottom", plot.title = ggplot2::element_text(face = "bold", size = 12))

                    out_png <- file.path(figures_path, glue::glue("{filename_prefix}_{yr}.png"))
                    ggplot2::ggsave(out_png, plot = p, width = 9, height = 7, dpi = 300, bg = "white")
                    log_msg(glue::glue("Saved map: {out_png}"))
                },
                error = function(e) {
                    log_msg(glue::glue("[WARNING] Failed to build/save map for `{value_col}` year `{yr}`: {conditionMessage(e)}"), level = "warning")
                }
            )
        }
    }

    plot_yearly_map(qoc_dt, shapes_sf, "TESTING_RATE","Testing rate (TEST / SUSP)","testing_rate",TRUE)
    plot_yearly_map(qoc_dt, shapes_sf, "TREATMENT_RATE","Treatment rate (MALTREAT / CONF)","treatment_rate",TRUE)
    plot_yearly_map(qoc_dt, shapes_sf, "CASE_FATALITY_RATE","In-hospital case fatality rate (MALDTH / MALADM)","case_fatality_rate",TRUE)
    plot_yearly_map(qoc_dt, shapes_sf, "PROP_ADM_MALARIA","Proportion admitted for malaria (MALADM / ALLADM)","prop_adm_malaria",TRUE)
    plot_yearly_map(qoc_dt, shapes_sf, "PROP_MALARIA_DEATHS","Proportion of malaria deaths (MALDTH / ALLDTH)","prop_malaria_deaths",TRUE)
    plot_yearly_map(qoc_dt, shapes_sf, "NON_MALARIA_ALL_CAUSE_OUTPATIENTS","Non-malaria all-cause outpatients (ALLOUT)","allout",FALSE)
    plot_yearly_map(qoc_dt, shapes_sf, "PRESUMED_CASES","Presumed cases (PRES)","presumed_cases",FALSE)

    log_msg(glue::glue("Saved yearly maps in: {figures_path}"))
    invisible(TRUE)
}

