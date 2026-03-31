compute_and_export_indicator_table <- function(
    design_obj,
    indicator_name,
    output_indicator_name = indicator_name,
    admin_name_col,
    admin_data,
    output_data_path,
    filename_without_extension
) {
    table_content <- survey::svyby(
        formula = as.formula(paste("~", indicator_name)),
        by = reformulate(admin_name_col),
        FUN = survey::svymean,
        design = design_obj,
        level = 0.95,
        vartype = "ci",
        na.rm = TRUE,
        influence = TRUE
    )

    data.table::setDT(table_content)
    lower_bound_col <- glue::glue("{toupper(output_indicator_name)}_CI_LOWER_BOUND")
    upper_bound_col <- glue::glue("{toupper(output_indicator_name)}_CI_UPPER_BOUND")
    sample_avg_col <- glue::glue("{toupper(output_indicator_name)}_SAMPLE_AVERAGE")

    names(table_content)[names(table_content) == "ci_l"] <- lower_bound_col
    names(table_content)[names(table_content) == "ci_u"] <- upper_bound_col
    names(table_content)[names(table_content) == indicator_name] <- sample_avg_col

    table_content[get(lower_bound_col) < 0, (lower_bound_col) := 0]
    table_content[get(upper_bound_col) > 1, (upper_bound_col) := 1]
    table_content[, (lower_bound_col) := get(lower_bound_col) * 100]
    table_content[, (upper_bound_col) := get(upper_bound_col) * 100]
    table_content[, (sample_avg_col) := get(sample_avg_col) * 100]

    table_content <- data.table::merge.data.table(admin_data, table_content, by = admin_name_col, all.x = TRUE)
    utils::write.csv(table_content, file = file.path(output_data_path, paste0(filename_without_extension, ".csv")), row.names = FALSE)
    arrow::write_parquet(table_content, file.path(output_data_path, paste0(filename_without_extension, ".parquet")))
    table_content
}


compute_dtp_indicator_tables <- function(
    dtp_design,
    vaccination_doses,
    indicator_access,
    admin_name_col,
    admin_cols,
    admin_data,
    output_data_path,
    country_code,
    data_source,
    admin_level
) {
    dtp_dropout <- data.table::copy(admin_data)
    dose_tables <- list()

    for (dose_number in vaccination_doses) {
        vaccine_colname <- glue::glue("DTP{dose_number}")
        table_name <- glue::glue("{toupper(indicator_access)}{dose_number}")
        filename_without_extension <- glue::glue("{country_code}_{data_source}_{admin_level}_{table_name}")

        df <- compute_and_export_indicator_table(
            design_obj = dtp_design,
            indicator_name = vaccine_colname,
            output_indicator_name = table_name,
            admin_name_col = admin_name_col,
            admin_data = admin_data,
            output_data_path = output_data_path,
            filename_without_extension = filename_without_extension
        )

        dose_tables[[table_name]] <- df
        dtp_dropout <- data.table::merge.data.table(dtp_dropout, df, by = admin_cols)
    }

    list(dtp_dropout = dtp_dropout, dose_tables = dose_tables)
}


compute_and_export_dtp_dropout <- function(
    dtp_dropout,
    vaccination_doses,
    indicator_access,
    indicator_attrition,
    output_data_path,
    country_code,
    data_source,
    admin_level
) {
    dtp_dropout[, grep("BOUND", names(dtp_dropout), value = TRUE) := NULL]

    for (current_dose in vaccination_doses) {
        for (reference_dose in 1:(current_dose - 1)) {
            if ((reference_dose >= 1) & (reference_dose < current_dose)) {
                attrition_col <- glue::glue("{toupper(indicator_attrition)}_{reference_dose}_{current_dose}")
                numerator_colname <- glue::glue("{toupper(indicator_access)}{current_dose}_SAMPLE_AVERAGE")
                denominator_colname <- glue::glue("{toupper(indicator_access)}{reference_dose}_SAMPLE_AVERAGE")
                dtp_dropout[, (attrition_col) := (1 - get(numerator_colname) / get(denominator_colname)) * 100]
            }
        }
    }

    dtp_dropout[, grep("SAMPLE_AVERAGE", names(dtp_dropout), value = TRUE) := NULL]
    filename <- glue::glue("{country_code}_{data_source}_{admin_level}_{indicator_attrition}")
    data.table::fwrite(dtp_dropout, file = file.path(output_data_path, paste0(filename, ".csv")))
    arrow::write_parquet(dtp_dropout, file.path(output_data_path, paste0(filename, ".parquet")))
    dtp_dropout
}


export_careseeking_reporting_plots <- function(
    plot_data,
    all_indicators,
    output_plots_path,
    country_code,
    data_source,
    admin_level
) {
    for (indicator_name in all_indicators) {
        plot_label <- gsub("PCT ", "", gsub("_", " ", indicator_name))
        indicator_plot <- make_dhs_map(
            plot_dt = plot_data,
            plot_colname = indicator_name,
            title_name = glue::glue("Percentage children: {plot_label}"),
            legend_title = "%",
            scale_limits = c(0, 100)
        )
        ggplot2::ggsave(
            indicator_plot,
            file = file.path(output_plots_path, glue::glue("{country_code}_{data_source}_{admin_level}_{toupper(indicator_name)}_plot.png")),
            dpi = 500
        )
    }
}


export_careseeking_reporting_ci_plots <- function(
    all_indicators,
    output_data_path,
    output_plots_path,
    country_code,
    data_source,
    admin_level,
    admin_name_col
) {
    for (indicator_name in all_indicators) {
        indicator_label <- gsub("_", " ", indicator_name)
        ci_data <- data.table::fread(
            file.path(output_data_path, glue::glue("{country_code}_{data_source}_{admin_level}_{indicator_name}.csv"))
        )

        sample_avg_col <- glue::glue("{indicator_name}_SAMPLE_AVERAGE")
        lower_bound_col <- glue::glue("{indicator_name}_CI_LOWER_BOUND")
        upper_bound_col <- glue::glue("{indicator_name}_CI_UPPER_BOUND")
        ci_plot <- make_ci_plot(
            df_to_plot = ci_data,
            admin_colname = admin_name_col,
            point_estimation_colname = sample_avg_col,
            ci_lower_colname = lower_bound_col,
            ci_upper_colname = upper_bound_col,
            title_name = glue::glue("{country_code} {data_source} {indicator_label} CI"),
            x_title = admin_name_col,
            y_title = glue::glue("{indicator_label} (%)")
        )
        ggplot2::ggsave(
            plot = ci_plot,
            filename = file.path(output_plots_path, glue::glue("{country_code}_{data_source}_{admin_level}_{toupper(indicator_name)}_CI_plot.png")),
            width = 8,
            height = 6,
            dpi = 300
        )
    }
}
