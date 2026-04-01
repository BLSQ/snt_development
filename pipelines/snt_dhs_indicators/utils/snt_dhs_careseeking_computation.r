compute_careseeking_indicators <- function(
    kr_design_sampling,
    indicator_names,
    admin_name_col,
    admin_data,
    output_data_path,
    country_code,
    data_source,
    admin_level
) {
    summary_table <- data.table::copy(admin_data)
    indicator_tables <- list()

    for (indicator_name in indicator_names) {
        table_content <- survey::svyby(
            formula = as.formula(paste("~", indicator_name)),
            by = reformulate(admin_name_col),
            FUN = survey::svymean,
            design = kr_design_sampling,
            level = 0.95,
            vartype = "ci",
            na.rm = TRUE,
            influence = TRUE
        )
        data.table::setDT(table_content)

        lower_bound_col <- glue::glue("{toupper(indicator_name)}_CI_LOWER_BOUND")
        upper_bound_col <- glue::glue("{toupper(indicator_name)}_CI_UPPER_BOUND")
        sample_avg_col <- glue::glue("{toupper(indicator_name)}_SAMPLE_AVERAGE")
        names(table_content)[names(table_content) == "ci_l"] <- lower_bound_col
        names(table_content)[names(table_content) == "ci_u"] <- upper_bound_col
        names(table_content)[names(table_content) == indicator_name] <- sample_avg_col

        table_content[get(lower_bound_col) < 0, (lower_bound_col) := 0]
        table_content[get(upper_bound_col) > 1, (upper_bound_col) := 1]
        table_content[, (lower_bound_col) := get(lower_bound_col) * 100]
        table_content[, (upper_bound_col) := get(upper_bound_col) * 100]
        table_content[, (sample_avg_col) := get(sample_avg_col) * 100]

        indicator_estimation_table <- table_content[
            ,
            .SD,
            .SDcols = c(admin_name_col, grep("SAMPLE_AVERAGE", names(table_content), value = TRUE))
        ]

        table_content <- data.table::merge.data.table(admin_data, table_content, by = admin_name_col)
        summary_table <- data.table::merge.data.table(summary_table, indicator_estimation_table, by = admin_name_col)

        filename_without_extension <- glue::glue("{country_code}_{data_source}_{admin_level}_{toupper(indicator_name)}")
        utils::write.csv(table_content, file = file.path(output_data_path, paste0(filename_without_extension, ".csv")), row.names = FALSE)
        arrow::write_parquet(table_content, file.path(output_data_path, paste0(filename_without_extension, ".parquet")))
        indicator_tables[[indicator_name]] <- table_content
    }

    names(summary_table) <- gsub("_SAMPLE_AVERAGE", "", names(summary_table))
    summary_filename_without_extension <- glue::glue("{country_code}_{data_source}_{admin_level}_PCT_CARESEEKING_SAMPLE_AVERAGE")
    utils::write.csv(summary_table, file = file.path(output_data_path, paste0(summary_filename_without_extension, ".csv")), row.names = FALSE)
    arrow::write_parquet(summary_table, file.path(output_data_path, paste0(summary_filename_without_extension, ".parquet")))

    list(summary_table = summary_table, indicator_tables = indicator_tables)
}
