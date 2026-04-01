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
