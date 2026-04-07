#' Bootstrap runtime context for DHS indicators pipelines.
#'
#' Loads shared utilities/packages, initializes OpenHEXA SDK, parses
#' configuration, and returns common paths/metadata used across DHS notebooks.
#'
#' @param root_path Root workspace path.
#' @param required_packages Character vector of required R packages.
#' @param load_openhexa Whether to import OpenHEXA SDK.
#' @return Named list with paths, config, country code and SDK handle.
bootstrap_dhs_indicators_context <- function(
    root_path = "~/workspace",
    required_packages = c(
        "haven", "sf", "glue", "survey", "data.table", "stringi",
        "jsonlite", "httr", "reticulate", "arrow"
    ),
    load_openhexa = TRUE
) {
    code_path <- file.path(root_path, "code")
    config_path <- file.path(root_path, "configuration")
    data_path <- file.path(root_path, "data")
    dhs_data_path <- file.path(data_path, "dhs", "raw")

    source(file.path(code_path, "snt_utils.r"))
    install_and_load(required_packages)

    Sys.setenv(PROJ_LIB = "/opt/conda/share/proj")
    Sys.setenv(GDAL_DATA = "/opt/conda/share/gdal")
    Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")

    openhexa <- NULL
    if (load_openhexa) {
        openhexa <- reticulate::import("openhexa.sdk")
    }
    assign("openhexa", openhexa, envir = .GlobalEnv)

    config_file_name <- "SNT_config.json"
    config_json <- tryCatch(
        {
            jsonlite::fromJSON(file.path(config_path, config_file_name))
        },
        error = function(e) {
            msg <- paste0("Error while loading configuration", conditionMessage(e))
            cat(msg)
            stop(msg)
        }
    )

    log_msg(paste0("SNT configuration loaded from  : ", file.path(config_path, config_file_name)))

    list(
        ROOT_PATH = root_path,
        CODE_PATH = code_path,
        CONFIG_PATH = config_path,
        DATA_PATH = data_path,
        DHS_DATA_PATH = dhs_data_path,
        config_json = config_json,
        COUNTRY_CODE = config_json$SNT_CONFIG$COUNTRY_CODE,
        openhexa = openhexa
    )
}

#' Load DHIS2 shapes used as spatial reference for DHS outputs.
#'
#' Downloads country-specific shapes from the configured DHIS2 formatted
#' dataset with explicit logging and stop-on-error behavior.
#'
#' @param dhis2_dataset Dataset identifier containing shapes.
#' @param country_code Country code used in filename prefix.
#' @return Spatial data object loaded from `*_shapes.geojson`.
load_dhs_spatial_data <- function(dhis2_dataset, country_code) {
    spatial_data_filename <- paste(country_code, "shapes.geojson", sep = "_")
    spatial_data <- tryCatch(
        {
            get_latest_dataset_file_in_memory(dhis2_dataset, spatial_data_filename)
        },
        error = function(e) {
            msg <- glue::glue("[ERROR] Error while loading DHIS2 shapes data for {country_code}: {conditionMessage(e)}")
            log_msg(msg, "error")
            stop(msg)
        }
    )
    log_msg(glue::glue("File {spatial_data_filename} successfully loaded from dataset version: {dhis2_dataset}"))
    spatial_data
}

#' Compute and export a DHS indicator table with confidence intervals.
#'
#' Runs survey-weighted estimation by admin unit, formats confidence interval
#' and sample-average columns, converts proportions to percentages, merges with
#' admin reference table, and exports CSV/Parquet outputs.
#'
#' @param design_obj Survey design object.
#' @param indicator_name Indicator variable name in design data.
#' @param output_indicator_name Indicator label used in output columns/files.
#' @param admin_name_col Admin grouping column name.
#' @param admin_data Admin reference table for full joins.
#' @param output_data_path Directory to write output files.
#' @param filename_without_extension Output file stem.
#' @return Formatted indicator table.
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


#' Compute DTP dose indicators and prepare dropout base table.
#'
#' Iterates over configured DTP doses, computes/export per-dose indicators, and
#' aggregates them into a single table used later to derive dropout metrics.
#'
#' @param dtp_design Survey design object for vaccination analysis.
#' @param vaccination_doses Integer vector of dose numbers (e.g., 1:3).
#' @param indicator_access Prefix used for access indicator naming.
#' @param admin_name_col Admin grouping column name.
#' @param admin_cols Admin key columns used for table merges.
#' @param admin_data Admin reference table.
#' @param output_data_path Directory to write output files.
#' @param country_code Country code used in filenames.
#' @param data_source DHS source code used in filenames.
#' @param admin_level Admin level label used in filenames.
#' @return List with `dtp_dropout` table and `dose_tables`.
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


#' Compute and export DTP dropout indicators.
#'
#' Derives dropout percentages between dose pairs from dose-level sample
#' averages, removes intermediate CI/sample columns, and exports final table.
#'
#' @param dtp_dropout Base table containing per-dose sample averages.
#' @param vaccination_doses Integer vector of dose numbers.
#' @param indicator_access Prefix used for dose access columns.
#' @param indicator_attrition Prefix used for dropout columns.
#' @param output_data_path Directory to write output files.
#' @param country_code Country code used in filenames.
#' @param data_source DHS source code used in filenames.
#' @param admin_level Admin level label used in filenames.
#' @return Dropout table with derived attrition columns.
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


#' Export careseeking choropleth plots for each indicator.
#'
#' Generates and saves map plots for all requested indicators using shared DHS
#' map styling helper.
#'
#' @param plot_data Spatial table with indicator values.
#' @param all_indicators Character vector of indicator names.
#' @param output_plots_path Directory to write plot images.
#' @param country_code Country code used in filenames.
#' @param data_source DHS source code used in filenames.
#' @param admin_level Admin level label used in filenames.
#' @return Invisibly saves plot files for each indicator.
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


#' Export careseeking confidence-interval bar plots.
#'
#' Reads per-indicator summary tables, builds CI plots, and exports one image
#' per indicator for reporting.
#'
#' @param all_indicators Character vector of indicator names.
#' @param output_data_path Directory containing per-indicator CSV files.
#' @param output_plots_path Directory to write plot images.
#' @param country_code Country code used in filenames.
#' @param data_source DHS source code used in filenames.
#' @param admin_level Admin level label used in filenames.
#' @param admin_name_col Admin label column used on plot axis.
#' @return Invisibly saves CI plot files for each indicator.
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
