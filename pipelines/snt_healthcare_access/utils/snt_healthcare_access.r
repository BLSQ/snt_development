#' Bootstrap runtime context for healthcare-access pipeline.
#'
#' Loads shared utilities/packages, initializes OpenHEXA SDK, configures terra
#' memory options, and creates output folders for data and report artifacts.
#'
#' @param root_path Root workspace path.
#' @param required_packages Character vector of required R packages.
#' @param load_openhexa Whether to import OpenHEXA SDK.
#' @return Named list with paths and OpenHEXA handle.
bootstrap_healthcare_access_context <- function(
    root_path = "~/workspace",
    required_packages = c(
        "jsonlite", "dplyr", "data.table", "ggplot2", "arrow", "glue",
        "sf", "terra", "httr", "reticulate", "stringr", "RColorBrewer"
    ),
    load_openhexa = TRUE
) {
    code_path <- file.path(root_path, "code")
    config_path <- file.path(root_path, "configuration")
    data_path <- file.path(root_path, "data")
    output_data_path <- file.path(data_path, "healthcare_access")
    outputs_path <- file.path(root_path, "pipelines", "snt_healthcare_access", "reporting", "outputs")
    output_plots_path <- file.path(outputs_path, "figures")
    intermediate_results_path <- file.path(output_data_path, "intermediate_results")
    dir.create(output_data_path, recursive = TRUE, showWarnings = FALSE)
    dir.create(outputs_path, recursive = TRUE, showWarnings = FALSE)
    dir.create(output_plots_path, recursive = TRUE, showWarnings = FALSE)
    dir.create(intermediate_results_path, recursive = TRUE, showWarnings = FALSE)

    source(file.path(code_path, "snt_utils.r"))
    install_and_load(required_packages)
    terra::terraOptions(memfrac = 0.5)

    Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")
    openhexa <- NULL
    if (load_openhexa) {
        openhexa <- reticulate::import("openhexa.sdk")
    }
    assign("openhexa", openhexa, envir = .GlobalEnv)

    list(
        ROOT_PATH = root_path,
        CODE_PATH = code_path,
        CONFIG_PATH = config_path,
        DATA_PATH = data_path,
        OUTPUT_DATA_PATH = output_data_path,
        OUTPUTS_PATH = outputs_path,
        OUTPUT_PLOTS_PATH = output_plots_path,
        INTERMEDIATE_RESULTS_PATH = intermediate_results_path,
        openhexa = openhexa
    )
}

#' Load spatial units data from custom file or default DHIS2 dataset.
#'
#' If a custom shapes file is provided, it is validated and loaded from disk.
#' Otherwise, default DHIS2 shapes are downloaded from the configured dataset.
#'
#' @param shapes_file Optional custom shapes file path.
#' @param dhis2_dataset Dataset identifier containing default shapes.
#' @param country_code Country code used in default shapes filename.
#' @return `sf` object with spatial units.
load_spatial_units_data <- function(shapes_file, dhis2_dataset, country_code) {
    if (!is.null(shapes_file) && !is.na(shapes_file) && trimws(shapes_file) != "") {
        custom_shapes_path <- path.expand(shapes_file)
        if (!file.exists(custom_shapes_path)) {
            stop(glue::glue("[ERROR] Custom shapes file was provided but does not exist: {custom_shapes_path}"))
        }

        spatial_units_data <- tryCatch(
            {
                sf::st_read(custom_shapes_path, quiet = TRUE)
            },
            error = function(e) {
                stop(glue::glue(
                    "[ERROR] Error while loading custom shapes file: {custom_shapes_path} [ERROR DETAILS] {conditionMessage(e)}"
                ))
            }
        )

        log_msg(glue::glue("Custom shapes file loaded successfully: {custom_shapes_path}"))
        log_msg(
            "[WARNING] Using a custom shapefile: hierarchy may not align with the extracted DHIS2 pyramid. During data assembly, this mismatch can result in missing values for some organizational units (especially at ADM2 level) if IDs do not match or do not exist in both files.",
            level = "warning"
        )
        return(spatial_units_data)
    }

    spatial_units_data <- tryCatch(
        {
            get_latest_dataset_file_in_memory(dhis2_dataset, paste0(country_code, "_shapes.geojson"))
        },
        error = function(e) {
            stop(glue::glue(
                "[ERROR] Error while loading DHIS2 Shapes data for: {paste0(country_code, '_shapes.geojson')} [ERROR DETAILS] {conditionMessage(e)}"
            ))
        }
    )
    log_msg(glue::glue("Default HMIS/NMDR shapes file downloaded successfully from dataset: {dhis2_dataset}"))
    spatial_units_data
}


#' Prepare spatial/admin objects used in healthcare-access computation.
#'
#' Reprojects shapes, removes invalid geometries, derives non-spatial admin
#' table, and computes country union geometry for clipping/intersection steps.
#'
#' @param spatial_units_data Input spatial units (`sf`).
#' @param country_epsg_degrees Target geographic EPSG code.
#' @return List with cleaned `spatial_units_data`, `admin_data`, and `all_country`.
prepare_spatial_admin_objects <- function(spatial_units_data, country_epsg_degrees) {
    spatial_units_data <- reproject_epsg(spatial_units_data, country_epsg_degrees)

    n_before <- nrow(spatial_units_data)
    spatial_units_data <- spatial_units_data %>%
        dplyr::filter(!is.na(sf::st_is_valid(.)), sf::st_is_valid(.), !sf::st_is_empty(.))
    if (nrow(spatial_units_data) < n_before) {
        log_msg(glue::glue("Dropped {n_before - nrow(spatial_units_data)} spatial unit(s) with null/empty/invalid geometry."))
    }

    admin_data <- data.table::setDT(sf::st_drop_geometry(spatial_units_data))
    all_country <- sf::st_union(spatial_units_data)

    list(
        spatial_units_data = spatial_units_data,
        admin_data = admin_data,
        all_country = all_country
    )
}


#' Compute total and covered population by administrative unit.
#'
#' Aggregates raster-based total and covered populations over admin polygons,
#' computes percent coverage, and joins results back to admin attributes.
#'
#' @param pop_total_raster Raster of total population.
#' @param pop_covered_raster Raster of covered population.
#' @param adm_raster Rasterized admin-id grid used for zonal aggregation.
#' @param admin_col Admin ID column name used for joins.
#' @param admin_data Admin attributes table.
#' @return Data table with `POP_TOTAL`, `POP_COVERED`, and `PCT_HEALTH_ACCESS`.
compute_population_by_admin <- function(pop_total_raster, pop_covered_raster, adm_raster, admin_col, admin_data) {
    pop_total_by_adm2 <- terra::zonal(
        pop_total_raster,
        adm_raster,
        fun = "sum",
        na.rm = TRUE
    )
    log_msg("Aggregated the total population by spatial units.")

    pop_cov_by_adm2 <- terra::zonal(
        pop_covered_raster,
        adm_raster,
        fun = "sum",
        na.rm = TRUE
    )
    log_msg("Aggregated the covered population by spatial units.")

    adm2_pop_total <- data.table::setDT(as.data.frame(pop_total_by_adm2))
    adm2_pop_covered <- data.table::setDT(as.data.frame(pop_cov_by_adm2))
    output_df <- data.table::merge.data.table(adm2_pop_total, adm2_pop_covered, by = admin_col, all = TRUE)

    if (nrow(output_df) != nrow(adm2_pop_total)) {
        stop("Error: There was an error when computing covered population.")
    }

    output_df$PCT_HEALTH_ACCESS <- output_df$POP_COVERED * 100 / output_df$POP_TOTAL
    data.table::merge.data.table(admin_data, output_df, by = admin_col, all.x = TRUE)
}
