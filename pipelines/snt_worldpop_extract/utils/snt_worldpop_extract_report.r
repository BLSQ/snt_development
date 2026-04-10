#' Find country-specific parquet file name in a dataset version.
#'
#' Iterates over files in the latest dataset version, detects parquet files,
#' and rebuilds expected country-prefixed filename convention.
#'
#' @param dataset_last_version OpenHEXA dataset version object.
#' @param country_code Country code prefix.
#' @return Character filename (or NULL when not found).
find_country_parquet_file <- function(dataset_last_version, country_code) {
    parquet_file <- NULL
    files_list <- reticulate::iterate(dataset_last_version$files)
    for (file in files_list) {
        if (endsWith(file$filename, ".parquet")) {
            parquet_file <- paste0(country_code, "_", substring(file$filename, 5))
            print(paste0("Parquet file found: ", parquet_file))
        }
    }
    parquet_file
}

#' Load a report input file from dataset with logging/error handling.
#'
#' @param dataset_name Dataset identifier/name.
#' @param filename File name to download from latest dataset version.
#' @param label Human-readable label for logs/errors.
#' @return Loaded dataset object (data frame / sf depending on source file).
load_worldpop_report_input <- function(dataset_name, filename, label = "dataset file") {
    data <- tryCatch(
        {
            get_latest_dataset_file_in_memory(dataset_name, filename)
        },
        error = function(e) {
            msg <- paste("Error while loading", label, filename, conditionMessage(e))
            cat(msg)
            stop(msg)
        }
    )
    log_msg(paste0(label, " loaded from dataset: ", dataset_name, " dataframe dimensions: ", paste(dim(data), collapse = ", ")))
    data
}


#' Compute comparable reference years for DHIS2 and WorldPop.
#'
#' Uses the earliest available WorldPop year and aligns DHIS2 year to nearest
#' available boundary when exact year is missing.
#'
#' @param worldpop_population WorldPop population table.
#' @param dhis2_population DHIS2 population table.
#' @return List with `worldpop_year` and `dhis2_year`.
get_comparison_years <- function(worldpop_population, dhis2_population) {
    worldpop_year <- min(worldpop_population$YEAR)
    if (worldpop_year %in% unique(dhis2_population$YEAR)) {
        dhis2_year <- worldpop_year
    } else if (worldpop_year < min(dhis2_population$YEAR)) {
        dhis2_year <- min(dhis2_population$YEAR)
    } else {
        dhis2_year <- max(dhis2_population$YEAR)
    }
    list(worldpop_year = worldpop_year, dhis2_year = dhis2_year)
}


#' Build ADM2 comparison table between DHIS2 and WorldPop.
#'
#' Joins ADM2 geometries with selected-year population values from both sources
#' and returns a spatial comparison dataset.
#'
#' @param shapes_data ADM2 shapes (`sf` table).
#' @param dhis2_population DHIS2 population table.
#' @param worldpop_population WorldPop population table.
#' @param dhis2_year Selected DHIS2 comparison year.
#' @param worldpop_year Selected WorldPop comparison year.
#' @return `sf` comparison table with `dhis2_value` and `worldpop_value`.
build_adm2_comparison <- function(shapes_data, dhis2_population, worldpop_population, dhis2_year, worldpop_year) {
    dhis2_pop_renamed <- dhis2_population %>%
        dplyr::filter(YEAR == dhis2_year) %>%
        dplyr::select(ADM2_ID, dhis2_value = POPULATION)

    worldpop_pop_renamed <- worldpop_population %>%
        dplyr::filter(YEAR == worldpop_year) %>%
        dplyr::select(ADM2_ID, worldpop_value = POPULATION)

    comparison_df <- dplyr::left_join(shapes_data, dhis2_pop_renamed[, c("ADM2_ID", "dhis2_value")], by = "ADM2_ID")
    dplyr::left_join(comparison_df, worldpop_pop_renamed[, c("ADM2_ID", "worldpop_value")], by = "ADM2_ID")
}


#' Build ADM1 comparison table between DHIS2 and WorldPop.
#'
#' Aggregates ADM2 geometries to ADM1 and sums population values by ADM1 for
#' both sources, then joins geometry and values for mapping/analysis.
#'
#' @param shapes_data ADM2 shapes (`sf` table).
#' @param dhis2_population DHIS2 population table.
#' @param worldpop_population WorldPop population table.
#' @return `sf` comparison table at ADM1 level.
build_adm1_comparison <- function(shapes_data, dhis2_population, worldpop_population) {
    dhis2_shapes_provinces <- shapes_data %>%
        dplyr::group_by(ADM1_ID) %>%
        dplyr::summarise(geometry = sf::st_union(geometry), .groups = "drop")

    dhis2_pop_prov <- dhis2_population %>%
        dplyr::group_by(ADM1_NAME, ADM1_ID) %>%
        dplyr::summarise(dhis2_value = sum(POPULATION, na.rm = TRUE), .groups = "drop")

    worldpop_pop_prov <- worldpop_population %>%
        dplyr::group_by(ADM1_NAME, ADM1_ID) %>%
        dplyr::summarise(worldpop_value = sum(POPULATION, na.rm = TRUE), .groups = "drop")

    comparison_df_prov <- dplyr::left_join(dhis2_shapes_provinces, dhis2_pop_prov, by = c("ADM1_ID"))
    dplyr::left_join(comparison_df_prov, worldpop_pop_prov, by = c("ADM1_ID"))
}
