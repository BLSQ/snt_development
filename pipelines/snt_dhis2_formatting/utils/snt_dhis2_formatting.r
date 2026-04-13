# Shared helpers for snt_dhis2_formatting code notebooks.

# Load base utils
source(file.path("~/workspace/code", "snt_utils.r"))   


#' Get Setup Variables for SNT Workspace
#' Initializes workspace paths, loads R packages, and imports OpenHEXA SDK.
#'
#' @param SNT_ROOT_PATH Character. Root path of the SNT workspace. Default: '~/workspace'
#' @param packages Character vector. R packages to install and load.
#' @return List with SNT paths.
#'
#' @export
get_setup_variables <- function(
    SNT_ROOT_PATH='~/workspace', 
    packages=c("arrow", "dplyr", "tidyr", "stringr", "stringi", "jsonlite", "httr", "glue")
) {
        
    # List required pcks
    required_packages <- unique(c(packages, "reticulate"))
    install_and_load(required_packages)

    # Set environment to load openhexa.sdk from the right environment
    Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")
    
    # Attempt to import the SDK
    tryCatch({
        sdk <- reticulate::import("openhexa.sdk")
        assign("openhexa", sdk, envir = .GlobalEnv)
    }, error = function(e) {
        log_msg("Could not import openhexa.sdk. Ensure it is installed in /opt/conda/bin/python", "warning")
    })    

    # Set paths (add paths here)
    paths_to_check = list(
        CONFIG_PATH = file.path(SNT_ROOT_PATH, "configuration"),  
        UPLOADS_PATH = file.path(SNT_ROOT_PATH, "uploads"),
        DATA_PATH = file.path(SNT_ROOT_PATH, "data")
    )

    # create if they do not exist
    lapply(paths_to_check, dir.create, recursive = TRUE, showWarnings = FALSE)
    
    return(paths_to_check)
}


#' Load SNT Configuration File
#' Reads and parses a JSON configuration file.
#' @param snt_config_path Character. Path to the configuration JSON file.
#' @return List containing parsed configuration.
#'
#' @export
load_snt_config <- function(snt_config_path) {

    # config file path 
    config_json <- tryCatch({ fromJSON(snt_config_path) },
                error = function(e) {
                    stop(glue::glue("[ERROR] Error while loading configuration: {snt_config_path}"))
                })
    
    log_msg(paste0("SNT configuration loaded from  : ", snt_config_path))
    return(config_json)    
}


#' Load Dataset File from OpenHEXA
#' Retrieves the latest version of a file from an OpenHEXA dataset.
#'
#' @param dataset_id Character. OpenHEXA dataset identifier.
#' @param filename Character. Name of file to load.
#' @param verbose Bool. Log messages
#' @return Dataframe containing the loaded data.
#'
#' @export
load_dataset_file <- function (dataset_id, filename, verbose=TRUE) {
    data <- tryCatch({ 
            get_latest_dataset_file_in_memory(dataset_id, filename) 
        }, error = function(e) {
            stop(glue::glue("[ERROR] Error while loading {filename} file from dataset: {dataset_id}"))
    })

    if (verbose) {        
        log_msg(glue::glue("{filename} data loaded from dataset : {dataset_id} dataframe dimensions: [{paste(dim(data), collapse=', ')}]"))
    }    
    return(data)
}


#' Load a CSV File with Error Handling
#'
#' @description 
#' Attempts to read a CSV file from the specified path. If the file cannot be loaded, 
#' it logs a high-level error message and stops execution.
#'
#' @param csv_file_path String representing the file path to the CSV.
#' @return A dataframe containing the contents of the CSV file.
#' 
#' @export
load_csv_file <- function(csv_file_path) {
    csv_data <- tryCatch({ read.csv(csv_file_path) },
        error = function(e) {
            stop(glue::glue("[ERROR] Error while loading the file: {csv_file_path}"))
        }
    )
    log_msg(glue::glue("File loaded: {csv_file_path}"))
    return(csv_data)
}


# -----------------------------------------------------------------------------------------
# Routine util functions ------------------------------------------------------------------
# -----------------------------------------------------------------------------------------


#' Select and Validate Indicator Definitions
#'
#' Processes indicator definitions by standardizing names to uppercase,
#' removing invalid entries (NULL, empty, or whitespace-only), and trimming values.
#'
#' @param config_ind_definitions Named list of indicator definitions from SNT configuration.
#'
#' @return List with two elements:
#'   \item{valid_indicators}{Named list of valid indicators with trimmed values.}
#'   \item{empty_indicators}{Character vector of indicator names with no available data.}
#' @export
indicators_selection <- function(config_ind_definitions) {
    
    # Get list of indicator definitions from SNT configuration
    ind_definitions <- config_ind_definitions
    names(ind_definitions) <- toupper(names(ind_definitions))
    valid_ind_definitions <- ind_definitions
    empty_ind_definitions <- c()  # empty are those indicators with no available data
    
    # Loop over the indicators and clean the list
    for (name in names(valid_ind_definitions)) {
      value <- valid_ind_definitions[[name]]
      
      # If value is NULL or length zero, leave as is or set to NULL
      if (is.null(value) || length(value) == 0 || all(value == "")) {
        valid_ind_definitions[[name]] <- NULL 
        empty_ind_definitions <- c(empty_ind_definitions, name)
        next
      }  
      # Trim whitespace and then check if empty string
      value_trimmed <- trimws(value)
      valid_ind_definitions[[name]] <- value_trimmed  
    }
    
    return(list(
        valid_indicators=valid_ind_definitions,  # This is a named list
        empty_indicators=empty_ind_definitions
    ))
}


#' Build Indicator Columns from Data Elements
#'
#' Creates indicator columns by summing specified data elements from DHIS2 routine data.
#' Handles missing data elements and optionally includes empty indicators as NA columns.
#'
#' @param data Data frame containing DHIS2 data elements.
#' @param valid_indicators Named list where names are indicators and values are 
#'   character vectors of data element UIDs to sum.
#' @param empty_indicators Character vector of indicator names with no definitions.
#' @param include_empty_ind Logical. If TRUE, adds empty indicators as NA columns. Default TRUE.
#'
#' @return Data frame with added indicator columns. Multi-element indicators are 
#'   summed; single-element indicators are copied; empty indicators become NA.
#' @export
build_indicators <- function(data, valid_indicators, empty_indicators, include_empty_ind=TRUE) {
    
    all_cols <- colnames(data)
    empty_data_indicators <- c()
    
    # loop over the definitions
    for (indicator in names(valid_indicators)) {
            
        data_element_uids <- valid_indicators[[indicator]]    
        col_names <- c()
    
        if (length(data_element_uids) > 0) {
                        
            for (dx in data_element_uids) {     
                dx_norm <- gsub("\\.", "_", dx) # dots to underscores to match columns
                        
                # CASE 1: dx_norm is a specific DE_CO pair
                if (dx_norm %in% all_cols) {
                    col_names <- unique(c(col_names, dx_norm))
                } 
                # CASE 2: dx_norm is a plain DE; find all matching DE_CO combinations
                else {
                    # Regex: Starts with UID, followed by an underscore or end of string            
                    pattern <- paste0("^", dx_norm, "(_|$)")
                    matches <- all_cols[grepl(pattern, all_cols)]                    
                    if (length(matches) > 0) {
                        col_names <- unique(c(col_names, matches))
                    } else {
                        log_msg(glue("Indicator '{indicator}': data points '{dx_norm}' not found"), level = "warning")
                    }
                }
            }
        
            # check if there are matching data elements
            if (length(col_names) == 0) {
                log_msg(glue("Indicator '{indicator}': No data elements found to build indicator, indicator skipped."), level="warning")
                empty_data_indicators <- c(empty_data_indicators, indicator)
                next
            }
            
            # logs
            log_msg(glue("Building indicator: {indicator} -> column selection: {paste(col_names, collapse = ', ')}"))
            
            if (length(col_names) > 1) {
                sums <- rowSums(data[, col_names], na.rm = TRUE)
                all_na <- rowSums(!is.na(data[, col_names])) == 0
                sums[all_na] <- NA  # Keep NA if all rows are NA!
                data[[indicator]] <- sums            
            } else {
                data[[indicator]] <- data[[col_names]]
            }
            
        } else {
            data[[indicator]] <- NA
            log_msg(glue("Setting indicator: {indicator} -> column selection : NULL"))
        }
    }
    
    # Add the empty indicator columns (if not needed this can be commented)
    if (include_empty_ind) {
        for (empty in unique(c(empty_indicators, empty_data_indicators))) {
            data[[empty]] <- NA            
            log_msg(glue("Building indicator: {empty} -> column selection : NA"))
        }    
    }
    
    return(data)
}



#' Merge and Format Routine Data with Metadata
#'
#' Combines routine DHIS2 data with organizational unit metadata and formats 
#' the output with standardized column names and temporal variables.
#'
#' @param data Data frame with routine data containing OU, PE, and indicator columns.
#' @param metadata Data frame with organizational unit metadata (OU hierarchies and names).
#' @param indicator_definitions Named list of indicator definitions (names used for column selection).
#'
#' @return Data frame with formatted columns: PERIOD, YEAR, MONTH, OU_ID, OU_NAME, 
#'   ADM1_NAME, ADM1_ID, ADM2_NAME, ADM2_ID, and all built indicators. Sorted by period.
#' @export
merge_and_format_routine_data <- function(data, metadata, indicator_definitions) {
            
    # Filter routine data columns by indicators
    built_indicators <- names(indicator_definitions)
    routine_data_selection <- data[, c("OU", "PE", built_indicators)]
    
    # left join with metadata 
    routine_data_merged <- merge(routine_data_selection, metadata, by = "OU", all.x = TRUE)
    
    # Select administrative columns
    adm_1_id_col <- gsub("_NAME", "_ID", ADMIN_1)
    adm_1_name_col <- ADMIN_1
    adm_2_id_col <- gsub("_NAME", "_ID", ADMIN_2)
    adm_2_name_col <- ADMIN_2
    
    # Select and Rename
    routine_data_formatted <- routine_data_merged %>%
        mutate(        
            YEAR = as.numeric(substr(PE, 1, 4)),
            MONTH = as.numeric(substr(PE, 5, 6)),
            PE = as.numeric(PE)
        ) %>%
        select(
            PERIOD = PE,
            YEAR,
            MONTH,
            OU_ID = OU,
            OU_NAME = !!sym(max_admin_col_name),
            ADM1_NAME = !!sym(adm_1_name_col),
            ADM1_ID = !!sym(adm_1_id_col),
            ADM2_NAME = !!sym(adm_2_name_col),
            ADM2_ID = !!sym(adm_2_id_col),
            all_of(built_indicators)
        )
    
    # Column names to upper case
    colnames(routine_data_formatted) <- clean_column_names(routine_data_formatted)
    
    # Sort dataframe by period
    routine_data_formatted <- routine_data_formatted[order(as.numeric(routine_data_formatted$PERIOD)), ]
}


# -----------------------------------------------------------------------------------------
# Shapes util functions -------------------------------------------------------------------
# -----------------------------------------------------------------------------------------


#' Convert GeoJSON Strings to sf Object
#'
#' Converts a data frame containing GeoJSON strings in a GEOMETRY column to an
#' sf (simple features) object with proper spatial geometry.
#'
#' @param data Data frame containing a GEOMETRY column with GeoJSON strings.
#' @param geom_col Character string. Name of the column containing GeoJSON 
#'   geometry strings (default: "GEOMETRY").
#' @param crs Coordinate reference system. Numeric EPSG code or CRS object 
#'   (default: 4326 for WGS84).
#'
#' @return An sf object with the GEOMETRY column converted to sfc geometry.
#'   Invalid or NA geometries are replaced with empty geometry collections.
#'
#' @export
geojson_to_sf <- function(data, geom_col = "GEOMETRY", crs = 4326) {

    # Make column name lookup case-insensitive
    geom_col_match <- names(data)[tolower(names(data)) == tolower(geom_col)]
    
    if (length(geom_col_match) == 0) {
        stop(paste0("Error: Column '", geom_col, "' not found in data"))
    }
    
    # Use the matched column
    geom_col <- geom_col_match[1]
    
    # Convert GeoJSON strings to sfg objects
    geometry_sfc <- lapply(data[[geom_col]], function(g) {
        if (is.na(g) || is.null(g)) return(sf::st_geometrycollection())
        tryCatch({
          geo <- geojsonsf::geojson_sfc(g)
          geo[[1]]  # extract sfg
        }, error = function(e) {
          sf::st_geometrycollection()  # return empty but valid geometry
        })
    })
    
    # Convert to sfc
    geometry_sfc <- sf::st_sfc(geometry_sfc, crs = crs)
    
    # Create sf object (exclude original geometry column)
    cols_to_keep <- setdiff(names(data), geom_col)
    data_no_geom <- data[, cols_to_keep, drop = FALSE]
    # data_no_geom <- data[, !names(data) %in% geom_col, drop = FALSE]
    shapes_sf <- sf::st_sf(data_no_geom, geometry = geometry_sfc)
    
    return(shapes_sf)
}


#' Simplify geometries in an sf object
#'
#' This function simplifies all geometries in an `sf` object using
#' `rmapshaper::ms_simplify` and ensures the result remains valid.
#'
#' @param sf_object An `sf` object containing geometries.
#' @param keep A numeric value between 0 and 1 indicating the proportion of points
#'   to retain during simplification (default is 0.05).
#'
#' @return An `sf` object with simplified and valid geometries.
#'
#' @details
#' - All geometries (POLYGON, MULTIPOLYGON, etc.) are simplified.
#' - Topology is preserved using `keep_shapes = TRUE`.
#' - Geometry validity is enforced using `sf::st_make_valid()`.
#'
#' @importFrom sf st_make_valid
#' @importFrom rmapshaper ms_simplify
#'
#' @export
simplify_geometries <- function(sf_object, keep = 0.05) {
    # Optional safety check
    if (!inherits(sf_object, "sf")) {
        stop("[ERROR] Input must be an sf object")
    }
    # Save original column order
    original_cols <- names(sf_object)
    
    # Simplify all geometries at once
    simplified_sf <- rmapshaper::ms_simplify(
        sf_object,
        keep = keep,
        keep_shapes = TRUE
    )
    
    # Ensure validity
    simplified_sf <- sf::st_make_valid(simplified_sf)
    simplified_sf <- simplified_sf[, original_cols, drop = FALSE]
    
    return(simplified_sf)
}


# -----------------------------------------------------------------------------------------
# Population util functions ---------------------------------------------------------------
# -----------------------------------------------------------------------------------------


#' Build Administrative Column Names
#'
#' Creates a list containing both the name and ID column names for an
#' administrative level by converting the _NAME suffix to _ID.
#'
#' @param admin_level Character string representing the administrative level
#'   column name (e.g., "PROVINCE_NAME")
#'
#' @return A named list with two elements:
#'   \item{name}{The original administrative level name column}
#'   \item{id}{The corresponding ID column (NAME replaced with ID)}
#' @export
build_admin_columns <- function(admin_level) {
    list(
        name = admin_level,
        id = gsub("_NAME", "_ID", admin_level)
    )
}

#' Extract Administrative Configuration
#'
#' Extracts and structures the administrative level configuration from the
#' SNT configuration object, converting both levels to uppercase and building
#' their corresponding column names.
#'
#' @param snt_configuration List containing SNT configuration with nested
#'   elements SNT_CONFIG$DHIS2_ADMINISTRATION_1 and DHIS2_ADMINISTRATION_2
#'
#' @return A named list with two elements:
#'   \item{level1}{List with name and id columns for administrative level 1}
#'   \item{level2}{List with name and id columns for administrative level 2}
#' @export
get_admin_config <- function(snt_configuration) {
    list(
        level1 = build_admin_columns(toupper(snt_configuration$SNT_CONFIG$DHIS2_ADMINISTRATION_1)),
        level2 = build_admin_columns(toupper(snt_configuration$SNT_CONFIG$DHIS2_ADMINISTRATION_2))
    )
}

#' Get Organizational Units Selection
#'
#' Selects and returns distinct organizational unit data for both administrative
#' levels, including both name and ID columns.
#'
#' @param ou_data Data frame containing organizational unit information with
#'   columns for administrative level names and IDs
#' @param admin_cols Named list with level1 and level2 administrative column
#'   configurations (output from \code{get_admin_config})
#'
#' @return A data frame with distinct rows containing four columns:
#'   administrative level 1 name and ID, administrative level 2 name and ID
#' @export
get_org_units_selection <- function(ou_data, admin_cols) {
    ou_data %>%
        select(
            all_of(c(
                admin_cols$level1$name,
                admin_cols$level1$id,
                admin_cols$level2$name,
                admin_cols$level2$id
            ))
        ) %>%
        distinct()
}


#' Aggregate Individual Population Indicator
#'
#' Filters population data for a specific indicator, aggregates values by
#' organizational unit and period, and formats the result for joining.
#'
#' @param pop_data Data frame containing population data with columns:
#'   DX (data element), PE (period), OU (organizational unit), VALUE
#' @param indicator_def List containing indicator definition with an 'ids'
#'   element specifying which DX values to include
#' @param ind_name Character string name for the indicator (used as column name
#'   in output)
#' @param admin_cols Named list with administrative column configurations
#'
#' @return A data frame with columns:
#'   \item{YEAR}{Numeric year from PE}
#'   \item{<admin_level2_id>}{Organizational unit ID (dynamically named)}
#'   \item{<ind_name>}{Aggregated integer value (dynamically named)}
#' @export
aggregate_indicator <- function(pop_data, indicator_def, ind_name, admin_cols) {
    pop_data %>%
        filter(DX %in% indicator_def$ids) %>%
        group_by(PE, OU) %>%
        summarise(
            VALUE = sum(as.integer(as.numeric(VALUE)), na.rm = TRUE),
            .groups = "drop"
        ) %>%
        transmute(
            YEAR = as.numeric(PE),
            !!admin_cols$level2$id := OU,
            !!ind_name := VALUE
        )
}


#' Build Population Indicators Dataset
#'
#' Creates a comprehensive dataset of population indicators by combining
#' organizational unit structure with population data across multiple years
#' and indicators.
#'
#' @param pop_data Data frame containing raw population data with columns:
#'   PE (period/year), OU (organizational unit), DX (data element),
#'   VALUE (numeric value)
#' @param ou_data Data frame containing organizational unit hierarchy with
#'   administrative level name and ID columns
#' @param snt_configuration List containing SNT configuration including:
#'   \itemize{
#'     \item SNT_CONFIG$DHIS2_ADMINISTRATION_1: First admin level name
#'     \item SNT_CONFIG$DHIS2_ADMINISTRATION_2: Second admin level name
#'     \item DHIS2_DATA_DEFINITIONS$POPULATION_DEFINITIONS$POPULATION_INDICATORS:
#'       Named list of indicator definitions
#'   }
#'
#' @return A data frame with one row per year-organizational unit combination,
#'   containing:
#'   \itemize{
#'     \item YEAR: Integer year
#'     \item Administrative level 1 name and ID columns
#'     \item Administrative level 2 name and ID columns
#'     \item One column per population indicator with aggregated values
#'   }
#'   Sorted by YEAR, admin level 1 name, and admin level 2 name.
#' @export
build_population_indicators <- function(pop_data, ou_data, snt_configuration) {
    
    # Extract configuration
    admin_cols <- get_admin_config(snt_configuration)
    pop_indicators <- snt_configuration$DHIS2_DATA_DEFINITIONS$POPULATION_DEFINITIONS[["POPULATION_INDICATORS"]]
    
    # Build organizational units template
    ou_selection <- get_org_units_selection(ou_data, admin_cols)
    pop_template <- crossing(
        YEAR = unique(as.integer(pop_data$PE)),
        ou_selection
    )
    
    # Process each population indicator
    for (ind_name in names(pop_indicators)) {
        ind_name_upper <- toupper(ind_name)
        log_msg(glue("Building DHIS2 population indicator: {ind_name_upper}."))        
        pop_template <- pop_template %>%
            left_join(
                aggregate_indicator(pop_data, pop_indicators[[ind_name]], ind_name_upper, admin_cols),
                by = c("YEAR", admin_cols$level2$id)
            )
    }
    
    # Sort and return
    pop_template %>% arrange(YEAR, !!sym(admin_cols$level1$name), !!sym(admin_cols$level2$name))
}


#' Format Administrative Level Names
#'
#' Applies string formatting to administrative level name columns in a data frame.
#'
#' @param data Data frame containing administrative level columns
#' @param admin_cols Named list with level1 and level2 administrative column
#'   configurations (output from \code{get_admin_config})
#'
#' @return Data frame with formatted administrative level name columns
#' @export
format_admin_names <- function(data, admin_cols) {
    data %>%
        mutate(
            !!admin_cols$level1$name := format_names(!!sym(admin_cols$level1$name)),
            !!admin_cols$level2$name := format_names(!!sym(admin_cols$level2$name))
        )
}

#' Standardize Population Table Column Names
#'
#' Renames administrative level columns to standard names (ADM1_NAME, ADM1_ID,
#' ADM2_NAME, ADM2_ID) and formats the name columns.
#'
#' @param population_table Data frame containing population data with
#'   administrative level columns
#' @param admin_cols Named list with level1 and level2 administrative column
#'   configurations (output from \code{get_admin_config})
#'
#' @return Data frame with:
#'   \itemize{
#'     \item Standardized column names (ADM1_NAME, ADM1_ID, ADM2_NAME, ADM2_ID)
#'     \item Formatted administrative level name strings
#'     \item All other columns preserved
#'   }
#' @export
standardize_population_columns <- function(population_table, admin_cols) {
    population_table %>%
        format_admin_names(admin_cols) %>%
        rename(
            ADM1_NAME = !!sym(admin_cols$level1$name),
            ADM1_ID = !!sym(admin_cols$level1$id),
            ADM2_NAME = !!sym(admin_cols$level2$name),
            ADM2_ID = !!sym(admin_cols$level2$id)
        )
}


# -----------------------------------------------------------------------------------------
# Pyramid util functions ------------------------------------------------------------------
# -----------------------------------------------------------------------------------------



#' Clean Dataframe Formats and Column Names
#'
#' @description Formats string values in any column containing "_NAME" and standardizes all column names.
#'
#' @param data A dataframe containing the data to be cleaned.
#'
#' @return A dataframe with formatted name values and cleaned column names.
#' @export
clean_input_data <- function(data, verbose=TRUE) {

    name_columns <- colnames(data)[grepl("_NAME", colnames(data))]
    for (column in name_columns){
        if (verbose) log_msg(glue("Formatting strings of : {column}"))    
        data[[column]] <- format_names(data[[column]])  # Clean strings 
    }
      
    # Column names to upper case
    colnames(data) <- clean_column_names(data)
    return(data)
}


#' Extract Coordinates from Geometry
#'
#' @description Extracts longitude and latitude from a specified geometry column, 
#' appends them as new columns, and removes the original geometry column.
#'
#' @param data A dataframe containing the geometry column.
#' @param geom_col A character string specifying the name of the geometry column. Defaults to "GEOMETRY".
#'
#' @return A dataframe with 'LONGITUDE' and 'LATITUDE' columns added, and the geometry column removed.
#' @export
extract_geometry_coordinates <- function(data, geom_col = "GEOMETRY", verbose=TRUE) {
  
    # 1. Use double brackets [[ ]] to dynamically access the column in Base R
    coords_list <- lapply(data[[geom_col]], extract_geo_coords)
    coords_df <- dplyr::bind_rows(lapply(coords_list, as.data.frame))

    # Log failures 
    parse_fail_idx <- which(!coords_df$parse_ok)
    parse_failures <- length(parse_fail_idx)
    if (parse_failures > 0) {
        parse_fail_examples <- paste(head(parse_fail_idx, 10), collapse = ", ")
        if (verbose) {
            log_msg(
                glue("{parse_failures} GEOMETRY records could not be parsed. Resulting LONGITUDE/LATITUDE are NA for those rows."),
                "warning"
            )  
        } 
    }
    
  # 2. Use dplyr::all_of() to dynamically drop a column by its string name
  data <- data %>%
    dplyr::mutate(
      LONGITUDE = coords_df$lon,
      LATITUDE = coords_df$lat
    ) %>%
    dplyr::select(-dplyr::all_of(geom_col))
    
  return(data)
}

#' Extract Coordinates from Geometry JSON
#'
#' @description Parses a JSON string representing geographic geometry and extracts the longitude and latitude. Includes error handling for empty, invalid, or missing coordinates.
#'
#' @param geom_json A character string containing the geometry data in JSON format.
#'
#' @return A named list containing three elements: \code{lon} (numeric), \code{lat} (numeric), and \code{parse_ok} (logical indicating if the parsing was successful).
#' @export
extract_geo_coords <- function(geom_json) {
  if (is.na(geom_json) || !nzchar(geom_json)) {
    return(list(lon = NA_real_, lat = NA_real_, parse_ok = FALSE))
  }

  parsed <- tryCatch(jsonlite::fromJSON(geom_json), error = function(e) NULL)
  coords <- parsed$coordinates

  if (is.null(coords) || length(coords) < 2) {
    return(list(lon = NA_real_, lat = NA_real_, parse_ok = FALSE))
  }

  list(
    lon = suppressWarnings(as.numeric(coords[[1]])),
    lat = suppressWarnings(as.numeric(coords[[2]])),
    parse_ok = TRUE
  )
}


#' Safely Read GeoJSON File
#'
#' @description Reads a GeoJSON file from a specified path with built-in error handling. 
#' Checks if the file exists and catches parsing errors if the file is corrupted.
#'
#' @param file_path A character string specifying the full path to the GeoJSON file.
#'
#' @return A spatial dataframe (sf object) if successful, or NULL if the process fails.
#' @export
read_geojson_safe <- function(file_path) {
  
    # 1. Check if the file exists in the folder
    if (!file.exists(file_path)) {
        # If you have a custom log_msg function from earlier, you can swap 'message' for it!
        log_msg(glue("File does not exist at the specified path: {file_path}"), "error")
        return(NULL)
    }
    
    # 2. Try to read the file and catch corruption/parsing errors
    geo_data <- tryCatch({ sf::read_sf(file_path, quiet = TRUE)}, 
        error = function(e) {
            log_msg(glue("Failed to parse the GeoJSON file. It may be corrupted. R says: {e$message}"), "error")
            return(NULL)
        })
    
    return(geo_data)
}


#' Safely Read GeoJSON File
#'
#' @description Reads a GeoJSON file from a specified path with built-in error handling. 
#' Checks if the file exists and catches parsing errors if the file is corrupted.
#' 
#' @details This function wraps \code{sf::read_sf()} inside a \code{tryCatch} block. 
#' It is particularly useful in automated data pipelines where missing or corrupted 
#' geographic files should be logged but shouldn't crash the entire script.
#'
#' @param file_path A character string specifying the full path to the GeoJSON file.
#'
#' @return A spatial dataframe (\code{sf} object) if successful, or \code{NULL} if the process fails.
#' 
#' @examples
#' \dontrun{
#'   # Example of a successful read
#'   my_geo_data <- read_geojson_safe("data/valid_regions.geojson")
#'   
#'   # Example of handling a missing file gracefully (returns NULL)
#'   missing_data <- read_geojson_safe("data/does_not_exist.geojson")
#' }
#' 
#' @importFrom sf read_sf
#' @importFrom glue glue
#' @export
points_within_country_batch <- function(lon_vec, lat_vec, boundary_sf) {
    out <- rep(FALSE, length(lon_vec))
    ok_idx <- which(
        !is.na(lon_vec) &
          !is.na(lat_vec) &
          (abs(lon_vec) <= 180) &
          (abs(lat_vec) <= 90)
        )
    
    if (length(ok_idx) == 0) return(out)
    
    pts <- sf::st_as_sf(
        data.frame(LONGITUDE = lon_vec[ok_idx], LATITUDE = lat_vec[ok_idx]),
        coords = c("LONGITUDE", "LATITUDE"),
        crs = 4326
    )
    
    out[ok_idx] <- as.logical(sf::st_within(pts, boundary_sf, sparse = FALSE)[, 1])
    return(out)
}


#' Prepare Country Boundary
#'
#' @description Merges multiple geographic shapes (e.g., internal administrative regions) into a single unified country boundary. It validates the input, ensures the Coordinate Reference System (CRS) is standard WGS 84 (EPSG:4326), and dissolves internal borders.
#'
#' @param country_shapes_sf An \code{sf} object containing the geographic shapes to be processed.
#'
#' @return A new \code{sf} object containing a single unified geometry column named \code{GEOMETRY}.
#' 
#' @importFrom sf st_crs st_transform st_union st_geometry st_sf
#' @export
prepare_country_boundary <- function(country_shapes_sf) {
    if (!inherits(country_shapes_sf, "sf")) {
        stop("[ERROR] Country shapes must be an sf object.")
    }
    
    # warn the user if the CRS is missing, but don't guess what it is!
    if (is.na(sf::st_crs(country_shapes_sf))) {
        warning("CRS is missing from the input shapefile. Proceeding without a defined CRS.")
    }
    
    country_boundary <- sf::st_union(sf::st_geometry(country_shapes_sf))
    
    # Return as sf object (st_union automatically preserves the original CRS)
    return(sf::st_sf(GEOMETRY = country_boundary))
}



fix_coordinate_pair_in_country <- function(lon, lat, boundary_sf, max_shift = 2) {
	if (is.na(lon) || is.na(lat)) {
		return(list(LONGITUDE = NA_real_, LATITUDE = NA_real_, METHOD = "MISSING_COORDINATES", VALID = FALSE))
	}
	
	candidates <- build_coordinate_candidates(lon, lat, max_shift = max_shift)
	candidate_names <- names(candidates)
	
	m <- matrix(NA_real_, nrow = length(candidate_names), ncol = 2)
	for (j in seq_along(candidate_names)) {
		cand <- candidates[[candidate_names[j]]]
		m[j, 1] <- as.numeric(cand[1])
		m[j, 2] <- as.numeric(cand[2])
	}
	
	earth_ok <- abs(m[, 1]) <= 180 & abs(m[, 2]) <= 90
	if (!any(earth_ok)) {
		return(list(LONGITUDE = NA_real_, LATITUDE = NA_real_, METHOD = "INVALID_NO_MATCH", VALID = FALSE))
	}
	
	pts <- sf::st_as_sf(
		data.frame(LONGITUDE = m[earth_ok, 1], LATITUDE = m[earth_ok, 2]),
		coords = c("LONGITUDE", "LATITUDE"),
		crs = 4326
	)
	
	inside <- rep(FALSE, nrow(m))
	inside[earth_ok] <- as.logical(sf::st_within(pts, boundary_sf, sparse = FALSE)[, 1])
	ok_idx <- which(earth_ok & inside)
	
	if (length(ok_idx) == 0) {
		return(list(LONGITUDE = NA_real_, LATITUDE = NA_real_, METHOD = "INVALID_NO_MATCH", VALID = FALSE))
	}
	
	j <- min(ok_idx)
	list(LONGITUDE = m[j, 1], LATITUDE = m[j, 2], METHOD = candidate_names[j], VALID = TRUE)
}


#' Fix Coordinate Pair within Country Boundary
#'
#' @description Takes a longitude and latitude pair, generates potential candidate 
#' coordinates (e.g., to fix typos or swapped coordinates), and checks which 
#' candidate falls within the provided geographic boundary.
#'
#' @param lon A numeric value representing longitude.
#' @param lat A numeric value representing latitude.
#' @param boundary_sf An \code{sf} polygon or multipolygon object representing the valid boundary.
#' @param max_shift A numeric value for the maximum coordinate shift/adjustment. Defaults to 2.
#'
#' @return A named list containing \code{LONGITUDE}, \code{LATITUDE}, \code{METHOD} 
#' (the name of the successful candidate transformation), and a \code{VALID} boolean flag.
#' 
#' @importFrom sf st_as_sf st_crs st_transform st_within
#' @export
fix_coordinate_pair_in_country <- function(lon, lat, boundary_sf, max_shift = 2) {
	if (is.na(lon) || is.na(lat)) {
		return(list(LONGITUDE = NA_real_, LATITUDE = NA_real_, METHOD = "MISSING_COORDINATES", VALID = FALSE))
	}

	if (is.na(lon) || is.na(lat)) {
		return(list(LONGITUDE = NA_real_, LATITUDE = NA_real_, METHOD = "MISSING_COORDINATES", VALID = FALSE))
	}
	
	candidates <- build_coordinate_candidates(lon, lat, max_shift = max_shift)
	candidate_names <- names(candidates)
	
	m <- matrix(NA_real_, nrow = length(candidate_names), ncol = 2)
	for (j in seq_along(candidate_names)) {
		cand <- candidates[[candidate_names[j]]]
		m[j, 1] <- as.numeric(cand[1])
		m[j, 2] <- as.numeric(cand[2])
	}
	
	# This check relies on standard geographic coordinates (WGS84 degrees)
	earth_ok <- abs(m[, 1]) <= 180 & abs(m[, 2]) <= 90
	if (!any(earth_ok)) {
		return(list(LONGITUDE = NA_real_, LATITUDE = NA_real_, METHOD = "INVALID_NO_MATCH", VALID = FALSE))
	}
	
	# 1. Define the points as standard lat/long (4326) since they are bounded by 180/90
	pts <- sf::st_as_sf(
	data.frame(LONGITUDE = m[earth_ok, 1], LATITUDE = m[earth_ok, 2]),
	coords = c("LONGITUDE", "LATITUDE"),
	crs = 4326
	)
	
	# 2. Dynamically transform the points to match the boundary's native CRS!
	if (!is.na(sf::st_crs(boundary_sf))) {
	pts <- sf::st_transform(pts, sf::st_crs(boundary_sf))
	}
	
	inside <- rep(FALSE, nrow(m))
	inside[earth_ok] <- as.logical(sf::st_within(pts, boundary_sf, sparse = FALSE)[, 1])
	ok_idx <- which(earth_ok & inside)
	
	if (length(ok_idx) == 0) {
	return(list(LONGITUDE = NA_real_, LATITUDE = NA_real_, METHOD = "INVALID_NO_MATCH", VALID = FALSE))
	}
	
	j <- min(ok_idx)
	list(LONGITUDE = m[j, 1], LATITUDE = m[j, 2], METHOD = candidate_names[j], VALID = TRUE)
}


#' Shift Decimal Point Left to Right
#'
#' @description A helper function that corrects misplaced or missing decimal points in a numeric value. 
#'				 It extracts all digits and forcibly places the decimal point exactly \code{k} digits from the left, 
#'				 preserving the original sign.
#'
#' @param value A numeric value (e.g., a coordinate with a typo like 45123 instead of 45.123).
#' @param k An integer specifying how many digits should appear before the new decimal point.
#'
#' @return A numeric value with the shifted decimal point, or \code{NA_real_} if the input is invalid or too short.
#' @export
shift_decimal_left_to_right <- function(value, k) {
  if (is.na(value)) return(NA_real_)

  sign_val <- ifelse(value < 0, -1, 1)
  digits_only <- gsub("[^0-9]", "", as.character(abs(value)))
  if (!nzchar(digits_only) || nchar(digits_only) <= k) return(NA_real_)

  shifted <- paste0(substr(digits_only, 1, k), ".", substr(digits_only, k + 1, nchar(digits_only)))
  sign_val * suppressWarnings(as.numeric(shifted))
}


#' Build Coordinate Correction Candidates
#'
#' @description Generates a comprehensive list of potential corrections for a pair of coordinates based on common human data entry errors. 
#'				 This includes swapping longitude and latitude, flipping signs (missing/extra negatives), and shifting decimal points.
#'
#' @param lon A numeric value representing the original longitude.
#' @param lat A numeric value representing the original latitude.
#' @param max_shift An integer specifying the maximum number of decimal places to test for shifting. Defaults to 2.
#'
#' @return A named list where each element is a numeric vector of length 2 (\code{c(lon, lat)}) representing a candidate correction.
#' @export
build_coordinate_candidates <- function(lon, lat, max_shift = 2) {
  candidates <- list(
    ORIGINAL = c(lon, lat),
    SWAP = c(lat, lon),
    FLIP_LAT = c(lon, -lat),
    FLIP_LON = c(-lon, lat),
    FLIP_BOTH = c(-lon, -lat),
    SWAP_FLIP_LAT = c(lat, -lon),
    SWAP_FLIP_LON = c(-lat, lon),
    SWAP_FLIP_BOTH = c(-lat, -lon)
  )

  for (k in seq_len(max_shift)) {
    lon_lr <- shift_decimal_left_to_right(lon, k)
    lat_lr <- shift_decimal_left_to_right(lat, k)

    candidates[[paste0("DECIMAL_LR_LON_", k)]] <- c(lon_lr, lat)
    candidates[[paste0("DECIMAL_LR_LAT_", k)]] <- c(lon, lat_lr)
    candidates[[paste0("DECIMAL_LR_BOTH_", k)]] <- c(lon_lr, lat_lr)
    candidates[[paste0("SWAP_DECIMAL_LR_LON_", k)]] <- c(lat_lr, lon)
    candidates[[paste0("SWAP_DECIMAL_LR_LAT_", k)]] <- c(lat, lon_lr)
    candidates[[paste0("SWAP_DECIMAL_LR_BOTH_", k)]] <- c(lat_lr, lon_lr)
  }

  candidates
}


#' Plot Corrected Coordinates on Country Boundary
#'
#' @description Extracts longitude and latitude from a results list, converts them to spatial points, and plots them in green over the provided country boundary.
#'
#' @param fix_results A list of coordinate results, where each element contains at least \code{$LONGITUDE} and \code{$LATITUDE}.
#' @param shapes_sf_boundary An \code{sf} polygon/multipolygon object representing the country boundaries.
#'
#' @return A \code{ggplot} object containing the map.
#' 
#' @importFrom sf st_as_sf
#' @importFrom ggplot2 ggplot geom_sf theme_minimal labs
#' @export
plot_fixed_coordinates <- function(fix_results, shapes_sf_boundary) {
  
  # 1. Extract the coordinates from the list into a dataframe
  # We use sapply to pull out all lons and lats quickly
  pts_df <- data.frame(
    LONGITUDE = sapply(fix_results, function(x) x$LONGITUDE),
    LATITUDE = sapply(fix_results, function(x) x$LATITUDE)
  )
  
  # 2. Filter out any rows where coordinates are NA (so ggplot doesn't complain)
  pts_df <- pts_df[!is.na(pts_df$LONGITUDE) & !is.na(pts_df$LATITUDE), ]
  
  if (nrow(pts_df) == 0) {
    message("No valid coordinates to plot.")
    return(NULL)
  }
  
  # 3. Convert the regular dataframe into an 'sf' spatial object (assuming standard 4326 CRS)
  pts_sf <- sf::st_as_sf(pts_df, coords = c("LONGITUDE", "LATITUDE"), crs = 4326)
  
  # 4. Create the plot using ggplot2
  map_plot <- ggplot2::ggplot() +
    # Layer 1: The country boundary (transparent fill, black outline)
    ggplot2::geom_sf(data = shapes_sf_boundary, fill = "transparent", color = "black", size = 0.5) +
    # Layer 2: The fixed points (green, size 2)
    ggplot2::geom_sf(data = pts_sf, color = "green", size = 2) +
    # Clean up the background
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = "Corrected Coordinates Validation",
      subtitle = paste("Showing", nrow(pts_df), "points inside boundaries")
    )
  
  # Print the plot to the Viewer/Notebook
  print(map_plot)
  
  # Return the plot object in case you want to save it later using ggsave()
  invisible(map_plot)
}