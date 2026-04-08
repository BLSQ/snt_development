# Shared helpers for snt_dhis2_formatting code notebooks.

# Load base utils
source(file.path("~/workspace/code", "snt_utils.r"))   



#' Get Setup Variables for SNT Workspace
#' Initializes workspace paths, loads R packages, and imports OpenHEXA SDK.
#'
#' @param SNT_ROOT_PATH Character. Root path of the SNT workspace. Default: '~/workspace'
#' @param packages Character vector. R packages to install and load.
#' @return List with CONFIG_PATH, POPULATION_DATA_PATH.
#'
#' @export
get_setup_variables <- function(
    SNT_ROOT_PATH='~/workspace', 
    packages=c("arrow", "dplyr", "tidyr", "stringr", "stringi", "jsonlite", "httr", "glue", "reticulate")
) {
        
    # List required pcks
    required_packages <- packages
    install_and_load(required_packages)
    
    # Set environment to load openhexa.sdk from the right environment
    Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")
    reticulate::py_config()$python
    
    # Import OpenHEXA SDK and make it available in the environment 
    assign("openhexa", reticulate::import("openhexa.sdk"), envir = .GlobalEnv)
    
    return(
        list(
            CONFIG_PATH = file.path(SNT_ROOT_PATH, "configuration"),  
            FORMATTED_DATA_PATH = file.path(SNT_ROOT_PATH, "data", "dhis2", "extracts_formatted")
        )
    )
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
                    msg <- glue::glue("[ERROR] Error while loading configuration: {snt_config_path}")
                    cat(msg)
                    stop(msg)
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
            if (verbose) log_msg(glue("[ERROR] Error while loading {filename} file for: {conditionMessage(e)}"), "error")
            stop(msg)
    })

    if (verbose) {
        msg <- glue("{filename} data loaded from dataset : {dataset_id} dataframe dimensions: [{paste(dim(data), collapse=', ')}]")
        log_msg(msg)
    }    
    return(data)
}


# ------------------------------------------------------------------------------------
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
build_indicators <- function(data, valid_indicators, empty_indicators, include_empty_ind=TRUE) {

    # loop over the definitions
    empty_data_indicators <- c()
    for (indicator in names(valid_indicators)) {
            
        data_element_uids <- valid_indicators[[indicator]]    
        col_names <- c()
    
        if (length(data_element_uids) > 0) {
            for (dx in data_element_uids) {
                dx_co <- gsub("\\.", "_", dx)            
                if (grepl("_", dx_co)) {
                    col_names <- c(col_names , dx_co)
                } else {
                    if (!any(grepl(dx, colnames(data)))) {  # is there no dx what match?
                        msg <- paste0("Data element : " , dx, " of indicator ", indicator , " is missing in the DHIS2 routine data.")
                        log_msg(msg, level="warning")
                    } else {
                        col_names <- c(col_names , colnames(data)[grepl(dx, colnames(data))])
                    }                
                }
            }
        
            # check if there are matching data elements
            if (length(col_names) == 0) {
                msg <- paste0("No data elements available to build indicator : " , indicator, ", skipped.")
                log_msg(msg, level="warning")
                empty_data_indicators <- c(empty_data_indicators, indicator)
                next
            }
            
            # logs
            msg <- paste0("Building indicator : ", indicator, " -> column selection : ", paste(col_names, collapse = ", "))        
            log_msg(msg)
            
            if (length(col_names) > 1) {
                sums <- rowSums(data[, col_names], na.rm = TRUE)
                all_na <- rowSums(!is.na(data[, col_names])) == 0
                sums[all_na] <- NA  # Keep NA if all rows are NA!
                data[[indicator]] <- sums            
            } else {
                data[indicator] <- data[, col_names] 
            }
            
        } else {
            data[indicator] <- NA
            
            # logs
            msg <- paste0("Building indicator : ", indicator, " -> column selection : NULL")
            log_msg(msg)
        }
    }

    # Add the empty indicator columns (if not needed this can be commented)
    if (include_empty_ind) {
        for (empty_indicator in empty_indicators) {
            data[empty_indicator] <- NA
            
            # logs
            msg <- paste0("Building indicator : ", empty_indicator, " -> column selection : NULL")
            log_msg(msg)
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
        stop("Input must be an sf object")
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




#
# Claude: Helpers for DHIS2 formatting pipeline.
#

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

shift_decimal_left_to_right <- function(value, k) {
  if (is.na(value)) return(NA_real_)

  sign_val <- ifelse(value < 0, -1, 1)
  digits_only <- gsub("[^0-9]", "", as.character(abs(value)))
  if (!nzchar(digits_only) || nchar(digits_only) <= k) return(NA_real_)

  shifted <- paste0(substr(digits_only, 1, k), ".", substr(digits_only, k + 1, nchar(digits_only)))
  sign_val * suppressWarnings(as.numeric(shifted))
}

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

prepare_country_boundary <- function(country_shapes_sf) {
  if (!inherits(country_shapes_sf, "sf")) {
    stop("Country shapes must be an sf object.")
  }

  if (is.na(sf::st_crs(country_shapes_sf))) {
    sf::st_crs(country_shapes_sf) <- 4326
  }

  country_shapes_sf <- sf::st_transform(country_shapes_sf, 4326)
  country_boundary <- sf::st_union(sf::st_geometry(country_shapes_sf))
  sf::st_sf(GEOMETRY = country_boundary)
}

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
  out
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
