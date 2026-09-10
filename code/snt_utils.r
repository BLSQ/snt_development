# ================================================
# Title: Utility Functions for SNT Process
# Description: This script contains utility functions used for SNT computation workflow.
# Author: Esteban Montandon
# Created: [2024-10-01]
# Last updated: [2026-09-09]
# Dependencies: stringi, httr, arrow, tools, jsonlite, data.table, dplyr, tidyr, sf, terra, glue,
#   tsibble, fable, DHS.rates, readr, rlang, utils, reticulate
# Notes:
#   - [Optional: Any special considerations, references, or tips]
# ================================================


#' Format Free-Text Strings for Matching
#'
#' Converts accented/special characters to ASCII, replaces non-alphanumeric
#' characters with spaces, uppercases the string, and collapses repeated
#' whitespace, producing a normalized string suitable for matching.
#'
#' @param x Character. String(s) to format.
#' @return Character. The formatted, normalized string(s).
#'
#' @export
format_names <- function(x) {
    # add any other matching logic here
    x <- stri_trans_general(str = x, id = "Latin-ASCII") # remove weird characters
    x <- gsub("[^a-zA-Z0-9]", " ", toupper(x))           # replace non-alphanum with space
    # x <- gsub("(?i)PROVINCE|ZONE DE SANTE|AIRE DE SANTE|CENTRE DE SANTE", "", x) # TEMPORARY SKIP
    x <- gsub("  +", " ", x)       # collapse multiple spaces
    trimws(x)
}


#' Clean Column Names of a Data Frame
#'
#' Replaces non-alphanumeric characters with underscores, removes internal
#' whitespace, and converts column names to uppercase.
#'
#' @param df Data frame or data.table. Table whose column names will be cleaned.
#' @return Character vector of cleaned column names.
#'
#' @export
clean_column_names <- function(df) {
    # Get column names
    col_names <- colnames(df)

    # Apply the transformation rules
    cleaned_names <- gsub("[^a-zA-Z0-9]", "_", col_names)  # Replace symbols with underscores
    cleaned_names <- gsub("\\s+", "", cleaned_names)       # Remove extra spaces
    cleaned_names <- toupper(cleaned_names)                # Convert to uppercase
    # Return cleaned column names
    return(trimws(cleaned_names))
}


#' Install and Load Required R Packages
#'
#' Installs any packages not already present (also ensuring 'scales' is at
#' least version 1.3.0), loads all requested packages, and prints their
#' loaded versions.
#'
#' @param packages Character vector. Names of the packages to install (if
#'   missing) and load.
#' @return Invisibly, a named character vector of package name/version strings.
#'
#' @export
install_and_load <- function(packages) {
    #  is the one that interferes with loading {tidyverse} if not updated version
    if (!requireNamespace("scales", quietly = TRUE) || packageVersion("scales") < "1.3.0") {
        suppressMessages(install.packages("scales"))
    }

    # Create vector of packages that are not installed
    missing_packages <- packages[!(packages %in% installed.packages()[, "Package"])]

    # Install missing packages
    if (length(missing_packages) > 0) {
        suppressMessages(install.packages(missing_packages))
    }

    # Load all the packages
    suppressMessages(lapply(packages, require, character.only = TRUE))

    # Retrieve and print package names and versions
    loaded_packages <- sapply(packages, function(pkg) {
        paste(pkg, packageVersion(pkg), sep = " ")
    })
    print(loaded_packages)
}


#' Initialize SNT Workspace Setup
#'
#' Initializes workspace paths, installs and loads required R packages,
#' imports the OpenHEXA SDK, and creates any missing workspace directories.
#'
#' @param snt_pipeline_name Character. Name of the SNT pipeline; used to build
#'   the pipeline-specific intermediate results path. Must be a non-empty string.
#' @param snt_root_path Character. Root path of the SNT workspace. Default: '~/workspace'
#' @param packages Character vector. R packages to install and load.
#' @return List with SNT paths.
#'
#' @export
init_snt_workspace <- function(
    snt_pipeline_name,
    snt_root_path='~/workspace',    
    packages=c("arrow", "dplyr", "tidyr", "stringr", "stringi", "jsonlite", "httr", "glue")
) {
    # Validate snt_pipeline_name
    if (missing(snt_pipeline_name) || is.null(snt_pipeline_name) || 
        !nzchar(trimws(snt_pipeline_name))) {
        stop("`snt_pipeline_name` must be provided and cannot be empty.", call. = FALSE)
    }
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
        CONFIG_PATH = file.path(snt_root_path, "configuration"),  
        UPLOADS_PATH = file.path(snt_root_path, "uploads"),
        DATA_PATH = file.path(snt_root_path, "data"),
        INTERMEDIATE_RESULTS = file.path(snt_root_path, "pipelines", snt_pipeline_name, "intermediate_results")        
    )
    lapply(paths_to_check, dir.create, recursive = TRUE, showWarnings = FALSE) # create if they do not exist
    return(paths_to_check)
}


#' Load the SNT Configuration File
#'
#' Reads and parses the SNT configuration JSON file at the given path,
#' logging a message on success and stopping with a clear error if loading
#' fails.
#'
#' @param snt_config_path Character. Full path to the SNT configuration JSON file.
#' @return List. The parsed configuration.
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
#'
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
#' Attempts to read a CSV file from the specified path. If the file cannot be
#' loaded, it logs a high-level error message and stops execution.
#'
#' @param csv_file_path Character. Path to the CSV file.
#' @return Data frame. The contents of the CSV file.
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


#' Safely Read GeoJSON File
#'
#' Reads a GeoJSON file from a specified path with built-in error handling.
#' Checks if the file exists and catches parsing errors if the file is corrupted.
#'
#' @param file_path Character. Full path to the GeoJSON file.
#' @return sf object with the spatial data if successful, or NULL if the process fails.
#'
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


#' Normalize YEAR and MONTH Column Types
#'
#' Converts the YEAR and MONTH columns of a data frame to integer type when
#' present, leaving other columns unchanged.
#'
#' @param input_df Data frame. Input data possibly containing YEAR and/or MONTH columns.
#' @param year_col Character. Name of the year column. Default: 'YEAR'.
#' @param month_col Character. Name of the month column. Default: 'MONTH'.
#' @return Data frame with YEAR and MONTH columns coerced to integer, where present.
#'
#' @export
normalize_year_month_types <- function(input_df, year_col = "YEAR", month_col = "MONTH") {
  output_df <- input_df
  if (year_col %in% names(output_df)) {
    output_df[[year_col]] <- as.integer(output_df[[year_col]])
  }
  if (month_col %in% names(output_df)) {
    output_df[[month_col]] <- as.integer(output_df[[month_col]])
  }
  return(output_df)
}


#' Reshape Routine Data to Long Format
#'
#' Selects fixed and indicator columns, pivots the indicator columns to long
#' format, and optionally deduplicates rows on standard identifying keys.
#'
#' @param routine_df Data frame. Routine data in wide format.
#' @param fixed_cols Character vector. Names of non-indicator columns to keep.
#' @param indicators Character vector. Names of indicator columns to pivot to long format.
#' @param deduplicate Logical. Whether to remove duplicate rows on the standard key columns. Default: TRUE.
#' @return Data frame in long format with INDICATOR and VALUE columns.
#'
#' @export
prepare_routine_long <- function(routine_df, fixed_cols, indicators, deduplicate = TRUE) {
  cols_to_select <- intersect(c(fixed_cols, indicators), names(routine_df))
  missing_indicators <- setdiff(indicators, names(routine_df))
  if (length(missing_indicators) > 0) {
    stop(paste0("[ERROR] Missing indicator column(s): ", paste(missing_indicators, collapse = ", ")))
  }

  routine_long <- routine_df %>%
    dplyr::select(dplyr::all_of(cols_to_select)) %>%
    tidyr::pivot_longer(
      cols = dplyr::all_of(indicators),
      names_to = "INDICATOR",
      values_to = "VALUE"
    )

  if (deduplicate) {
    dedup_keys <- intersect(c("ADM1_ID", "ADM2_ID", "OU_ID", "PERIOD", "YEAR", "MONTH", "INDICATOR"), names(routine_long))
    routine_long <- routine_long %>%
      dplyr::distinct(dplyr::across(dplyr::all_of(dedup_keys)), .keep_all = TRUE)
  }

  return(routine_long)
}


#' Build a Standardized Output Path
#'
#' Constructs a path under the data root for a given domain (and optional
#' subdomain), creating the directory if requested.
#'
#' @param data_root_path Character. Root data directory.
#' @param domain Character. Domain subfolder name.
#' @param subdomain Character. Optional subdomain subfolder name. Default: NULL.
#' @param create_dir Logical. Whether to create the directory if it doesn't exist. Default: TRUE.
#' @return Character. The constructed output path.
#'
#' @export
standard_output_path <- function(data_root_path, domain, subdomain = NULL, create_dir = TRUE) {
  target_path <- if (is.null(subdomain) || nchar(subdomain) == 0) {
    file.path(data_root_path, domain)
  } else {
    file.path(data_root_path, domain, subdomain)
  }

  if (create_dir && !dir.exists(target_path)) {
    dir.create(target_path, recursive = TRUE, showWarnings = FALSE)
  }

  return(target_path)
}


#' Safely Retrieve a Parameter with a Default Fallback
#'
#' Retrieves a named parameter from a list if present and non-null, casting
#' it with the given function; otherwise returns the default value.
#'
#' @param params_list List. Input parameters (can be NULL).
#' @param target_param Character. Name of the parameter to retrieve.
#' @param default The default value to return if the parameter is missing or NULL.
#' @param cast_method Function. Used to cast the retrieved value. Default: identity.
#' @return The (cast) parameter value, or the default.
#'
#' @export
get_param <- function(params_list, target_param, default, cast_method = identity) {
  if (!is.null(params_list) && is.list(params_list) &&
      !is.null(params_list[[target_param]])) {
    return(cast_method(params_list[[target_param]]))
  } else {
    return(default)
  }
}


#' Download and Read a File from the Latest Dataset Version
#'
#' Fetches the latest version of an OpenHEXA dataset, downloads the requested
#' file into memory, and parses it based on its extension (parquet, csv,
#' geojson, or json).
#'
#' @param dataset Character. Identifier of the OpenHEXA dataset.
#' @param filename Character. Name of the file to retrieve.
#' @return Data frame (or sf object for geojson, list for json) with the file's content.
#'
#' @export
get_latest_dataset_file_in_memory <- function(dataset, filename) {
    # Get the dataset file object

    dataset_last_version <- openhexa$workspace$get_dataset(dataset)$latest_version
    dataset_file <- dataset_last_version$get_file(filename)

    # Perform the GET request and keep the content in memory
    response <- httr::GET(dataset_file$download_url)

    if (httr::status_code(response) != 200) {
        stop("Failed to download the file.")
    }

    print(paste0("File downloaded successfully from dataset version: ",dataset_last_version$name))

    # Convert the raw content to a raw vector (the content of the file)
    raw_content <- httr::content(response, as = "raw")
    temp_file <- rawConnection(raw_content, "r")
    file_extension <- tolower(tools::file_ext(filename))

    if (file_extension == "parquet") {
        df <- arrow::read_parquet(temp_file)
    }
    else if (file_extension == "csv") {
        df <- utils::read.csv(temp_file, stringsAsFactors = FALSE)
    }
    else if (file_extension == "geojson") {
        tmp_geojson <- tempfile(fileext = ".geojson")
        writeBin(raw_content, tmp_geojson)
        df <- sf::st_read(tmp_geojson, quiet = TRUE)
    }
    else if (file_extension == "json") {
        # Read JSON from raw content
        json_txt <- rawToChar(raw_content)
        df <- jsonlite::fromJSON(json_txt, flatten = TRUE)
    }
    else {
      stop(paste("Unsupported file type:", file_extension))
    }

    # Return the dataframe
    return(df)
}


#' Log a Message to OpenHEXA and the Console
#'
#' Prints a message and, if connected to an OpenHEXA pipeline run, also logs
#' it at the given severity level (info, warning, or error). Safe to call
#' outside a pipeline run: if the `openhexa` object doesn't exist or has no
#' current run, the message is only printed locally instead of raising an
#' error.
#'
#' @param msg Character. The message to log.
#' @param level Character. Log level: 'info', 'warning', or 'error'. Default: 'info'.
#'   Only validated when connected to a pipeline run; an unrecognized level is
#'   silently ignored otherwise.
#' @return Invisible NULL. Called for its side effects.
#'
#' @export
log_msg <- function(msg, level = "info") {
  print(msg)
  
  has_openhexa_run <- tryCatch({
    exists("openhexa") && !is.null(openhexa) && !is.null(openhexa$current_run)
  }, error = function(e) FALSE)
  
  if (has_openhexa_run) {
    level <- tolower(level)
    if (level == "info") {
      openhexa$current_run$log_info(msg)
    } else if (level == "warning") {
      openhexa$current_run$log_warning(msg)
    } else if (level == "error") {
      openhexa$current_run$log_error(msg)
    } else {
      stop("Unsupported log level")
    }
  }
}


#' Log a Message, Falling Back When Not Connected to a Pipeline
#'
#' Attempts to log a message via log_msg(); if it fails because the OpenHEXA
#' pipeline object is unavailable, prints the message locally instead of
#' raising an error.
#'
#' @param target_msg Character. The message to log.
#' @param pipeline_error Character. Error text identifying the "not connected
#'   to pipeline" condition. Default: "object 'openhexa' not found".
#' @param extra_info Character. Message printed when falling back to local
#'   logging. Default: 'not connected to pipeline run'.
#' @return Invisible NULL. Called for its side effects.
#'
#' @export
pipeline_msg <- function(
  target_msg,
  pipeline_error = "object 'openhexa' not found",
  extra_info = "not connected to pipeline run"
) {
  tryCatch(
    {
      log_msg(target_msg)
    },
    error = function(e) {
      # if the error is about openhexa
      if (grepl(pipeline_error, e$message)) {
        print(extra_info)
        print(target_msg)
      } else {
        # rethrow the other errors
        stop(e)
      }
    }
  )
}


#' Create a Directory if it Doesn't Already Exist
#'
#' Checks whether a directory exists and creates it (including parent
#' directories) if not, printing a message either way.
#'
#' @param dir_path Character. Path of the directory to create.
#' @return Invisible NULL. Called for its side effects.
#'
#' @export
safe_create_dir <- function(dir_path){
  # Check if the directory exists, and if not, create it
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
    message("Directory created: ", dir_path)
  } else {
    message("Directory already exists: ", dir_path)
  }
}


#' Export Data to CSV or Parquet
#'
#' Creates the output directory if needed and writes the given object to disk
#' as CSV or Parquet, based on the file extension.
#'
#' @param data_object Data frame. The data to export.
#' @param file_path Character. Destination file path (must end in '.csv' or '.parquet').
#' @return Invisible NULL. Called for its side effects.
#'
#' @export
export_data <- function(data_object, file_path) {

    # Get directory and create if it doesn't exist
    output_dir <- dirname(file_path)
    if (!dir.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE)
        log_msg(paste0("Output folder created : ", output_dir))
    }

    # get file name extension
    file_extension <- tools::file_ext(file_path)

    # Export the data based on file type
    if (file_extension == "csv") {
        write_csv(data_object, file_path)
    } else if (file_extension == "parquet") {
        arrow::write_parquet(data_object, file_path)
    } else {
        stop("Unsupported file type. Please use 'csv' or 'parquet'.")
    }

    # Log the export
    log_msg(paste0("Exported : ", file_path))
}


#' Convert Data Table Columns to Target Types
#'
#' Modifies a data.table in place, converting the columns listed under each
#' type name in col_type_map to that type.
#'
#' @param dt data.table. Table to modify in place.
#' @param col_type_map List. Named list mapping type names (e.g. 'numeric') to
#'   vectors of column names.
#' @return Invisible NULL; dt is modified in place.
#'
#' @export
convert_columns <- function(dt, col_type_map) {

  for (type in names(col_type_map)) {
    cols <- col_type_map[[type]]
    convert_fun <- match.fun(paste0("as.", type))
    dt[, (cols) := lapply(.SD, convert_fun), .SDcols = cols]
  }
}


#' Retrieve the Latest Dataset File and Log its Dimensions
#'
#' Loads the given file from the latest version of a dataset, logging its
#' dimensions, with an informative error message on failure.
#'
#' @param target_dataset Character. The dataset containing the file.
#' @param target_filename Character. Name of the file to retrieve.
#' @param target_country_code Character. Country code, used in error messages.
#' @return The loaded data (data frame, sf object, or list, depending on file type).
#'
#' @export
get_newest_dataset_file <- function(target_dataset, target_filename, target_country_code){

  # load file
  output_data <- tryCatch({ get_latest_dataset_file_in_memory(target_dataset, target_filename) },
                          error = function(e) {
                            msg <- glue("Error while loading {target_country_code} {target_filename}, {conditionMessage(e)}")  # log error message
                            cat(msg)
                            stop(msg)
                          })
  data_dimensions <- paste(dim(output_data), collapse=", ")

  msg <- glue("{target_filename} loaded from dataset {target_dataset}; the dataframe has the following dimensions: {data_dimensions}")
  log_msg(msg)

  return(output_data)
}


#' Match Column Classes to a Reference Data Table
#'
#' Coerces columns in input_dt to match the classes of same-named columns in
#' reference_dt, for common columns whose classes differ.
#'
#' @param input_dt Data frame or data.table. Table whose columns will be adapted.
#' @param reference_dt data.table. Table with the target column classes.
#' @return data.table. A copy of input_dt with common columns coerced to match reference_dt's classes.
#'
#' @export
match_column_classes <- function(input_dt, reference_dt) {
  output_dt <- copy(as.data.table(input_dt))

  if (!is.data.table(reference_dt)) {
    stop("The reference data must be a data.table.")
  }

  common_cols <- intersect(names(output_dt), names(reference_dt))

  for (col in common_cols) {
    ref_class <- class(reference_dt[[col]])

    # keep reference semantics
    if (!inherits(output_dt[[col]], ref_class)) {
      new_col <- switch(
        ref_class[1],  # use only the primary class
        character = as.character(output_dt[[col]]),
        integer   = as.integer(output_dt[[col]]),
        numeric   = as.numeric(output_dt[[col]]),
        factor    = as.factor(output_dt[[col]]),
        Date      = as.Date(output_dt[[col]]),
        {
          warning(paste("Unsupported class for column:", col, "-", ref_class[1]))
          output_dt[[col]]
        }
      )
      set(output_dt, j = col, value = new_col)
    }
  }
  return(output_dt)
}


#' Build a Cartesian Product of Admin Units and Monthly Periods
#'
#' Creates the full cross join of unique administrative units and a
#' continuous monthly sequence spanning the minimum to maximum period found
#' in the input data, to ensure every place-time combination exists.
#'
#' @param input_dt Data frame or data.table. Input data containing admin, year, and month columns.
#' @param admin_colname Character. Name of the administrative unit column.
#' @param year_colname Character. Name of the year column.
#' @param month_colname Character. Name of the month column.
#' @return List of length two: the total number of periods in the generated
#'   sequence, and a data.table with the cartesian place-time rows.
#'
#' @export
make_cartesian_admin_period <- function(input_dt, admin_colname, year_colname, month_colname) {

  dt <- copy(as.data.table(input_dt))

  # select only relevant columns to work with
  cols <- c(admin_colname, year_colname, month_colname)
  dt <- dt[, ..cols]

  # make the table of unique administrative units
  admin_dt <- unique(dt[, .(
    get(admin_colname),
    placeholder = 1
  )])
  setnames(admin_dt, old = names(admin_dt)[1], new = admin_colname)

  # make the table with all possible monthly periods, between the minimum and the maximum of input_dt
  dt[, date := as.IDate(paste0(dt[[year_colname]], '-', dt[[month_colname]], '-01'))]
  min_date <- min(dt$date, na.rm = TRUE)
  max_date <- max(dt$date, na.rm = TRUE)
  date_seq <- seq(min_date, max_date, by = "1 month")
  dates_dt <- data.table(
    YEAR = year(date_seq),
    MONTH = month(date_seq),
    placeholder = rep(1, length(date_seq))
  )

  # make the cartesian product between administrative units and monthly periods, using a placeholder column
  result_dt <- merge.data.table(admin_dt, dates_dt, by = c('placeholder'), allow.cartesian = TRUE)
  result_dt <- result_dt[, -'placeholder']

  return(list(nrow(dates_dt), result_dt))
}


#' Cartesian Product of a Data Table and a Vector
#'
#' Cross-joins every row of input_dt with every element of input_vector,
#' adding the vector's values as a new column.
#'
#' @param input_dt Data frame or data.table. Input data.
#' @param input_vector Vector. Values used to expand rows.
#' @param new_colname Character. Name of the new column holding the vector's values.
#' @return data.table with the cartesian product of the two inputs.
#'
#' @export
make_cartesian_dt_vector <- function(input_dt, input_vector, new_colname){

  if (!is.data.table(input_dt)) {
    input_dt <- as.data.table(input_dt)
  }

  # vector to data table
  vector_dt <- setnames(data.table(input_vector), new_colname)

  # dummy columns to both for cross join
  input_dt[, dummy := 1]
  vector_dt[, dummy := 1]

  # cartesian
  output_dt <- merge(input_dt, vector_dt, by = "dummy", allow.cartesian = TRUE)

  # temove the dummy column
  output_dt[, dummy := NULL]

  return(output_dt)
}


#' Merge Data onto a Full Time-Space Grid
#'
#' Left-joins input_dt with a set of full time-space rows (e.g. from
#' make_cartesian_admin_period()), introducing NA rows ("holes") for any
#' missing admin-period combinations in the target column.
#'
#' @param input_dt Data frame or data.table. Input data containing the target column and common keys.
#' @param full_rows_dt Data frame or data.table. Full set of admin-period rows to merge with.
#' @param target_colname Character. Column to keep from input_dt, which may end up with missing values.
#' @param admin_colname Character. Name of the administrative unit column. Default: 'ADM2_ID'.
#' @param year_colname Character. Name of the year column. Default: 'YEAR'.
#' @param month_colname Character. Name of the month column. Default: 'MONTH'.
#' @return data.table merged on the common keys, with NA in target_colname
#'   wherever no matching row existed.
#'
#' @export
make_full_time_space_data <- function(input_dt, full_rows_dt, target_colname, admin_colname = 'ADM2_ID', year_colname = 'YEAR', month_colname = 'MONTH') {

  common_colnames <- c(admin_colname, year_colname, month_colname)

  # make sure data is data table
  output_dt <- copy(as.data.table(input_dt)[, .SD, .SDcols = c(common_colnames, target_colname)])
  full_rows_dt <- as.data.table(full_rows_dt)[, .SD, .SDcols = common_colnames]

  output_dt <- merge.data.table(
    output_dt,
    full_rows_dt,
    by = common_colnames,
    all = TRUE
  )

  # # fill in missings in the administrative unit columns for future imputation grouping
  # output_dt <- output_dt[, `:=`(
  #   ADM1_ID = ifelse(is.na(ADM1_ID), unique(ADM1_ID[!is.na(ADM1_ID)]), ADM1_ID),
  #   ADM1 = ifelse(is.na(ADM1), unique(ADM1[!is.na(ADM1)]), ADM1),
  #   ADM2 = ifelse(is.na(ADM2), unique(ADM2[!is.na(ADM2)]), ADM2)
  # ), by = ADM2_ID]
  #
  return(output_dt)
}


#' Extract All Rows for Groups Containing Missing Values
#'
#' Identifies groups with at least one missing value in the target column,
#' then returns all rows belonging to those groups.
#'
#' @param input_dt data.table. Input data.
#' @param target_colname Character. Column where missing values should be identified.
#' @param id_colname Character. Grouping column (unit of observation).
#' @return data.table with all observations for groups containing at least one missing value.
#'
#' @export
extract_dt_with_missings <- function(input_dt, target_colname, id_colname){

  ids_with_missings <- input_dt[is.na(get(target_colname)), unique(get(id_colname))]
  dt_with_missings <- input_dt[get(id_colname) %in% ids_with_missings]
  return(dt_with_missings)
}


#' Impute Missing Values with a Seasonal ARIMA Model
#'
#' Fits a seasonal ARIMA model on the log of the target variable for a single
#' admin unit's time series and uses it to interpolate missing values, then
#' restores the original scale.
#'
#' @param district_data data.table. Time series data for a single administrative unit.
#' @param original_values_colname Character. Column containing the values to be imputed.
#' @param estimated_values_colname Character. Name of the new column to hold the imputed values.
#' @param admin_colname Character. Column containing the administrative unit id.
#' @param period_colname Character. Column containing the year-month period.
#' @param threshold_for_missing Numeric. Values at or below this threshold
#'   are treated as missing for imputation purposes. Default: 0.0.
#' @return data.table with the new estimated_values_colname column added.
#'
#' @export
fill_missing_cases_ts <- function(district_data, original_values_colname, estimated_values_colname, admin_colname, period_colname, threshold_for_missing = 0.0){

  district_id <- district_data[, unique(get(admin_colname))]

  # compute the log, to avoid estimating negative values during imputation
  # values of 0 are re-added back at the end
  log_values_colname = paste(original_values_colname, 'LOG', sep = '_')
  district_data[, (log_values_colname) := ifelse(get(original_values_colname) > threshold_for_missing, log(get(original_values_colname)), NA)]
  district_data$PERIOD <- yearmonth(district_data$PERIOD)
  district_ts <- tsibble(district_data, index = PERIOD)

  # fit ARIMA model to the column with missing values, then estimate values based on model
  ts_fill <- district_ts |>
    # for parsimony and speed, so the fit doesn't go crazy in the orders to chase good AIC's
    model(predefined_sarima = ARIMA(!!sym(log_values_colname) ~ 0 + pdq(1, 1, 0) + PDQ(1, 1, 0))) |>
    interpolate(district_ts) |>
    mutate(!!sym(estimated_values_colname) := round(exp(!!sym(log_values_colname)))) |>
    mutate(!!sym(admin_colname) := district_id)

  district_data_filled <- as.data.table(ts_fill)

  # drop the log of cases and merge the data
  district_data_filled <- merge.data.table(
    district_data[, (log_values_colname) := NULL],
    district_data_filled[, (log_values_colname) := NULL],
    by = c(admin_colname, period_colname)
  )

  # reformat back to original
  district_data_filled[, YEAR := year(PERIOD)]
  district_data_filled[, MONTH := month(PERIOD)]

  district_data_filled[, PERIOD := NULL]

  # this is the general situation, if outlier detection has already happened; in this specific case, the line below should not be run, because all zeroes are errors
  district_data_filled[get(original_values_colname) <= threshold_for_missing, (estimated_values_colname) := get(original_values_colname)]

  return(district_data_filled)
}


#' Add String Versions of Selected Integer Columns
#'
#' For each column matching input_pattern, creates a new character column
#' (named via output_pattern) with NA values replaced by a placeholder label.
#'
#' @param input_dt Data frame or data.table. Table to modify.
#' @param input_pattern Character. Regex pattern used to identify columns to convert.
#' @param output_pattern Character. Replacement pattern used to form new column names.
#' @param missing_label Character. String used to replace NA values in the new columns. Default: '<=60%'.
#' @return data.table containing the original data plus the newly created string columns.
#'
#' @export
add_str_col_from_int <- function(input_dt, input_pattern, output_pattern, missing_label = "<=60%"){

  output_dt <- copy(as.data.table(input_dt))

  old_colnames <- grep(input_pattern, names(input_dt), value=TRUE)

  for(old_colname in old_colnames){
  new_colname <- gsub(input_pattern, output_pattern, old_colname)
  output_dt[, (new_colname) := as.character(get(old_colname))]
  output_dt[is.na(get(new_colname)), (new_colname) := missing_label]
  }

  return(output_dt)
}


#' Bin a Numeric Column into Categories
#'
#' Adds a new ordered factor column to a data.table by cutting a numeric
#' column into bins, using cut().
#'
#' @param dt data.table. Table to modify in place.
#' @param breaks Numeric vector. Cut points defining bin boundaries.
#' @param labels Character vector. Labels for the resulting bins.
#' @param col_in Character. Name of the numeric input column to bin.
#' @param col_out Character. Name of the output column to create.
#' @param include.lowest Logical. Whether to include the lowest breakpoint. Default: TRUE.
#' @param right Logical. Whether intervals are closed on the right. Default: FALSE.
#' @return Invisibly, the modified data.table (dt is also modified in place).
#'
#' @export
bin_column_dt <- function(dt, breaks, labels,col_in, col_out, include.lowest = TRUE, right = FALSE) {

  if (!is.data.table(dt)) {
    stop("dt must be a data.table")
  }

  dt[, (col_out) := cut(
    get(col_in),
    breaks = breaks,
    labels = labels,
    include.lowest = include.lowest,
    right = right,
    ordered_result = TRUE
  )]

  invisible(dt)
}


#%% SEASONALITY COMPUTATION -------------------------------------------------------------------


#' Compute Month-Level Seasonality Indicators
#'
#' Computes month-level seasonality indicators using forward-looking month
#' blocks. By default implements the WHO month-block reasoning for
#' seasonality computation; a calendar-year denominator is available via a
#' parameter.
#'
#' @param input_dt Data frame or data.table. Input data.
#' @param indicator Character. Type of indicator (case/rainfall/etc.); uppercased and used to build the output column names.
#' @param values_colname Character. Name of the indicator column on which the computations are made.
#' @param vector_of_durations Numeric vector. Number of months in a block (e.g. 3/4/5).
#' @param admin_colname Character. Name of the administrative unit column to group by. Default: 'ADM2_ID'.
#' @param year_colname Character. Name of the year column. Default: 'YEAR'.
#' @param month_colname Character. Name of the month column. Default: 'MONTH'.
#' @param proportion_threshold Numeric. Proportion of the indicator which needs to occur in a block, to qualify for seasonality. Default: 0.6.
#' @param use_calendar_year_denominator Logical. If TRUE, uses the total accumulated value of the current
#'   calendar year (Jan-Dec) as the denominator. If FALSE (default), uses the 12-month forward-looking
#'   sliding window as denominator (WHO approach).
#' @return data.table with the additional columns for seasonality indicators.
#'
#' @export
compute_month_seasonality <- function(input_dt, indicator, values_colname, vector_of_durations,
                                      admin_colname = 'ADM2_ID', year_colname = 'YEAR', month_colname = 'MONTH',
                                      proportion_threshold = 0.6,
                                      use_calendar_year_denominator = FALSE) {

  indicator <- toupper(indicator)
  output_dt <- copy(as.data.table(input_dt))

  # ensure correct order
  output_dt <- output_dt[order(get(admin_colname), get(year_colname), get(month_colname))]

  # compute denominator
  if (use_calendar_year_denominator) {
    # alternative approach (calendar): per calendar year
    denominator_colname <- paste(indicator, "SUM_CALENDAR_YEAR", sep = "_")

    output_dt[, (denominator_colname) := sum(get(values_colname), na.rm = TRUE),
       by = c(admin_colname, year_colname)]

  } else {
    # default approach (WHO): 12-month forward-looking sliding sum (left-aligned)
    denominator_colname <- paste(indicator, "SUM", 12, "MTH", "FW", sep = "_")

    output_dt[, (denominator_colname) := frollsum(get(values_colname),
                                           n = 12,
                                           align = "left",
                                           na.rm = TRUE),
       by = admin_colname]
  }

  # numerator and proportions
  for (n in vector_of_durations) {
    numerator_colname  <- paste(indicator, "SUM", n, "MTH", "FW", sep = "_")
    prop_name <- paste(indicator, n, "MTH", "ROW", "PROP", sep = "_")
    seasonality_colname <- paste(indicator, n, "MTH", "ROW", "SEASONALITY", sep = "_")

    # numerator: sliding window sum of next n months
    output_dt[, (numerator_colname) := frollsum(get(values_colname),
                                         n = n,
                                         align = "left",
                                         na.rm = TRUE),
       by = admin_colname]

    # proportion: numerator / denominator
    output_dt[, (prop_name) :=
          fifelse(get(denominator_colname) > 0, get(numerator_colname) / get(denominator_colname), NA_real_)]

    # flag for seasonality
    output_dt[, (seasonality_colname) := as.integer(get(denominator_colname) > 0 &
                                             get(prop_name) >= proportion_threshold)]
  }

  # return the data
  return(output_dt)
}


#' Determine Whether Admin Units Are Seasonal
#'
#' Computes whether or not an admin unit is "seasonal", based on WHO
#' guidelines.
#'
#' @param input_dt Data frame or data.table. Input data, expected to contain the row-level
#'   seasonality columns produced by compute_month_seasonality().
#' @param indicator Character. Type of indicator; uppercased and used to match the row-level seasonality columns.
#' @param vector_of_durations Numeric vector. Block sizes (in months) to check.
#' @param admin_colname Character. Name of the administrative unit column to group by. Default: 'ADM2_ID'.
#' @param year_colname Character. Name of the year grouping column. Default: 'YEAR'.
#' @param month_colname Character. Name of the month grouping column. Default: 'MONTH'.
#' @param proportion_seasonal_years_threshold Numeric. Minimum proportion of seasonal years for the admin unit to qualify as seasonal. Default: 0.5.
#' @return data.table with, per admin unit and block size, the proportion of seasonal years and a
#'   dichotomous seasonal/non-seasonal flag.
#'
#' @export
process_seasonality <- function(input_dt, indicator, vector_of_durations, admin_colname = 'ADM2_ID', year_colname = 'YEAR', month_colname = 'MONTH', proportion_seasonal_years_threshold = 0.5){

  indicator <- toupper(indicator)

  # make an "empty" data.table, with only the admin units
  output_dt <- input_dt[, setNames(list(unique(get(admin_colname))), admin_colname)]

  for (num_months in vector_of_durations) {

    regex_pattern <- paste(toupper(indicator), num_months, "MTH_ROW_SEASONALITY$", sep = '_')

    row_seasonality_colname <- grep(regex_pattern, names(input_dt), value = TRUE)

    subset_dt <- input_dt[, .SD, .SDcols = c(admin_colname, year_colname, month_colname, row_seasonality_colname)]

    subset_dt <- subset_dt[!is.na(get(row_seasonality_colname)),]

    num_seasonal_years_colname = paste(toupper(indicator), num_months, "MTH_NUM_SEASONAL_YEARS", sep = '_')

    num_total_years_colname = paste(toupper(indicator), num_months, "MTH_NUM_TOTAL_YEARS", sep = '_')

    subset_dt <- subset_dt[, setNames(
      # list of new column values
      .(
        sum(get(row_seasonality_colname)), # sum of all rows where seasonality is 1
        uniqueN(get(year_colname)) # number of total years where the month in question appears for a given admin unit
      ), c( # vector of new column names
        num_seasonal_years_colname,
        num_total_years_colname
      )),
      by = .(get(admin_colname), get(month_colname))
    ]

    # retrieve column names (overwritten when summarizing/aggregating with "get")
    names(subset_dt)[1] <- admin_colname
    names(subset_dt)[2] <- month_colname

    # compute proportion of seasonal years for each month
    proportion_colname = paste('PROP', 'SEASONAL', toupper(indicator), num_months, 'MTH', sep = '_')
    seasonality_colname = paste('SEASONALITY', toupper(indicator), num_months, 'MTH', sep = '_')

    # aggregate by admin unit, to get the dichotonomous variable whether the admin unit is seasonal by this criterion
    subset_dt <- subset_dt[, (proportion_colname) := get(num_seasonal_years_colname) / get(num_total_years_colname),
                           by = .(get(admin_colname), get(month_colname))]
    subset_dt <- subset_dt[, (seasonality_colname) := ifelse(get(proportion_colname) >= proportion_seasonal_years_threshold, 1, 0),
                           by = .(get(admin_colname), get(month_colname))]

    # aggregate to keep only the admin unit and whether or not the seasonality is 1
    subset_dt <- subset_dt[
      order(get(admin_colname), -get(seasonality_colname)),
      .SD[1],
      .SDcols = c(proportion_colname, seasonality_colname), # possible to add the month_colname here, and filter all cases where seasonality is 1
      by = get(admin_colname)
    ]

    # retrieve column names (overwritten when summarizing/aggregating with "get")
    names(subset_dt)[1] <- admin_colname

    # merge with the output_dt
    output_dt <- merge.data.table(output_dt, subset_dt, by = admin_colname)

  }

  return(output_dt)
}


#' Retrieve the Minimum Seasonality Block Size
#'
#' Retrieves, per row, the minimum number of months which constitute a
#' seasonality block.
#'
#' @param input_dt data.table. Input data.
#' @param seasonality_column_pattern Character. Regex pattern used to select the seasonality columns.
#' @param vector_of_possible_month_block_sizes Numeric vector. Block sizes, in the same order as the matched seasonality columns.
#' @param seasonal_blocksize_colname Character. Name of the output column to create.
#' @param valid_value Numeric. Value in the seasonality columns which indicates seasonality. Default: 1.
#' @return data.table with the added blocksize column (NA where no block qualifies as seasonal).
#'
#' @export
compute_min_seasonality_block <- function(
  input_dt,
  seasonality_column_pattern,
  vector_of_possible_month_block_sizes,
  seasonal_blocksize_colname,
  valid_value = 1
){

  # column names which match pattern
  seasonality_cols <- grep(
    seasonality_column_pattern,
    names(input_dt),
    ignore.case = TRUE,
    value = TRUE
  )

  # validate block sizes with columns
  if (length(vector_of_possible_month_block_sizes) != length(seasonality_cols)) {
    stop("Input possible month block sizes should correspond to number of relevant columns.")
  }

  block_sizes <- as.integer(vector_of_possible_month_block_sizes)

  # rowwise compute the new column
  output_dt <- input_dt[, (seasonal_blocksize_colname) :=
    apply(.SD, 1, function(row) {

      # find block sizes corresponding to the target value
      valid_blocks <- block_sizes[row == valid_value]

      # change to NA if no seasonality
      if (length(valid_blocks) == 0) return(NA_integer_)

      # minimum block size
      return(min(valid_blocks))
    }),
    .SDcols = seasonality_cols
  ]

  return(output_dt)
}


#' Filter Groups with All Values Present
#'
#' Filters groups where all unique values of a column are present.
#'
#' @param input_dt Data frame or data.table. Input data.
#' @param upper_colname Character. Name of the grouping column.
#' @param lower_colname Character. Name of the column containing values to check for completeness.
#' @return data.table containing only groups where all unique values from the original lower_colname are present.
#'
#' @export
filter_complete_groups <- function(input_dt, upper_colname, lower_colname) {
  all_vals <- unique(input_dt[[lower_colname]])

  output_dt <- copy(as.data.table(input_dt))

  output_dt[, if (all(all_vals %in% get(lower_colname))) .SD, by = upper_colname]
}


#' Add a Chronologically Ordered Year-Month Column
#'
#' Makes a year_month column by combining year and month columns, converted
#' to a factor ordered chronologically.
#'
#' @param input_dt Data frame or data.table. Input data.
#' @param year_colname Character. Name of the year column.
#' @param month_colname Character. Name of the month column.
#' @return data.table with the new year_month factor column added.
#'
#' @export
add_year_month <- function(input_dt, year_colname, month_colname) {
  output_dt <- copy(as.data.table(input_dt))
  output_dt[, year_month := paste0(
    get(year_colname), "-", sprintf("%02d", get(month_colname))
  )]
  output_dt[, year_month := factor(
    year_month,
    levels = unique(year_month[order(get(year_colname), get(month_colname))])
  )]
  return(output_dt)
}


#' Convert a String to Title Case
#'
#' Converts a string of uppercase words separated by spaces to title case.
#'
#' @param text Character. String of uppercase words separated by spaces.
#' @return Character. The string with each word in title case.
#'
#' @export
to_title_case <- function(text) {
  words <- strsplit(text, " ")[[1]]
  titled <- paste(
    toupper(substring(words, 1, 1)),
    tolower(substring(words, 2)),
    sep = ""
  )
  paste(titled, collapse = " ")
}


#' Find the Group(s) with the Highest Summed Value
#'
#' Sums a target indicator by group to get the group(s) with the highest
#' total.
#'
#' @param dt data.table. Input data.
#' @param target_indicator_colname Character. Name of the column to sum.
#' @param grouping_colname Character. Name of the column to group by.
#' @return List with: top_groups, a vector of the top group name(s); top_value, their aggregated
#'   value; n_top, the count of (possibly tied) top groups.
#'
#' @export
get_top_summed_group <- function(dt, target_indicator_colname, grouping_colname) {
  agg <- dt[, .(total = sum(get(target_indicator_colname), na.rm = TRUE)), by = grouping_colname]
  max_val <- max(agg[["total"]])
  top_groups <- agg[total == max_val][[grouping_colname]]
  list(
    top_groups = top_groups,
    top_value  = max_val,
    n_top      = length(top_groups)
  )
}


#' Compute Forward-Looking Block Percentage of Annual Total
#'
#' Creates forward-looking month blocks summing values and divides them by
#' the annual (calendar year) sum of values.
#'
#' @param input_dt Data frame or data.table. Input data.
#' @param values_colname Character. Name of the indicator column on which the computations are made.
#' @param vector_of_durations Numeric vector. Number of months in a block (e.g. 3/4/5).
#' @param admin_colname Character. Name of the administrative unit column to group by. Default: 'ADM2_ID'.
#' @param year_colname Character. Name of the year grouping column. Default: 'YEAR'.
#' @param month_colname Character. Name of the month grouping column. Default: 'MONTH'.
#' @param percentage_threshold Numeric. Percentage of the annual total which needs to occur in a block, to qualify for seasonality. Default: 0.6.
#' @return data.table with the additional numerator, percentage, and attained-threshold columns.
#'
#' @export
compute_block_percentage <- function(input_dt, values_colname, vector_of_durations, admin_colname = 'ADM2_ID', year_colname = 'YEAR', month_colname = 'MONTH', percentage_threshold = 0.6) {

  dt <- copy(as.data.table(input_dt))

  # ensure correct order
  dt <- dt[order(get(admin_colname), get(year_colname), get(month_colname))]

  # denominator: annual (calendar year) sum
  denominator_colname <- toupper("sum_calendar_year")
  dt[
    ,
    (denominator_colname) := sum(get(values_colname), na.rm = TRUE),
    by = c(admin_colname, year_colname)
  ]

  # numerators for each of the durations (forward-looking)
  for (n in vector_of_durations) {
    numerator_colname  <- toupper(glue("sum_{n}_mth_fw"))
    pct_colname <- toupper(glue("pct_{n}_mth_row"))
    target_colname <- toupper(glue("attained_{percentage_threshold}_cases_{n}_mth"))

    dt[, (numerator_colname) := frollsum(get(values_colname),
                                n = n,
                                align = "left",
                                na.rm = TRUE),
       by = c(admin_colname)]

    dt[, (pct_colname) :=
          # make NA's where it would be division by zero
          fifelse(get(denominator_colname) > 0, get(numerator_colname)*100 / get(denominator_colname), NA_real_)]

    dt[, (target_colname) := as.integer(get(denominator_colname) > 0 &
                                   get(pct_colname) >= percentage_threshold)]
  }

  return(dt)
}


#' Filter Cycle Data to a Reference Month and Years
#'
#' Filters data on cycles, to only the relevant beginning month, keeping only
#' the useful columns for plotting/presentation.
#'
#' @param input_dt data.table. Input cycle data.
#' @param pattern_cycle_colnames Character. Regex pattern to select cycle columns.
#' @param id_colnames Character vector. Identifier columns to retain.
#' @param year_colname Character. Name of the year column.
#' @param reference_years_vector Numeric vector. Years to keep.
#' @param month_colname Character. Name of the month column.
#' @param target_month_num Integer. Month value to filter on. Default: 8.
#' @return data.table filtered to target_month_num and reference_years_vector,
#'   with id and matching cycle columns.
#'
#' @export
filter_cycles_data <- function(input_dt, pattern_cycle_colnames, id_colnames, year_colname, reference_years_vector, month_colname, target_month_num=8){
  output_dt <- input_dt[
    get(month_colname) == target_month_num,
    .SD,
    .SDcols=c(id_colnames, grep(pattern_cycle_colnames, names(input_dt), value=TRUE))
  ]

  output_dt[, (month_colname) := NULL]

  output_dt <- output_dt[
  get(year_colname) %in% reference_years_vector
]

  return(output_dt)
}


#' Reshape Wide Cycle Coverage Data to Long Format
#'
#' Transforms a wide-format data table containing cycle coverage columns into
#' a cleaned long-format table, extracts coverage percentages from column
#' names, and keeps the maximum percentage covered per group.
#'
#' @param input_dt Data frame or data.table. Wide-format data containing cycle coverage columns.
#' @param space_colname Character. Name of the spatial grouping column.
#' @param time_colname Character. Name of the temporal grouping column.
#' @return data.table in long format with columns: space_colname (spatial identifier),
#'   time_colname (temporal identifier), num_cycles (number of cycles, non-missing
#'   values only), and max_pct_covered (maximum extracted percentage per group).
#'
#' @export
prep_cycles_long <- function(input_dt, space_colname, time_colname){
  output_dt <- melt(
    data=input_dt,
    id.vars=c(space_colname, time_colname),
    variable.name="category",
    value.name="num_cycles"
  )

  # extract the number covered as integer, from the "category" column
  output_dt[,max_pct_covered:= as.integer(sub(".*?(\\d+).*", "\\1", category))]

  # remove the original "category" column
  output_dt[, category :=NULL]

  # remove rows with missing number of cycles
  output_dt <- output_dt[
    !is.na(num_cycles)
  ]

  # keep only the maximum number of cases covered, for each number of cycles
  output_dt <- output_dt[, .(max_pct_covered = max(max_pct_covered)), by=c(space_colname, time_colname, "num_cycles")]

  return(output_dt)
}


#' Fill Missing Values in Long-Format Cycle Data
#'
#' Fills missing values in long-format cycles data using last observation
#' carried forward within groups.
#'
#' @param input_dt Data frame or data.table. Long-format data containing spatial, temporal, ordering and value columns.
#' @param spatial_colname Character. Name of the spatial grouping column.
#' @param temporal_colname Character. Name of the temporal grouping column.
#' @param order_colname Character. Name of the column for within-group ordering (determines the sequence for LOCF filling).
#' @param value_colname Character. Name of the column containing values to be filled.
#' @return data.table with missing values in value_colname filled using last observation carried forward within groups.
#'
#' @export
fill_long_cycles_dt <- function(input_dt, spatial_colname, temporal_colname, order_colname, value_colname){
	output_dt <- copy(as.data.table(input_dt))

	# ensure correct order before filling
	setorderv(
	  output_dt,
	  c(spatial_colname, temporal_colname, order_colname),
	  c(1L, 1L, 1L)
	)

	output_dt[, (value_colname) := nafill(get(value_colname), type = "locf"), by = c(spatial_colname, temporal_colname)]

}


#%% DHS ------------------------------


#' Get the Filename of the Latest DHS Recode Version
#'
#' Searches a folder for files matching a given DHS recode and file type,
#' then returns the filename with the highest version number.
#'
#' @param data_folder_path Character. Path to the folder containing all of the DHS files (zips).
#' @param recode_name Character. Name of the recode: 'KR', 'BR', 'IR', 'HR', etc.
#' @param file_type Character. File type to be used (part of the name of the zip; generally 'SV'). Default: 'SV'.
#' @return Character. Name of the target file.
#'
#' @export
extract_latest_dhs_recode_filename <- function(data_folder_path, recode_name, file_type='SV'){

  # candidate_files <- dir(path = data_folder_path, pattern = glue("*{toupper(recode_name)}*"))
  candidate_files <- list.files(
    path = data_folder_path,
    pattern = glue(".*{recode_name}.*{file_type}\\.zip$"),  # e.g. 'PR' followed by anything, then by "SV", ending with '.zip'
    full.names = FALSE,
    ignore.case = TRUE
  )
  all_versions <- sapply(candidate_files, (function(x) as.numeric(gsub("\\D", "", x))) )
  latest_version <- max(all_versions)
  chosen_file <- grep(as.character(latest_version), candidate_files, value=TRUE)
  return(chosen_file)
}


#' Check if Two DHS Filenames Share the Same Version
#'
#' Extracts the numeric version/issue codes from two DHS filenames and
#' compares them.
#'
#' @param dhs_filename_a Character. First DHS filename.
#' @param dhs_filename_b Character. Second DHS filename.
#' @return Logical. TRUE if both the version and issue are the same.
#'
#' @export
check_dhs_same_version <- function(dhs_filename_a, dhs_filename_b) {

  a_number <- stri_extract_all_regex(dhs_filename_a, "\\d+")
  b_number <- stri_extract_all_regex(dhs_filename_b, "\\d+")

  return(as.integer(a_number) == as.integer(b_number))
}


#' Check if Two Columns Have Identical Unique Values
#'
#' Checks if two columns have exactly the same unique values.
#'
#' @param dt_a Data frame or data.table. First table.
#' @param merge_col_a Character. Column name in dt_a to compare.
#' @param dt_b Data frame or data.table. Second table.
#' @param merge_col_b Character. Column name in dt_b to compare.
#' @return Logical. TRUE if both columns have the same unique values, FALSE otherwise.
#'
#' @export
check_perfect_match <- function(dt_a, merge_col_a, dt_b, merge_col_b){
  values_a <- dt_a[[merge_col_a]]
  values_b <- dt_b[[merge_col_b]]
  values_only_a <- setdiff(values_a, values_b)
  values_only_b <- setdiff(values_b, values_a)
  return(
    ((length(values_only_a) == 0) & (length(values_only_b) == 0))
    )
}


#' Build an Admin Name/Code Table from DHS Labelled Data
#'
#' Makes a data.table with admin names and admin id columns for DHS data, for
#' easier matching with DHIS2 data.
#'
#' @param input_dhs_df Data frame. The DHS data.
#' @param original_admin_column Character. Name of the column holding the labelled vector of codes + labels for the admin units. Default: 'V024'.
#' @param new_admin_name_colname Character. Name to give the admin labels column. Default: 'DHS_ADM1_NAME'.
#' @param new_admin_code_colname Character. Name to give the admin codes column (used for merging later on). Default: 'DHS_ADM1_CODE'.
#' @return data.table with only the codes and the names of the admin units, for subsequent merging with the DHS full data.
#'
#' @export
make_dhs_admin_df <- function(input_dhs_df, original_admin_column="V024", new_admin_name_colname='DHS_ADM1_NAME', new_admin_code_colname='DHS_ADM1_CODE'){

  admin_labels <- attr(input_dhs_df[[original_admin_column]], "labels")
  admin_dt <- data.frame(
    names = names(admin_labels),
    ids = as.vector(admin_labels),
    row.names = NULL,
    stringsAsFactors = FALSE
  )
  setDT(admin_dt)
  setnames(admin_dt, c("names", "ids"), c(new_admin_name_colname, new_admin_code_colname))
  return(admin_dt)
}


#' Compute Under-Five Mortality for a DHS Admin1 Unit
#'
#' Uses chmort() from the DHS.rates library to compute the sample average,
#' lower/upper 95% CI for under-five (u5) mortality. Results tested against
#' the Burkina Faso 2021 DHS report. Relies on `end_date_col`, a variable naming the survey's
#' date-of-interview column, which must already be defined in the calling environment — it is
#' not a function argument.
#'
#' TODO: see about adding column names as params (case-insensitive).
#'
#' @param dhs_adm1_dt data.table. DHS individual-recode data containing only one region (adm1 unit).
#' @return data.table with the DHS adm1 id and u5 mortality (sample average, lower CI, upper CI).
#'
#' @export
make_dhs_adm1_u5mort_dt <- function(dhs_adm1_dt){
  adm1_id <- as.integer(unique(dhs_adm1_dt[["V024"]]))
  mort_dt <- as.data.table(
    chmort(
      dhs_adm1_dt,
      JK = "Yes",
      Strata = "V023",
      Cluster = "V021",
      Weight = "V005",
      Date_of_interview = end_date_col,
      Date_of_birth = "B3",
      Age_at_death = "B7",
      Period = 120
    ),
    keep.rownames = TRUE
  )

  u5mort_dt <- mort_dt[
    rn == "U5MR",
    .SD,
    .SDcols = c('R', 'LCI', 'UCI')
  ]
  u5mort_dt[, DHS_ADM1_CODE := adm1_id]

  # print(u5mort_dt)
  return(u5mort_dt)
}


#%% MISC FUNCTIONS


#' Find Values Present in One Column but Not the Other
#'
#' Compares the unique values of two columns from two df/dt.
#'
#' @param df1 Data frame or data.table. First table.
#' @param df1_colname Character. Column name in df1.
#' @param df2 Data frame or data.table. Second table.
#' @param df2_colname Character. Column name in df2.
#' @return Unique values found only in df1's column.
#'
#' @export
compare_values <- function(df1, df1_colname, df2, df2_colname) {

  # make both be data tables
  df1 <- as.data.table(df1)
  df2 <- as.data.table(df2)

  # sort the unique values from the respective columns
  df1_values <- sort(unique(df1[[df1_colname]]))
  df2_values <- sort(unique(df2[[df2_colname]]))

  # values only in df1
  values_only_df1 <- df1_values[!df1_values %in% df2_values]

  return(values_only_df1)
}


#' Find Column Combinations Present in One Table but Not the Other
#'
#' Compares unique combinations of columns between two df and returns those
#' in the first, but not in the second.
#'
#' @param df1 Data frame or data.table. First table.
#' @param df1_colnames Character vector. Column names in df1.
#' @param df2 Data frame or data.table. Second table.
#' @param df2_colnames Character vector. Column names in df2, in the same order as df1_colnames.
#' @return data.table of unique column-value combinations found only in df1.
#'
#' @export
compare_combinations <- function(df1, df1_colnames, df2, df2_colnames) {

  # make both be data tables
  df1 <- as.data.table(df1)
  df2 <- as.data.table(df2)

  # extract unique combinations of the specified columns
  df1_combos <- unique(df1[, ..df1_colnames])
  df2_combos <- unique(df2[, ..df2_colnames])

  # align column names so fsetdiff can compare them
  setnames(df2_combos, old = df2_colnames, new = df1_colnames)

  # return combinations only in df1
  values_only_df1 <- fsetdiff(df1_combos, df2_combos)
  return(values_only_df1)
}


#' Aggregate Geometries by Admin Unit
#'
#' Aggregates the geometries of sf data, at a specified level, given by id
#' and name columns.
#'
#' @param sf_data sf object. Input spatial data.
#' @param admin_id_colname Character. Name of the column containing the admin unit ids.
#' @param admin_name_colname Character. Name of the column containing the admin unit names.
#' @return sf object with the aggregated geometries.
#'
#' @export
aggregate_geometry <- function(sf_data, admin_id_colname, admin_name_colname) {
  by_list <- list(
    sf_data[[admin_id_colname]],
    sf_data[[admin_name_colname]]
  )
  names(by_list) <- c(admin_id_colname, admin_name_colname)

  result <- aggregate(sf_data["geometry"], by = by_list, FUN = sf::st_union)
  return(result)
}


#' Delete Files Not Matching a Given Extension
#'
#' Deletes files which don't have a given extension, from a given folder.
#'
#' @param folder_path Character. Directory path.
#' @param extension_to_retain Character. Extension of the files to keep. Default: '.zip'.
#' @return Invisible NULL. Called for its side effects.
#'
#' @export
delete_otherextension_files <- function(folder_path, extension_to_retain=".zip"){
  pattern_to_keep <- paste0("*", extension_to_retain)
  non_delete_files <- dir(path = folder_path, pattern = pattern_to_keep, ignore.case=TRUE)
  delete_files <- setdiff(dir(path = folder_path), non_delete_files)
  if (length(delete_files) == 0){
    print("No files to delete.")
  } else{
    if(length(non_delete_files) == 0){
      print("Deleting all files from folder.")
    }
    unlink(file.path(folder_path, delete_files), recursive=TRUE)
  }
}


#' Clean Administrative Unit Names
#'
#' Cleans the admin names of certain countries' pyramids, by removing
#' string_to_remove if present (these are usually "Province" or "Zone de
#' santé" or "District") and then removing the prefix (some countries have
#' one).
#'
#' @param input_vector Character vector. Admin names to clean.
#' @param string_to_remove Character. Substring to delete from the admin names, if present (e.g. an admin unit type indicator). Default: 'province'.
#' @return Character vector. The cleaned admin names.
#'
#' @export
clean_admin_names <- function(input_vector, string_to_remove='province') {
  sapply(input_vector, function(input_string) {
    parts <- strsplit(input_string, " ")[[1]]
    parts <- parts[toupper(parts) != toupper(string_to_remove)]
    if (length(parts) > 1) {
      parts_without_prefix <- parts[-1]
      output_string <- paste(parts_without_prefix, collapse = " ")
    } else {
      output_string <- ""  # if input has <=2 words
    }
    return(output_string)
  }, USE.NAMES = FALSE)
}


#' List Files Matching Suffix, Extension, and Name Pattern
#'
#' Filters files in a directory by suffix, extension, and a required
#' substring in the filename, printing the matches and stopping if none are
#' found.
#'
#' @param target_path Character. Directory to search.
#' @param vector_of_file_suffixes Character vector. Allowed filename suffixes. Default: c('wide', 'long').
#' @param vector_of_extensions Character vector. Allowed file extensions. Default: c('.csv', '.parquet').
#' @param must_contain_string Character. Substring that must appear in the filename. Default: "".
#' @return Invisible NULL. Prints the list of matching files as a side effect
#'   (stops if none are found).
#'
#' @export
filter_files_to_save <- function(
    target_path,
    vector_of_file_suffixes = c('wide', 'long'),
    vector_of_extensions = c('.csv', '.parquet'),
    must_contain_string = ""){
    # build pattern to check
    pattern <- paste0("(", paste0(vector_of_file_suffixes, collapse = "|"), ")",
                      "(", paste0("\\", vector_of_extensions, collapse = "|"), ")$")

    # list matching files
    target_files <- list.files(path = target_path, pattern = pattern, full.names = TRUE)

    # further filter by the string which must appear in the filename
    target_files <- target_files[grepl(must_contain_string, basename(target_files))]

    # check if there are any files which match the pattern
    if (length(target_files) == 0) {
        stop("No files found in directory: ", target_path, " matching the conditions.")
    }

    # print files which match the pattern
    print("Files found:")
    print(target_files)
}


#%% Healthcare access -------------------------


#' Load the Default DHIS2 FOSA Dataset
#'
#' Logs the reason for falling back, then loads the pyramid parquet file for
#' the given country from the default DHIS2 dataset.
#'
#' @param helper_dhis2_dataset Character. Identifier of the DHIS2 dataset.
#' @param helper_country_code Character. Country code used to build the filename.
#' @param reason Character. Reason for falling back to the default dataset, used in the log message.
#' @return Data frame. The loaded default FOSA/pyramid data.
#'
#' @export
load_default_dataset <- function(helper_dhis2_dataset, helper_country_code, reason) {
    log_msg(glue::glue("{reason}: using default DHIS2 FOSA dataset. To use input data, please input a different file and rerun pipeline."))
    dhis2_data <- tryCatch(
        {
            get_latest_dataset_file_in_memory(
                helper_dhis2_dataset,
                glue::glue("{helper_country_code}_pyramid.parquet")
            )
            # setDT(dhis2_data)
        },
        error = function(e) {
            msg <- paste("Error loading DHIS2 FOSA default data:", conditionMessage(e))
            stop(msg)
        }
    )
    return(dhis2_data)
}


#' Import and Validate FOSA Location Data
#'
#' Validates a user-supplied FOSA CSV file (existence, extension, required
#' latitude/longitude columns, readability, and column types), falling back
#' to the default DHIS2 pyramid dataset whenever a check fails.
#'
#' @param input_file_path Character. Path to the candidate input CSV file.
#' @param pipeline_dhis2_dataset Character. Identifier of the fallback DHIS2 dataset.
#' @param pipeline_country_code Character. Country code used for the fallback filename.
#' @param latitude_colname Character. Expected name of the latitude column. Default: 'LATITUDE'.
#' @param longitude_colname Character. Expected name of the longitude column. Default: 'LONGITUDE'.
#' @return Data frame with the validated FOSA data, or the default dataset if validation fails.
#'
#' @export
import_fosa_data <- function(
    input_file_path,
    pipeline_dhis2_dataset,
    pipeline_country_code,
    latitude_colname="LATITUDE",
    longitude_colname="LONGITUDE"
){
    # helper to load fallback dataset
    load_default_dataset <- function(helper_dhis2_dataset, helper_country_code, reason) {
        log_msg(glue::glue("{reason}: using default DHIS2 FOSA dataset. To use input data, please input a different file and rerun pipeline."))
        dhis2_data <- tryCatch(
            {
                get_latest_dataset_file_in_memory(
                    helper_dhis2_dataset,
                    glue::glue("{helper_country_code}_pyramid.parquet")
                )
                # setDT(dhis2_data)
            },
            error = function(e) {
                msg <- paste("Error loading DHIS2 FOSA default data:", conditionMessage(e))
                stop(msg)
            }
        )
        return(dhis2_data)
    }

    # check if the file exists
    condition_existence <- !is.null(input_file_path) && file.exists(input_file_path)
    if(!condition_existence){return(load_default_dataset(pipeline_dhis2_dataset, pipeline_country_code, "No valid input FOSA data file supplied"))}

    # check if the file is of .csv type
    condition_extension <- grepl("\\.csv$", input_file_path, ignore.case = TRUE)
    if(!condition_extension){return(load_default_dataset(pipeline_dhis2_dataset, pipeline_country_code, "The input FOSA data is not a .csv file"))}

    # check if the file has the necessary column names
    test_input_df <- tryCatch(
        data.table::fread(input_file_path, nrows = 0),
        error = function(e) {
            # message("Error reading input FOSA file: ", e$message)
            return(NULL)
        }
    )
    if (is.null(test_input_df)){return(load_default_dataset(pipeline_dhis2_dataset, pipeline_country_code, "Unable to read input file columns"))}

    # check if the file contains the necessary column names
    input_latitude_cols <- grep(glue::glue("^{latitude_colname}$"), names(test_input_df), ignore.case = TRUE, value = TRUE)
    input_longitude_cols <- grep(glue::glue("^{longitude_colname}$"), names(test_input_df), ignore.case = TRUE, value = TRUE)

    condition_latitude_name <- length(input_latitude_cols) == 1
    condition_longitude_name <- length(input_longitude_cols) == 1

    if(!(condition_latitude_name && condition_longitude_name)){
        return(load_default_dataset(pipeline_dhis2_dataset, pipeline_country_code, "The input FOSA data does not contain the 'LATITUDE' and 'LONGITUDE' columns"))
    }

    # check if the file is fully readable
    input_df <- tryCatch(
        read.csv(input_file_path, header = TRUE, na.strings = c("NA", "")),
        error = function(e){return(NULL)}
    )
    if(is.null(input_df)){
        return(load_default_dataset(pipeline_dhis2_dataset, pipeline_country_code, "The input FOSA file is corrupt"))
    }

    # check if the necessary columns are of the right type
    condition_latitude_type <- is.numeric(input_df[[input_latitude_cols]])
    condition_longitude_type <- is.numeric(input_df[[input_longitude_cols]])

    if(!condition_latitude_type || !condition_longitude_type){
        return(load_default_dataset(pipeline_dhis2_dataset, pipeline_country_code, "The input FOSA data file's 'LATITUDE' and/or 'LONGITUDE' columns are not valid numeric"))
    }

    log_msg("Input FOSA data validated and loaded successfully.")
    return(input_df)
}


#' Reproject an sf or terra Vector to a Target CRS
#'
#' Reprojects an sf or terra SpatVector object to the given EPSG code,
#' skipping the operation if it is already in the target CRS.
#'
#' @param x sf, sfc, or SpatVector object. Object to reproject.
#' @param epsg_value Integer. EPSG code to reproject to.
#' @return The input object, reprojected to the target CRS if needed.
#'
#' @export
reproject_epsg <- function(x, epsg_value) {

  # check if input is sf
  if (inherits(x, "sf") || inherits(x, "sfc")) {
    current_epsg <- sf::st_crs(x)$epsg
    target_epsg  <- epsg_value

    if (is.na(current_epsg) || current_epsg != target_epsg) {
      print(glue::glue("Info: reprojecting sf object to EPSG:{target_epsg}."))
      x <- sf::st_transform(x, target_epsg)
    } else {
      print("Info: no reprojection needed for sf object.")
    }

  # check if input is terra vector
  } else if (inherits(x, "SpatVector")) {
    current_crs <- terra::crs(x, describe = TRUE)$code
    target_crs  <- epsg_value

    if (is.na(current_crs) || current_crs != target_crs) {
      print(glue::glue("Info: reprojecting terra vector to EPSG:{target_crs}."))
      x <- terra::project(x, paste0("EPSG:", target_crs))
    } else {
      print("Info: no reprojection needed for terra vector.")
    }

  } else {
    stop("Input must be an sf or terra vector object.")
  }

  return(x)
}


#' Filter Points Within Polygon Boundaries
#'
#' Filters points within polygon boundaries using terra.
#'
#' @param locations_vect SpatVector. Point geometries.
#' @param boundaries_vect SpatVector. Polygon geometries.
#' @param epsg_value_degrees Integer. EPSG code for the geographic (degree-based) CRS (e.g. for Burkina Faso, 4326).
#' @return SpatVector with only the points within the boundaries.
#'
#' @export
filter_points_within_boundaries <- function(locations_vect, boundaries_vect, epsg_value_degrees) {

  print("Input data 1/2 (point locations):")
  locations_vect <- reproject_epsg(locations_vect, epsg_value_degrees)
  print("Input data 2/2 (boundaries polygon):")
  boundaries_vect <- reproject_epsg(boundaries_vect, epsg_value_degrees)

  # spatial relation: keep only points within polygons
  within_matrix <- relate(locations_vect, boundaries_vect, relation = "within")

  # get indices of points with at least one 'within' relation
  # point_indices_within <- which(lengths(within_matrix) > 0)
  point_indices_within <- which(within_matrix)

  point_indices_outside <- which(!within_matrix)

  print(glue("There were {length(point_indices_within)} points within the boundaries, and {length(point_indices_outside)} points outside. Only those within are returned."))

  # subset the points
  filtered_locations_vect <- locations_vect[point_indices_within, ]

  return(filtered_locations_vect)
}


#' Get or Create an OpenHEXA Dataset by Slug
#'
#' Checks if a dataset with the given slug exists in the given workspace. If
#' not, creates it; if yes, retrieves it. Also ensures the dataset has at
#' least one version.
#'
#' @param target_workspace Workspace object. Workspace to search/create in.
#' @param target_dataset_slug Character. Identifier of the dataset to create/search.
#' @param target_dataset_name Character. Name of the new dataset to create (if necessary).
#' @param target_dataset_description Character. Description of the new dataset to create (if necessary).
#' @return The new or existing dataset which matches the slug.
#'
#' @export
check_or_create_dataset <- function(
    target_workspace,
    target_dataset_slug,
    target_dataset_name,
    target_dataset_description) {

  # fetch existing datasets
  existing_datasets <- target_workspace$list_datasets()

  # check if the dataset already exists
  output_dataset <- NULL
  matching_datasets <- Filter(function(x) x$slug == target_dataset_slug, existing_datasets)

  if (length(matching_datasets) > 0) {
    output_dataset <- matching_datasets[[1]]
    message(paste("Dataset already exists:", output_dataset$slug))
  } else {
    # create a new dataset
    output_dataset <- target_workspace$create_dataset(
      name = target_dataset_name,
      description = target_dataset_description
    )
    message(paste("Created new dataset:", output_dataset$slug))
  }

  # check if the dataset has any versions
  if (is.null(output_dataset$latest_version)) {
    message("Dataset has no versions. Creating initial version 'v0'...")
    initial_version <- output_dataset$create_version("v0")
    message(paste("Created version:", initial_version$name))
  } else {
    message(paste("Dataset already has versions. Latest version:", output_dataset$latest_version$name))
  }

  return(output_dataset)

}


#' Create the Next Incremented Dataset Version
#'
#' Reads the dataset's latest version name, increments the latest version
#' number, and creates a new version with that name.
#'
#' @param target_dataset Dataset object. Dataset to make the new version in; must already have an initial version.
#' @return New dataset version object, created with an incremented version number.
#'
#' @export
make_new_dataset_version <- function(target_dataset){
    # get latest dataset version name
    latest_version_name <- target_dataset$latest_version$name
    latest_version_num <- as.numeric(gsub("v", "", latest_version_name))

    # create a new version, which increments the current version by 1
    new_version <- target_dataset$create_version(paste0("v", latest_version_num + 1))

    return(new_version)
}


#' Reassign Child Facilities to a New Administrative Level
#'
#' Helper function for NER child-parent updates. Updates child facilities
#' dynamically for any level -> moves "level" to "target_level", updating
#' parent references.
#'
#' @param new_level_table Data frame. Table mapping old parent ids to new level ids/names.
#' @param group_table Data frame. Table of facilities to update.
#' @param level Integer. Current level of the facilities to move (must be at least 2).
#' @param target_level Integer. Level to move the facilities to.
#' @param parent_level Integer. Level used to identify parent groups.
#' @return Data frame of updated child facilities with level, id, name, and parent columns reassigned.
#'
#' @export
get_updated_children <- function(new_level_table, group_table, level, target_level, parent_level) {

    if (level < 2) stop(glue("level must be at least 2, received: {level}"))

    # Determine column names dynamically
    level_id_col <- paste0("level_", level, "_id")
    level_name_col <- paste0("level_", level, "_name")
    target_level_id <- glue("level_{target_level}_id")
    target_level_name <- glue("level_{target_level}_name")
    parent_level_id <- glue("level_{parent_level}_id")
    parent_level_name <- glue("level_{parent_level}_name")

    parent_level_ids <- unique(group_table[[parent_level_id]])
    child_updated <- group_table[0, ]

    for (parent_id in parent_level_ids) {
        # Find the old parent in level 6 (the target_level_id in new_level_table corresponds to the old parent level)
        old_parent <- head(new_level_table[new_level_table[[target_level_id]] == parent_id, ], 1)

        if (nrow(old_parent) > 0) {
            # Select child facilities from the table
            child_selection <- group_table[group_table[[parent_level_id]] == parent_id, ]

            if (nrow(child_selection) > 0) {
                print(glue("Fixing child facilities under: {old_parent$name}"))

                # Update columns dynamically
                child_selection$level <- target_level
                child_selection[[target_level_id]] <- child_selection[[level_id_col]]
                child_selection[[target_level_name]] <- child_selection[[level_name_col]]

                # Update parent references
                child_selection[[parent_level_id]] <- old_parent[[parent_level_id]]
                child_selection[[parent_level_name]] <- old_parent[[parent_level_name]]

                # Reset child level columns
                child_selection[[level_id_col]] <- NA
                child_selection[[level_name_col]] <- NA

                # Append
                child_updated <- rbind(child_updated, child_selection)
            }
        }
    }
    return(child_updated)
}