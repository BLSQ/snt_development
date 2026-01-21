# ================================================
# Title: Utility Functions for SNT Process
# Description: This script contains utility functions used for SNT computation workflow.
# Author: Esteban Montandon
# Created: [2024-10-01]
# Last updated: [2025-08-26]
# Dependencies: stringi, httr, arrow, tools, jsonlite
# Notes:
#   - [Optional: Any special considerations, references, or tips]
# ================================================
                           
# add any other matching logic here
format_names <- function(x) {
    x <- stri_trans_general(str = x, id = "Latin-ASCII") # remove weird characters
    x <- gsub("[^a-zA-Z0-9]", " ", toupper(x))           # replace non-alphanum with space
    # x <- gsub("(?i)PROVINCE|ZONE DE SANTE|AIRE DE SANTE|CENTRE DE SANTE", "", x) # TEMPORARY SKIP
    x <- gsub("  +", " ", x)       # collapse multiple spaces
    trimws(x)
}

                           
# Clean column names formatting                         
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

                           
# Function to check if packages are installed -> install missing packages
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



# # Load a file from the last version of a dataset (last version of the dataset)            
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
    file_extension <- tools::file_ext(filename)
    
    if (file_extension == "parquet") {
        df <- arrow::read_parquet(temp_file)
    } else if (file_extension == "csv") {        
        df <- utils::read.csv(temp_file, stringsAsFactors = FALSE)
    } else if (file_extension == "geojson") {
        tmp_geojson <- tempfile(fileext = ".geojson")
        writeBin(raw_content, tmp_geojson)
        df <- sf::st_read(tmp_geojson, quiet = TRUE)        
    }
    else {
      stop(paste("Unsupported file type:", file_extension))
    }
    
    # Return the dataframe
    return(df)
}


# helper function for OpenHEXA logging
log_msg <- function(msg , level="info") {
    print(msg)
    if (!is.null(openhexa$current_run)) {
            level <- tolower(level)
            if (level == "info"){
                openhexa$current_run$log_info(msg)    
            } else if(level == "warning"){
                openhexa$current_run$log_warning(msg)    
            } else if(level == "error"){
                openhexa$current_run$log_error(msg)    
            } else {
                stop("Unsupported log level")
            }
        }
}


# Helper function for exporting data (csv and parquet files) 
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


# Helper for quick data table summary
printdim <- function(df, name = deparse(substitute(df))) {
  cat("Dimensions of", name, ":", nrow(df), "rows x", ncol(df), "columns\n\n")
}

    

#%% SEASONALITY -------------------------------------------------------------------
                                                 
#############
convert_columns <- function(dt, col_type_map) {
    
  # convert specified columns of a data.table to target types
  # @param dt a data.table to modify in place
  # @param col_type_map a named list mapping type names (e.g., "numeric") to vectors of column names
  # @returns the modified data.table
    
  for (type in names(col_type_map)) {
    cols <- col_type_map[[type]]
    convert_fun <- match.fun(paste0("as.", type))
    dt[, (cols) := lapply(.SD, convert_fun), .SDcols = cols]
  }
}
                                                 
#############
get_newest_dataset_file <- function(target_dataset, target_filename, target_country_code){
  
  #' Retrieve the latest version from a dataset
  #' @param target_dataset the dataset which contains the file
  #' @param target_filename the file to retrieve
  #' @param target_country_code the code name of the country to retrieve
  #' @returns the dataframe of the latest version on the dataset, for the target country
  
  # load file
  output_data <- tryCatch({ get_latest_dataset_file_in_memory(target_dataset, target_filename) }, 
                          error = function(e) {
                            msg <- glue("Error while loading {ountry_code} {target_filename}, {conditionMessage(e)}")  # log error message
                            cat(msg)
                            stop(msg)
                          })
  data_dimensions <- paste(dim(output_data), collapse=", ")
  
  msg <- glue("{target_filename} loaded from dataset {target_dataset}; the dataframe has the following dimensions: {data_dimensions}")
  log_msg(msg)
  
  return(output_data)
}
                                                 
#############
match_column_classes <- function(input_dt, reference_dt) {
  #' ensure that columns in an input data.table have the same classes as those in a reference data.table
  #' @param input_dt the data table whose columns will be adapted
  #' @param reference_dt the data table with the target column classes
  #' @return a data table with the same column names and data as input_dt, but with classes of common columns made to match those in reference_dt
  
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

#############
make_cartesian_admin_period <- function(input_dt, admin_colname, year_colname, month_colname) {
  #' make a place-time cartesian product, to ensure each possible combination exists, between a minimum period and a maximum period
  #' @param input_dt the original data.table
  #' @param admin_colname place (admin) column
  #' @param year_colname time column 1
  #' @param month_colname time column 2
  #' @return the total number of periods (to check if the data contains enough periods)
  #' @return a new data.table, with the cartesian place-time rows
  
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

#############
make_cartesian_dt_vector <- function(input_dt, input_vector, new_colname){
  #' cartesian product of a data table and a vector
  #' @param input_dt data
  #' @param input_vector extra vector to add rows based on
  #' @param new_colname name of the new column (the one which has the values of the vector)
  #' @returns data table with the cartesian product of the two parameters
  
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

#############
make_full_time_space_data <- function(input_dt, full_rows_dt, target_colname, admin_colname = 'ADM2_ID', year_colname = 'YEAR', month_colname = 'MONTH') {
  
  #' add missing administrative rows, from a time-place cartesian dataset, into a data table "with holes"
  #'
  #' @param input_dt input data
  #' @param full_rows_dt full set of rows to merge with
  #' @param target_colname the column which will eventually need to be imputed (which will have holes)
  #' @param admin_colname the admin unit id
  #' @param year_colname year
  #' @param month_colname month
  #' @returns data table with merged and imputed administrative unit columns
  
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

#############
extract_dt_with_missings <- function(input_dt, target_colname, id_colname){
  #' extract the id's of all rows which have missing values on target_colname
  #' @param input_dt the input data.table
  #' @param target_colname the column where missings should be identified
  #' @param id_colname the grouping column (units of observation)
  #' @return a new data.table, which filters only those id's and returns all of the observations associated with them
  
  ids_with_missings <- input_dt[is.na(get(target_colname)), unique(get(id_colname))]
  dt_with_missings <- input_dt[get(id_colname) %in% ids_with_missings]
  return(dt_with_missings)
}

#############
fill_missing_cases_ts <- function(district_data, original_values_colname, estimated_values_colname, admin_colname, period_colname, threshold_for_missing = 0.0){
  #' SARIMA-based imputation for the values_colname variable of values_colname (generally confirmed malaria cases):
  #'    - fits a seasonal ARIMA model on the values_colname variable
  #'    - generates estimations for the missing values
  #' @param district_data a data.table with the values of a specific admin unit
  #' @param original_values_colname the name of the column which contains the values to be imputed
  #' @param estimated_values_colname the name of the column which will contain the new, imputed values
  #' @param admin_colname the name of the column which contains the administrative unit ids
  #' @param period_colname the name of the column which contains the year-month periods
  #' @param threshold_for_missing a threshold below which values_colname is considered as missing for the imputation purposes; default is 0 and should ideally stay like that
  #' @return a new data.table, with the filled column, called <values_colname>'_EST' added
  
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

#############
compute_month_seasonality <- function(input_dt, indicator, values_colname, vector_of_durations, admin_colname = 'ADM2_ID', year_colname = 'YEAR', month_colname = 'MONTH', proportion_threshold = 0.6) {
  #' create forward-looking month blocks summing values based on the WHO month-block reasoning for seasonality computation - allows for different block sizes
  #' @param input_dt an input data table (or data frame)
  #' @param indicator a string to specify the type of indicator (case/rainfall/etc. - will be added to the output variable name)
  #' @param values_colname the indicator column, on which the computations are made
  #' @param vector_of_durations the vector with the number of months in a block (3/4/5)
  #' @param admin_colname the administrative units to group 
  #' @param year_colname year grouping column
  #' @param montn_colname month grouping column
  #' @param proportion_threshold the proportion of indicator which needs to occur in a block, to qualify for seasonality
  #' @return an output data table with the additional column

  indicator <- toupper(indicator)
  dt <- copy(as.data.table(input_dt))
  
  # ensure correct order
  dt <- dt[order(get(admin_colname), get(year_colname), get(month_colname))]
  
  # denominator: 12-month forward-looking sliding sum (left-aligned)
  denominator_colname <- paste(indicator, "SUM", 12, "MTH", "FW", sep = "_")
  dt[, (denominator_colname) := frollsum(get(values_colname),
                                n = 12,
                                align = "left",
                                na.rm = TRUE),
     by = admin_colname]
  
  # numerators for each of the durations (forward-looking)
  for (n in vector_of_durations) {
    numerator_colname  <- paste(indicator, "SUM", n, "MTH", "FW", sep = "_")
    prop_name <- paste(indicator, n, "MTH", "ROW", "PROP", sep = "_")
    seasonality_colname <- paste(indicator, n, "MTH", "ROW", "SEASONALITY", sep = "_")
    
    dt[, (numerator_colname) := frollsum(get(values_colname),
                                n = n,
                                align = "left",
                                na.rm = TRUE),
       by = admin_colname]
    
    dt[, (prop_name) := 
          # make NA's where it would be division by zero
          fifelse(get(denominator_colname) > 0, get(numerator_colname) / get(denominator_colname), NA_real_)]
    
    dt[, (seasonality_colname) := as.integer(get(denominator_colname) > 0 &
                                   get(prop_name) >= proportion_threshold)]
  }
  
  # return the data
  dt[]
}

#############
process_seasonality <- function(input_dt, indicator, vector_of_durations, admin_colname = 'ADM2_ID', year_colname = 'YEAR', month_colname = 'MONTH', proportion_seasonal_years_threshold = 0.5){
  #' compute whether or not an admin unit is "seasonal", based on WHO guidelines
  #' TODO: I'm passing all columns as arguments (needs resetting column names after each summarizing; to see if the data may have different column names; if not, no need to pass these as arguments and would make the code lighter)
  #' @param input_dt the input data table/frame
  #' @param indicator the type of indicator
  #' @param vector_of_durations the block sizes to check
  #' @param admin_colname the place column
  #' @param year_colname the year grouping column
  #' @param month_colname the month grouping column
  #' @param proportion_seasonal_years_threshold the minimum number of seasonal years, for the admin unit to qualify as seasonal
  #' @return the output data table, with extra dichotomous variables (seasonal/non-seasonal for each size of month-blocks)
  
  
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

#############
compute_min_seasonality_block <- function(
  input_dt,
  seasonality_column_pattern,
  vector_of_possible_month_block_sizes,
  seasonal_blocksize_colname,
  valid_value = 1
){
  #' retrieve the minimum number of months which constitute a seasonality block
  #' @param input_dt input data.table
  #' @param seasonality_column_pattern pattern for seasonality columns
  #' @param vector_of_possible_month_block_sizes numeric vector of block sizes
  #' @param seasonal_blocksize_colname name of output column
  #' @param valid_value value indicating seasonality
  #' @return data.table with added blocksize column (NA if none)
  
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

#############
make_seasonality_plot <- function(spatial_seasonality_df, seasonality_colname, title_label){
  #' map seasonality with predefined colors
  #' areas are categorized as "Seasonal" or "Not seasonal"
  #'
  #' @param spatial_seasonality_df sf data frame with spatial geometry and seasonality data
  #' @param seasonality_colname string with the name of the column indicating seasonality (values should be 0 or 1)
  #' @param title_label string to customize the legend title
  #'
  #' @return a ggplot object of the seasonality map
  seasonality_plot <- ggplot(spatial_seasonality_df) +
    geom_sf(aes(fill = as.factor(get(seasonality_colname))))+
    scale_fill_manual(values = c("1" = "chartreuse2", "0" = "#1E2044"),
                      labels = c("1" = "Seasonal", "0" = "Not seasonal")) +  # Custom labels
    coord_sf() + # map projection
    # guides(fill=guide_legend(title= paste0("Seasonality (", title_label, ")"), nrow = 2)) +
    guides(fill=guide_legend(title=title_label, nrow = 2)) +
    theme_classic() +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          legend.position = "bottom", legend.key.width = unit(2,"cm"), legend.text=element_text(size=10))
  
  print(seasonality_plot)
  
  return(seasonality_plot)
}

#############
make_seasonality_duration_plot <- function(spatial_seasonality_df, seasonality_duration_colname, title_label, palette_name = 'BrBG', none_label="No seasonality"){
  #' map the duration of seasonality (in how many months x% of annual rain falls)
  #'
  #' @param spatial_seasonality_df sf data with spatial and seasonality columns
  #' @param seasonality_duration_colname column name (string) for seasonality duration (number of months)
  #' @param title_label string for the legend title
  #' @param palette_name colorbrewer palette (default is 'BrBG')
  #' @param none_label legend label when there is no seasonality (defaults to "Not seasonal")
  #' 
  #' @return ggplot object
  #' 
  duration_plot <- ggplot(spatial_seasonality_df) +
    geom_sf(aes(fill = as.character(get(seasonality_duration_colname))))+
    coord_sf() + # map projection
    # scale_fill_discrete(
    #   # values = sort(unique(as.character(plot_df[['seasonality_duration_colname']]))),  # custom colors
    #   labels = function(x) {
    #     ifelse(x == "Inf", none_label, x) # custom labels
    #     }
    #   )    +
    # 
    scale_fill_brewer(palette = palette_name, labels = function(x) {
      ifelse(is.na(x), none_label, x) # custom labels
    }
    ) +
    # guides(fill=guide_legend(title= paste0("Number of months (", title_label, ")" ), nrow = 2)) +
    guides(fill=guide_legend(title=title_label, nrow = 2)) +
    theme_classic() +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          legend.position = "bottom", legend.key.width = unit(2,"cm"), legend.text=element_text(size=10))
  
  print(duration_plot)
  return(duration_plot)
}


#%% DHS ------------------------------

######################################
extract_latest_dhs_recode_filename <- function(data_folder_path, recode_name, file_type='SV'){
  #' Get the file name of the most recent version of a specific DHS recode
  #' @param data_folder_path the path to all of the DHS files (zips)
  #' @param recode_name the name of the recode: 'KR', 'BR', 'IR', 'HR', 'IR', etc.
  #' @param file_type the file type to be used (part of the name of the zip; generally extracting 'SV')
  #' @returns the name of the target file
  
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

###################################
check_dhs_same_version <- function(dhs_filename_a, dhs_filename_b) {
  #' check if two DHS filenames have the same version and issue numbers
  #' @param dhs_filename_a
  #' @param dhs_filename_b
  #' @returns True if both the version and issue are the same
  #' 
  
  a_number <- stri_extract_all_regex(dhs_filename_a, "\\d+")
  b_number <- stri_extract_all_regex(dhs_filename_b, "\\d+")
  
  return(as.integer(a_number) == as.integer(b_number))
}

##################################
check_perfect_match <- function(dt_a, merge_col_a, dt_b, merge_col_b){
  #' check if two columns have exactly the same unique values
  #' 
  #' @param dt_a first data frame
  #' @param merge_col_a column name (string) in dt_a to compare
  #' @param dt_b second data frame
  #' @param merge_col_b column name (string) in dt_b to compare
  #' 
  #' @return TRUE if both columns have the same unique values, FALSE otherwise
  values_a <- dt_a[[merge_col_a]]
  values_b <- dt_b[[merge_col_b]]
  values_only_a <- setdiff(values_a, values_b)
  values_only_b <- setdiff(values_b, values_a)
  return(
    ((length(values_only_a) == 0) & (length(values_only_b) == 0))
    )
}
                                           
#######################################
delete_otherextension_files <- function(folder_path, extension_to_retain=".zip"){
  #' Delete files which don't have a given extension, from a given folder
  #' @param folder_path the directory path
  #' @param extension_to_retain the extension with which files will be keps
  
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
                                           
############################################
make_dhs_admin_df <- function(input_dhs_df, original_admin_column="V024", new_admin_name_colname='DHS_ADM1_NAME', new_admin_code_colname='DHS_ADM1_CODE'){
  
  #' make a data.table with admin names and admin id columns for DHS data, for easier matching with DHIS2 data
  #' @param input_dhs_df the DHS data
  #' @param original_admin_column the column which contains the named vector of codes + labels for the admin units
  #' @param new_admin_name_column how to call the admin labels column
  #' @param new_admin_code_column how to call the admin codes column (these will be used for merging later on)
  #' @returns a data.table with only the codes and the names of the admin units, for subsequent merging with the DHS full data
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

###########################
make_dhs_adm1_u5mort_dt <- function(dhs_adm1_dt){
  #' TODO see about adding column names as params (case-insensitive)
  #' use chmort from DHS.rates library, to compute smaple avg, lower/upper 95% CI for under-five (u5) mortality
  #' chmort results for under-five mortality (mortalité infanto-juvénile) tested against Burkina Faso 2021 DHS report
  #' @param dhs_adm1_dt a data.table containing only one region (adm1 unit)
  #' @returns a data.table with the DHS adm1 id, u5 mortality (sample average, lower CI, upper CI)
  #' 
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


###################################
make_dhs_map <- function(plot_dt, plot_colname, title_name, legend_title="Percentage", scale_limits = c(0, 100)) {
  #' make, show and save coverage map (coropleth of coverage proportions
  #' TODO in a subsequent version, it was requested these be percentages. code is changes, maybe also change the names
  #' prints the plot, saves it to a file, and returns the plot object
  #'
  #' @param plot_dt spatial df with attribute data
  #' @param plot_colname string with the name of the column that contains the coverage values
  #' @param title_name name for the title
  #' @param legend_title scale of the indicator
  #' @param scale_limits vector for range of scale values
  #' @return ggplot object of the map
  
  plot_obj <- ggplot(plot_dt) +
    geom_sf(aes(fill = get(plot_colname))) +
    coord_sf() +
    scale_fill_gradient(
      limits = scale_limits,
      low = "white",
      high = "navy",
      na.value = "grey90"
    ) +
    theme_classic() +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      legend.position = "bottom",
      legend.key.width = unit(2, "cm"),
      legend.text = element_text(size = 10)
    ) +
    labs(
      title = title_name,
      fill = legend_title
    )
  
  print(plot_obj)
  
  return(plot_obj)
}
                                           
####################################
make_ci_plot <- function(df_to_plot, admin_colname, point_estimation_colname, ci_lower_colname, ci_upper_colname, title_name, x_title, y_title){
  #' Make confidence interval plots for DHS data
  ci_plot <- ggplot(data = df_to_plot)
  ci_plot <- ci_plot + geom_bar(aes(x=get(admin_colname), y=get(point_estimation_colname)), fill = "#a8aabc", stat="identity")
  ci_plot <- ci_plot + geom_errorbar(aes(
    x=get(admin_colname),
    ymin=get(ci_lower_colname),
    ymax=get(ci_upper_colname)),
    width = 0.4, color ="#091bb8", linewidth = 1.5
  )
  # # Uncomment below to add value labels
  # # text for the lower bound
  # ci_plot <- ci_plot + geom_text(aes(
  #   x=get(admin_colname),
  #   y=get(ci_lower_colname),
  #   label = round(get(ci_lower_colname),1)
  # ),
  # size= 2, vjust = 1
  # )
  # # text for the upper bound
  # ci_plot <- ci_plot + geom_text(aes(
  #   x=get(admin_colname),
  #   y=get(ci_upper_colname),
  #   label = round(get(ci_upper_colname),1)
  # ),
  # size= 2, vjust = 1
  # )
  ci_plot <- ci_plot + labs(title = title_name)
  ci_plot <- ci_plot + labs(x= x_title, y = y_title)
  ci_plot <- ci_plot + theme_minimal()
  ci_plot <- ci_plot + coord_flip()
  print(ci_plot)
  return(ci_plot)
}
                                           
#%% MISC FUNCTIONS
                                           
#########################################
aggregate_geometry <- function(sf_data, admin_id_colname, admin_name_colname) {
  #' aggregate the geometries of sf data, at a specified level, given by id and name columns
  #' @param sf_data the input data
  #' @param admin_id_colname the column name which contains the id's
  #' @param admin_name_colname the column name which contains the names
  #' @returns the aggregated sf data
  by_list <- list(
    sf_data[[admin_id_colname]],
    sf_data[[admin_name_colname]]
  )
  names(by_list) <- c(admin_id_colname, admin_name_colname)
  
  result <- aggregate(sf_data["geometry"], by = by_list, FUN = sf::st_union)
  return(result)
}

#################################
delete_otherextension_files <- function(folder_path, extension_to_retain=".zip"){
  #' Delete files which don't have a given extension, from a given folder
  #' @param folder_path the directory path
  #' @param extension_to_retain the extension with which files will be keps
  
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

##########################
clean_admin_names <- function(input_vector, string_to_remove='province') {
  #' Clean the admin names of certain countries' pyramids, by removing the string_to_remove if present (these are usually "Province" or "Zone de santé" or "District") and then removing the prefix (some countries hav)
  #' @param input_vector the vector of admin names to clean
  #' @param string_to_remove the substring to delete from the admin names, if a string indicating the admin unit type is present
  #' @returns the cleaned vector
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

####################
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

                                           
################################
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
        read.csv(input_file_path, header = TRUE),
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


################################
reproject_epsg <- function(x, epsg_value) {
  #' reproject a sf or terra vector to a given epsg code
  #' @param x: object
  #' @param epsg_value: integer epsg code to reproject to
  #' @returns: input object reprojected to the target crs if needed
  
  # check if input is sf
  if (inherits(x, "sf") || inherits(x, "sfc")) {
    current_epsg <- sf::st_crs(x)$epsg
    target_epsg  <- epsg_value
    
    if (is.na(current_epsg) || current_epsg != target_epsg) {
      message(glue::glue("Info: reprojecting sf object to EPSG:{target_epsg}."))
      x <- sf::st_transform(x, target_epsg)
    } else {
      message("Info: no reprojection needed for sf object.")
    }
    
  # check if input is terra vector
  } else if (inherits(x, "SpatVector")) {
    current_crs <- terra::crs(x, describe = TRUE)$code
    target_crs  <- epsg_value
    
    if (is.na(current_crs) || current_crs != target_crs) {
      message(glue::glue("Info: reprojecting terra vector to EPSG:{target_crs}."))
      x <- terra::project(x, paste0("EPSG:", target_crs))
    } else {
      message("Info: no reprojection needed for terra vector.")
    }
    
  } else {
    stop("Input must be an sf or terra vector object.")
  }
  
  return(x)
}

########################
filter_points_within_boundaries <- function(locations_vect, boundaries_vect, epsg_value_degrees) {

  #' filter points within polygon boundaries using terra
  #'
  #' @param locations_vect: vector with point geometries
  #' @param boundaries_vect: vector with polygon geometries
  #' @param epsg_value_degrees: EPSG code for the geographic (degree-based) CRS (eg, for Burkina 4326)
  #'
  #' @return vector with only the points within the boundaries
  #'
  #' @import terra
  #'
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

########################
make_coverage_radii_sf <- function(
  input_vect,
  coordinate_colnames,
  epsg_value_degrees,
  epsg_value_meters,
  radius_meters
){

  #' make circles of a given radius around each point (longitude/latitude) in the sf vector input data
  #'
  #' @param input_vect: sf vector of spatial points (in any CRS)
  #' @param coordinate_colnames: names of the longitude and latitude columns
  #' @param epsg_value_degrees: EPSG code for the geographic (degree-based) CRS (eg, for Burkina 4326)
  #' @param epsg_value_meters: EPSG code for the projected (meter-based) CRS (eg, for Burkina 3857)
  #' @param radius_meters: Integer of the radius (in meters) of the  coverage area to create around each point
  #'
  #' @return: sf vector of the circle coverages in the degree CRS
  #'
  #' @details 
  #' 1. check that input is in the correct degree CRS (reproject if needed)
  #' 2. project it to a meter CRS for distance calculations
  #' 3. create circular buffers (coverage radii) around each point
  #' 4. reproject the buffer geometries back to the original degree CRS
  #'
  #' @import sf

  # check CRS and reproject to degree CRS if necessary
  input_vect <- reproject_epsg(input_vect, epsg_value_degrees)
  
  # reproject to a meter CRS
  vect_meters <- st_transform(input_vect, epsg_value_meters)
  
  # create the circles/buffers around each point
  coverage_radii_meters <- st_buffer(vect_meters, dist = radius_meters)
  
  # reproject back to degree CRS for mapping
  coverage_radii_degrees <- st_transform(coverage_radii_meters, epsg_value_degrees)
  
  return(coverage_radii_degrees)
}

########################
make_coverage_radii_terra <- function(
  
  input_vect,
  coordinate_colnames,
  epsg_value_degrees,
  epsg_value_meters,
  radius_meters
){

  #' make circles of a given radius around each point (longitude/latitude) in the spatial input data
  #'
  #' @param input_vect: terra vector of spatial points (in any CRS)
  #' @param coordinate_colnames: names of the longitude and latitude columns
  #' @param epsg_value_degrees: EPSG code for the geographic (degree-based) CRS (eg, for Burkina 4326)
  #' @param epsg_value_meters: EPSG code for the projected (meter-based) CRS (eg, for Burkina 3857)
  #' @param radius_meters: Integer of the radius (in meters) of the  coverage area to create around each point
  #'
  #' @return: terra vector of the circle coverages in the degree CRS
  #'
  #' @details 
  #' 1. check that input is in the correct degree CRS (reproject if needed)
  #' 2. project it to a meter CRS for distance calculations
  #' 3. create circular buffers (coverage radii) around each point
  #' 4. reproject the buffer geometries back to the original degree CRS
  #'
  #' @import terra
  #'
  
  # check CRS and reproject to degree CRS if necessary

  input_vect <- reproject_epsg(input_vect, epsg_value_degrees)

  # reproject to a meter CRS
  vect_meters <- project(input_vect, paste0("EPSG:", epsg_value_meters))
  
  # create the circles/buffers around each point
  coverage_radii_meters <- buffer(vect_meters, width = radius_meters)
  
  # reproject back to degree CRS for mapping
  coverage_radii_degrees <- project(coverage_radii_meters, paste0("EPSG:", epsg_value_degrees))
  
  return(coverage_radii_degrees)
}

########################
make_overlaid_sf_plot <- function(
  #' plot overlaying a) administrative boundaries, b) location of healthcare units, c) buffers around each healthcare unit
  
  admin_unit_vect,
  points_sf_vect,
  buffer_vect,
  epsg_value_degrees,
  plot_title
){

  #' map overlaying a) healthcare unit locations, b) administrative boundaries, and c) buffer zones around healthcare units, projected to a common CRS
  #'
  #' @param admin_unit_vect: sf vector of administrative boundaries
  #' @param points_sf_vect: sf vector with coordinate columns of healthcare units
  #' @param buffer_vect: sf vector of buffer zones around healthcare units
  #' @param epsg_value_degrees: EPSG code (in degrees) for CRS
  #' @param plot_title: title of plot
  #'
  #' @return ggplot object showing the spatial overlay

  # get all 3 data objects to the same projection

  # a) ensure the healthcare locations have the proper projection
  points_sf_vect <- reproject_epsg(points_sf_vect, epsg_value_degrees)

  # b) ensure the admin geo data has the proper projection
  admin_unit_vect <- reproject_epsg(admin_unit_vect, epsg_value_degrees)

  # c) ensure the buffer data has the proper projection
  buffer_vect <- reproject_epsg(buffer_vect, epsg_value_degrees)

  plot <- ggplot() +
    geom_sf(data = admin_unit_vect, fill = "gray95", color = "black") +
    geom_sf(data = buffer_vect, fill = "dodgerblue", alpha = 0.3) +
    geom_sf(data = points_sf_vect, color = "dodgerblue4", size = 0.5) +
    theme_minimal() +
    ggtitle(plot_title)

  # print(plot)  # do not print

  return(plot)
}
           

########################
make_rasterized_inclusion_data <- function(
  buffer_vect, 
  raster_data,
  epsg_value_degrees,
  value_inside = 1,
  value_outside = 0
){

  #' make a new raster layer aligned with the original raster, where each cell is a specific value if it intersects any buffer in the vector data and another specific value if not

  #' @param buffer_vect: vector with the buffer geometries to rasterize
  #' @param raster_data: raster to use as the template for resolution and extent
  #' @param epsg_value_degrees: EPSG of the target CRS in degrees
  #' @param value_inside: value to assign to raster cells that intersect any buffer
  #' @param value_outside: value to assign to raster cells that do not intersect any buffer

  #' @return raster with cells assigned values based on intersection with the buffer vector
  
  # reproject raster to the correct CRS (degrees)
  raster_data <- project(raster_data, glue("epsg:{epsg_value_degrees}"))
  
  # if buffer CRS differs, reproject buffer to raster CRS
  buffer_vect <- reproject_epsg(buffer_vect, epsg_value_degrees)
  
  # convert sf to terra SpatVector for rasterization
  buffer_vect_terra <- terra::vect(buffer_vect)
  
  # rasterize the buffer: cells inside = value_inside, outside = value_outside
  inclusion_data <- terra::rasterize(
    buffer_vect_terra,
    raster_data,
    field = value_inside,
    background = value_outside
  )
  
  return(inclusion_data)
}
                                        
########################                                                 
check_or_create_dataset <- function(
    target_workspace, 
    target_dataset_slug, 
    target_dataset_name, 
    target_dataset_description) {
  
  #' Check if a dataset with the given slug exists in the given workspace
  #' If not, create it; if yes, retrieve it
  #' @param target_workspace the workspace to search/create in
  #' @param target_dataset_slug identifier of the dataset to create/search
  #' @param target_dataset_name the name of the new dataset to create (if necessary)
  #' @param target_dataset_description the description of the new dataset to create (if necessary)
  #' @returns the new or existing dataset which matches the slug
  #'
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

######################                             
make_new_dataset_version <- function(target_dataset){
    #' create a new version of a given dataset
    #' increments the latest version number of the given dataset and create a new version
    #' @param target_dataset the dataset to make the new version in; must already have an initial version
    #' @returns a new dataset version object created with an incremented version number
    
    # get latest dataset version name
    latest_version_name <- target_dataset$latest_version$name
    latest_version_num <- as.numeric(gsub("v", "", latest_version_name))
    
    # create a new version, which increments the current version by 1
    new_version <- target_dataset$create_version(paste0("v", latest_version_num + 1))

    return(new_version)
}


## helper function for NER child-parent updates
# Function to update child facilities dynamically for any level -> move "level" to "target_level" updating parents
get_updated_children <- function(new_level_table, group_table, level, target_level, parent_level) {

    if (level < 2) stop(glue("level must be greater than 2, received: {level}"))

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
