# ================================================
# Title: Utility Functions for SNT Process
# Description: This script contains utility functions used for SNT computation workflow.
# Author: Esteban Montandon
# Created: [2024-10-01]
# Last updated: [2025-06-20]
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


# helper function para loggear
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



# iulia -------------------------------------------------------------------
                                                 
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
  
  output_dt <- copy(as.data.table(input_dt))
  
  output_dt <- output_dt[order(get(admin_colname), get(year_colname), get(month_colname))]  # Sort by location and time
  
  # compute common denominator (block of 12 months - annual values)
  denominator_colname <- paste(toupper(indicator), 'SUM', 12, 'MTH', 'FW', sep = '_')
  output_dt[, (denominator_colname) := Reduce(`+`, shift(get(values_colname), 0:(12 - 1), type = "lead"), init = 0), by = get(admin_colname)]
  
  # compute numerators, proportions and dichotomous seasonality variables
  for (num_months in vector_of_durations) {
    
    # compute the block of durations (3/4/5 months for example) to serve as numerator in the algorithm
    numerator_colname <- paste(toupper(indicator), 'SUM', num_months, 'MTH', 'FW', sep = '_')
    output_dt[, (numerator_colname) := Reduce(`+`, shift(get(values_colname), 0:(num_months - 1), type = "lead"), init = 0), by = get(admin_colname)]
    
    # compute proportion of the indicator which happens in the numerator block
    proportion_colname = paste(toupper(indicator), num_months, 'MTH', 'ROW', 'PROP', sep = '_')
    seasonality_colname = paste(toupper(indicator), num_months, 'MTH', 'ROW', 'SEASONALITY', sep = '_')
    output_dt[, (proportion_colname) := get(numerator_colname) / get(denominator_colname)]
    output_dt[, (seasonality_colname) := as.integer(get(denominator_colname) > 0 & get(proportion_colname) >= proportion_threshold)]
  }
  
  return(output_dt)
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
compute_min_seasonality_block <- function(input_dt, seasonality_column_pattern, vector_of_possible_month_block_sizes, indicator, seasonal_blocksize_colname, valid_value = 1){
  #' retrieve the minimum number of months which constitute a seasonality block
  #' @param input_dt input data.table
  #' @param seasonality_column_pattern in the names of the columns which represent the seasonality status (0/1)
  #' @param vector_of_possible_month_block_sizes possible sizes of the month blocks (as a vector of integers)
  #' @param indicator name of the new column
  #' @param valid_value value which indicates there is seasonality
  #' @return an output data table which has the extra column; it will be Inf (infinite) if there is no seasonality, and the number of months in a block, if there is seasonality
    
  indicator <- toupper(indicator)
  
  # Extract column names matching the pattern
  seasonality_cols <- grep(toupper(seasonality_column_pattern), names(input_dt), value = TRUE)
  
  # compute the new column
  # seasonal_blocksize_colname <- paste(indicator, 'SEASONALITY_DURATION', sep = '_')
  # output_dt <- input_dt[, (seasonal_blocksize_colname) := apply(.SD, 1, function(row) {
  #   min(vector_of_possible_month_block_sizes[which(row == 1)], na.rm = TRUE)
  # }), .SDcols = seasonality_cols]

  # compute the new column  
  # muffle the warnings for the admin units where the seasonality block duration is infinity (no seasonality)
  output_dt <- input_dt[, (seasonal_blocksize_colname) := apply(.SD, 1, function(row) {
    withCallingHandlers(
      min(vector_of_possible_month_block_sizes[which(row == valid_value)], na.rm = TRUE),
      warning = function(w) {
        if (grepl("no non-missing arguments to min; returning Inf", conditionMessage(w))) {
          invokeRestart("muffleWarning")
        }
      }
    )
  }), .SDcols = seasonality_cols]
  
  # change the infinite values (no seasonality) to missing
  output_dt[is.infinite(get(seasonal_blocksize_colname)), (seasonal_blocksize_colname) := NA]
  
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
    guides(fill=guide_legend(title= paste0("Seasonality (", title_label, ")"), nrow = 2)) +
    theme_classic() +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          legend.position = "bottom", legend.key.width = unit(2,"cm"), legend.text=element_text(size=10))
  
  print(seasonality_plot)
  
  return(seasonality_plot)
}

#############
make_seasonality_duration_plot <- function(spatial_seasonality_df, seasonality_duration_colname, title_label, palette_name = 'BrBG'){
  #' map the duration of seasonality (in how many months x% of annual rain falls)
  #'
  #' @param spatial_seasonality_df sf data with spatial and seasonality columns
  #' @param seasonality_duration_colname column name (string) for seasonality duration (number of months)
  #' @param title_label string for the legend title
  #' @param palette_name colorbrewer palette (default is 'BrBG')
  #'
  #' @return ggplot object
  #' 
  duration_plot <- ggplot(spatial_seasonality_df) +
    geom_sf(aes(fill = as.character(get(seasonality_duration_colname))))+
    coord_sf() + # map projection
    # scale_fill_discrete(
    #   # values = sort(unique(as.character(plot_df[['seasonality_duration_colname']]))),  # custom colors
    #   labels = function(x) {
    #     ifelse(x == "Inf", "No seasonality", x) # custom labels
    #     }
    #   )    +
    # 
    scale_fill_brewer(palette = palette_name, labels = function(x) {
      ifelse(is.na(x), "No seasonality", x) # custom labels
    }
    ) +
    guides(fill=guide_legend(title= paste0("Number of months (", title_label, ")" ), nrow = 2)) +
    theme_classic() +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          legend.position = "bottom", legend.key.width = unit(2,"cm"), legend.text=element_text(size=10))
  
  print(duration_plot)
  return(duration_plot)
}