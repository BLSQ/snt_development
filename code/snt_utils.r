
# ================================================
# Title: Utility Functions for SNT Process
# Description: This script contains utility functions used for SNT computation workflow.
# Author: Esteban Montandon
# Created: [2024-10-01]
# Last updated: [2025-04-24]
# Dependencies: stringi, httr, arrow, tools, jsonlite
# Notes:
#   - [Optional: Any special considerations, references, or tips]
# ================================================

 


# function to create (make dir) the required directories for data sub-folders
# create_data_directories <- function(folders){
#     # Create directories
#     for (dir in dirs_to_create) {
#       if (!dir.exists(dir)) {
#         print(paste('Creating ', dir))
#         dir.create(dir, recursive = TRUE)
#       }
#     }    
# }



# Check if all names are aligned
# check_identical_names <- function(alignment_list) {  
  
#   reference <- alignment_list[[1]]
#   all_identical <- all(sapply(alignment_list, function(x) setequal(reference, x)))
#   return(all_identical)  
# }


# Create ALIAS name for name matching
# create_alias_file_OLD <- function(alignment_list, path) {

#     sorted_lists <- lapply(alignment_list, sort)
#     max_length <- max(sapply(sorted_lists, length))        
#     padded_lists <- lapply(sorted_lists, function(x) c(x, rep(NA, max_length - length(x)))) # Pad vectors
#     first_list_name <- paste0(names(alignment_list)[1], "_REFERENCE") 
    
#     # Initialize the data frame with the first list, using its name as the column name
#     df <- data.frame(first_list_name = padded_lists[[1]], stringsAsFactors = FALSE)
#     names(df)[1] <- first_list_name   
    
#     # Loop through the remaining lists to add columns and their aliases
#     for (i in 2:length(padded_lists)) {
#         current_name <- names(alignment_list)[i]
#         current_list <- padded_lists[[i]]
    
#         # Add the current list as a column
#         df[[current_name]] <- current_list
        
#         # Create the alias column
#         alias_column_name <- paste0(current_name, "_ALIAS")
#         df[[alias_column_name]] <- ifelse(df[[first_list_name]] == current_list, df[[first_list_name]], "-")
#     }
    
#     # Write the data frame to a CSV file
#     write.csv(df, file = path, row.names = FALSE)
# }

                           

# Create ALIAS name for name matching
# create_alias_file <- function(alignment_list, path) {
#     sorted_lists <- lapply(alignment_list, sort)
#     max_length <- max(sapply(sorted_lists, length))        
#     padded_lists <- lapply(sorted_lists, function(x) c(x, rep(NA, max_length - length(x)))) # Pad vectors
#     first_list_name <- names(alignment_list)[1] 
    
#     # Initialize the data frame with the first list, using its name as the column name
#     df <- data.frame(first_list_name = padded_lists[[1]], stringsAsFactors = FALSE)
#     names(df)[1] <- first_list_name   
#     wrong_cols <- c()
                           
#     # Loop through the remaining lists to add columns and their aliases
#     for (i in 2:length(padded_lists)) {
#         current_name <- names(alignment_list)[i]
#         current_list <- padded_lists[[i]]
    
#         # Add the current list as a column
#         df[[current_name]] <- current_list
        
#         # Create the alias column
#         alias_column_name <- paste0(current_name, "_ALIAS")
#         set_value <- ifelse(df[[first_list_name]] == current_list, df[[first_list_name]], "-") #
#         df[[alias_column_name]] <- set_value #
#         if (any(set_value == "-")) {wrong_cols <- unique(c(wrong_cols, current_name, alias_column_name))} #
#     }
#     #
#     df <- df[,c(first_list_name, wrong_cols)]          
#     # Filter by rows?
#     # filtered_df <- df %>%
#     #                 rowwise() %>% 
#     #                 filter(any(c_across(everything()) == "-")) %>%
#     #                 ungroup()
#     # Write the data frame to a CSV file
#     # write.csv(filtered_df, file = path, row.names = FALSE) 
#     write.csv(df, file = path, row.names = FALSE) 
# }


# Check the alias file is correctly formatted 
# check_correct_alias_file <- function(alias_file_path, col_reference) {

#     # load the file and check the names in ALIAS column
#     alias_file <- read.csv(alias_file_path)
#     alias_columns <- grep("_ALIAS", colnames(alias_file), value = TRUE)
    
#     if (length(alias_columns) == 0){
#         return(FALSE)
#     }    

#     # Check if the file is updated
#     if (any(alias_file[alias_columns] == "-")) {
#         return(FALSE)
#     }

#     # Check all alias values are equal to the reference column
#     result <- all(
#         sapply(alias_columns, function(col_name) {
#             setequal(alias_file[[col_name]] , alias_file[['DHIS2_REFERENCE']])
#             })
#         )
    
#     return(result)     
# }


                 
# function to replace the unmatched admin names in a specific column  
# set_alias_names <- function(update_df, adm_col_selection, alias_file_path, ref_column, org_names_column, alias_column){
        
#     alias_file <- read.csv(alias_file_path)
#     ref_names <- unique(alias_file[[ref_column]])
#     alias_names <- unique(alias_file[[alias_column]])
    
#     org_names <- unique(update_df[[adm_col_selection]]) # check names over the original df
#     names_to_update <- setdiff(org_names, ref_names)
    
#     for (a in names_to_update){
#         alias_value <- alias_file[alias_file[[org_names_column]] == a, alias_column]
#         print(paste0("Replacing: ",  a, " with: ", alias_value, " from ", alias_column))           
#         update_df[update_df[[adm_col_selection]] == a, adm_col_selection] <- alias_value
#     }

#     return(update_df)
# }

                           
                           
# add any other matching logic here
format_names <- function(x) {
    x <- stri_trans_general(str = x, id = "Latin-ASCII") # remove weird characters
    trimws(gsub("  +", " ", toupper(gsub("[^a-zA-Z0-9]", " ", x))))  # keep numbers, remove spaces
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
log_msg <- function(msg) {
    print(msg)
    if (!is.null(openhexa$current_run)) {
        openhexa$current_run$log_info(msg)
        }
}


 