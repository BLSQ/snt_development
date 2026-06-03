
# List of palettes to match the default colors used in WebApp 
# (courtesy to Benjamin Wilfart)

# Use: source() this script to import palettes in any R notebook/script (i.e., mostly report nbs)


# Palettes ----------------------------------------------------------------------

## OpenHEXA WebApp SNT default color palettes -----------------------------------

FOUR_SHADES = c(
  "#A2CAEA",
  "#ACDF9B",
  "#F2B16E",
  "#A93A42"
)

FIVE_SHADES = c(
  "#A2CAEA",
  "#ACDF9B",
  "#F5F1A0",
  "#F2B16E",
  "#A93A42"
  )

SIX_SHADES = c(
  "#A2CAEA",
  "#ACDF9B",
  "#F5F1A0",
  "#F2B16E",
  "#E4754F",
  "#A93A42"
  )

SEVEN_SHADES = c(
  "#A2CAEA",
  "#6BD39D",
  "#ACDF9B",
  "#F5F1A0",
  "#F2B16E",
  "#E4754F",
  "#A93A42"
  )

EIGHT_SHADES = c(
  "#A2CAEA",
  "#6BD39D",
  "#ACDF9B",
  "#F5F1A0",
  "#F2B16E",
  "#E4754F",
  "#C54A53",
  "#A93A42"
  )

NINE_SHADES = c(
  "#A2CAEA",
  "#80B3DC",
  "#6BD39D",
  "#ACDF9B",
  "#F5F1A0",
  "#F2B16E",
  "#E4754F",
  "#C54A53",
  "#A93A42"
  )

TEN_SHADES = c(
  "#A2CAEA",
  "#80B3DC",
  "#6BD39D",
  "#ACDF9B",
  "#F5F1A0",
  "#F2D683",
  "#F2B16E",
  "#E4754F",
  "#C54A53",
  "#A93A42"
  )


### OpenHEXA WebApp SNT default risk level colors ----------------------------

RISK_LOW = "#A5D6A7"
RISK_MEDIUM = "#FFECB3"
RISK_HIGH = "#FECDD2"
RISK_VERY_HIGH = "#FFAB91"

# TBD if needed and how to use it in R ... 
# ORDINAL = {
#   2: [RISK_LOW, RISK_VERY_HIGH],
#   3: [RISK_LOW, RISK_MEDIUM, RISK_VERY_HIGH],
#   4: [RISK_LOW, RISK_MEDIUM, RISK_HIGH, RISK_VERY_HIGH],
# }


### Custom palettes ---------------------------------------------------

palette_pfpr_map_mis <- c(
  "#EEF3F3",
  "#F6B7B2",
  "#DB675E",
  "#C10534",
  "#851B2E",
  "#611924"
)


# From "221216_NER_Stratification.pptx" (NER GC7 WHO)
palette_incidence_ner_cg7 <- c(
  "#ADD8E6",
  "#82C0E9",
  "#008BBC",
  "#FF8080",
  "#C10534"
)

# BDI AHADI categorical palette for access to healthcare
bdi_cat_healthcare_access_ahadi_palette <- c(
  "#8B0020", # 0-95%
  "#E8E8A0", # 95-99%
  "#1A6B2A" # 99-100%
)


# Functions (related to palettes) ------------------------------------------------------
# I would keep palette definitions and functions in the same file (no need to move to snt_utils.r)

get_range_from_count <- function(count) {
  if (count == 3) {
    return(FOUR_SHADES)
  }
  if (count == 4) {
    return(FIVE_SHADES)
  }
  if (count == 5) {
    return(SIX_SHADES)
  }
  if (count == 6) {
    return(SEVEN_SHADES)
  }
  if (count == 7) {
    return(EIGHT_SHADES)
  }
  if (count == 8) {
    return(NINE_SHADES)
  }
  if (count == 9) {
    return(TEN_SHADES)
  }
  return(SEVEN_SHADES)
}

# # Example usage:
# get_range_from_count(5)
# # [1] "#A2CAEA" "#ACDF9B" "#F5F1A0" "#F2B16E" "#E4754F" "#A93A42"

#' @description
#' make equal frequency breaks for binning the distributions (for maps)
#'
#' @param x vector (numeric)
#' @param num_categories number of groups to make
#' @return the vector of breakpoints for binning
make_equal_frequency_breaks <- function(x, num_categories) {
  probs <- seq(0, 1, length.out = num_categories + 1)
  as.numeric(quantile(x, probs = probs, na.rm = TRUE))
}


#' @description
#' make k-means based breaks for binning the distributions (for maps)
#'
#' @param x vector (numeric)
#' @param num_categories number of groups to make
#' @return the vector of breakpoints for binning
make_k_means_breaks <- function(x, num_categories) {
  km <- kmeans(x[!is.na(x)], centers = num_categories, nstart = 25)
  centers <- sort(as.numeric(km$centers))
  midpoints <- (centers[-length(centers)] + centers[-1]) / 2
  c(min(x, na.rm = TRUE), midpoints, max(x, na.rm = TRUE))
}

cut_to_categories <- function(input_dt, input_colname, output_colname, breaks, num_decimals = 1, suffix = "%") {

  if (!input_colname %in% names(input_dt)) {
      stop(paste("Input numeric column to cut not found in the input data"))
  }
    
  if (length(breaks) < 2) {
      stop("At least 2 breaks required to cut the input data")
  }
    
  if (!all(diff(breaks) > 0)) {
      stop("Breaks to cut the input data, must be sorted in ascending order")
  }

  output_dt <- copy(as.data.table(input_dt)) 
  
  # make the labels from breaks
  low  <- breaks[-length(breaks)] # remove last value from labels
  high <- breaks[-1] # remove first value from labels 
    
  labels <- character(length(low))

  # first label (strictly smaller than x)
  labels[1] <- paste0("< ", round(high[1], num_decimals), suffix)

  # last label (greater or equal than x)
  labels[length(labels)] <- paste0(">= ", round(low[length(low)], num_decimals), suffix)
  
  # all middle intervals
  if (length(low) > 2) {

      mid_idx <- seq(2, length(low) - 1)

      labels[mid_idx] <- paste0(
        "[",
        round(low[mid_idx], num_decimals),
        "-",
        round(high[mid_idx], num_decimals),
        ")",
        suffix
      )
  }
    
  # cut and assign
  output_dt[, (output_colname) := cut(
      get(input_colname),
      breaks = breaks,
      labels = labels,
      include.lowest = TRUE,
      right = FALSE
  )]

  return(output_dt)
}









