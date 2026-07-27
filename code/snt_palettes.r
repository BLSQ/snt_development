
# List of palettes to match the default colors used in WebApp 
# (courtesy to Benjamin Wilfart)

# Use: source() this script to import palettes in any R notebook/script (i.e., mostly report nbs)


# Palettes ----------------------------------------------------------------------

## OpenHEXA WebApp SNT default color palettes -----------------------------------

na_color <- "#D3D3D3"

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


cat_healthcare_covered_palette <- c("#f4b5d0", "#f5efa5", "#a5e9c8")

# cat_reporting_rate_palette <- c(
#   "#8d0801ff", # Blood Red
#   "#bf0603ff", # Brick Ember
#   "#f4d58dff", # Jasmine
#   "#708d81ff", # Deep Teal
#   "#001427ff"  # Prussian Blue
# )

cat_reporting_rate_palette <- c(
  "#bc4b51ff", # Blushed Brick
  "#f4a259ff", # Sandy Brown
  "#f4e285ff", # Light Gold
  "#8cb369ff", # Muted Olive
  "#5b8e7dff"  # Jungle Teal
)

