# Helper functions to create palettes and make plots

make_month_colors <- function(){
  month_colors <- c(
    "#18a7e0e8",   # Janvier
    "#2060E8",     # Février
    "#5030D0",     # Mars
    "#8020B8",     # Avril
    "#B01890",     # Mai
    "#D41E55",     # Juin
    "#E83020",     # Juillet
    "#F56800",     # Août
    "#F5A800",     # Septembre
    "#C8C800",     # Octobre
    "#60C820",     # Novembre
    "#20b883"      # Décembre
  )

  return(month_colors)
}

make_month_labels_fr <- function(){
  month_labels_fr <- c(
    "Janvier", "Février", "Mars", "Avril",
    "Mai", "Juin", "Juillet", "Août",
    "Septembre", "Octobre", "Novembre", "Décembre"
  )

  return(month_labels_fr)
}


#%% SEASONALITY PLOTS -------------------------------------------------------------------

#' @description
#' make ridgeline plot sorting y-axis categories by amount/height and set x-axis labels to show only the beginning of each year (1st month)
#' 
#' @param dt data.table with the plotting data
#' @param x_colname column for x-axis values (period)
#' @param y_colname column for y-axis categories
#' @param height_colname column which gives the ridge height (the values)
#' @param year_colname column for x-axis labels
#' @param month_colname column to filter for 1st month/January labels
#' @param plot_title map title (defaults to NULL)
#' @param plot_subtitle map subtitle (defaults to NULL)
#' @param plot_caption map caption (defaults to NULL)
#' @param scale_constant divisor to limit the heights of the ridges
#' @param ridge_color ridgeline fill
#' @reorder_fun the statistic to use for assigning the order on the y axis (defaults to mean, but median and max are also possible)
#' 
#' @return ggplot object
make_ridgeline_plot <- function(
  dt,
  x_colname,
  y_colname,
  height_colname,
  year_colname,
  month_colname,
  plot_title = NULL,
  plot_subtitle = NULL,
  plot_caption = NULL,
  scale_constant = 15000,
  ridge_color = "#008080",
  reorder_fun = mean
) {
    ridge_plot <- ggplot(
      dt,
      aes(
        x = .data[[x_colname]],
        y = fct_reorder(.data[[y_colname]], .data[[height_colname]], reorder_fun),
        height = .data[[height_colname]] / scale_constant,
        group = .data[[y_colname]],
        scale = 2
      )
    ) +

  
    geom_ridgeline(
      alpha = 0.4, scale = 4.5, linewidth = 0.2,
      fill = ridge_color, color = "white"
    ) +
    scale_x_discrete(
      breaks = dt[get(month_colname) == 1, get(x_colname)],
      labels = dt[get(month_colname) == 1, get(year_colname)]
    ) +
    
    # titles
    labs(
        title = plot_title,
        subtitle = plot_subtitle,
        y = "",
        x = "",
        caption = plot_caption
    ) +

    # map theme
    theme_minimal() +
    theme(
      axis.ticks.y = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.border = element_blank()
    ) +

    theme(
      plot.title = ggplot2::element_text(
          face = "bold", size = 10,
          margin = ggplot2::margin(b = 4)
      ),
      plot.subtitle = ggplot2::element_text(
          size = 8, colour = "grey40",
          margin = ggplot2::margin(b = 8)
      ),
      plot.caption = ggplot2::element_text(
          size = 8,
          colour = "grey55",
          hjust = 1,
          margin = ggplot2::margin(t = 8)
      ),
      plot.margin = ggplot2::margin(10, 10, 10, 10)
    )

    return(ridge_plot)
  
  }
  

#' @description 
#' map seasonality with predefined colors
#' 
#' areas are categorized as "Seasonal" or "Not seasonal"
#'
#' @param spatial_seasonality_df sf data frame with spatial geometry and seasonality data
#' @param seasonality_colname string with the name of the column indicating seasonality (values should be 0 or 1)
#' @param plot_title map title (defaults to NULL)
#' @param plot_subtitle map subtitle (defaults to NULL)
#' @param plot_caption map caption (defaults to NULL)
#' @param legend_title title of the legend (defaults to NULL)
#' @param seasonal_color color for areas which are seasonal
#' @param seasonal_label legend label for areas which are seasonal
#' @param not_seasonal_color color for areas which are not seasonal
#' @param not_seasonal_label legend label for areas which are not seasonal
#' 
#' @return a ggplot object of the seasonality map
make_seasonality_plot <- function(
  spatial_seasonality_df,
  seasonality_colname,
  plot_title = NULL,
  plot_subtitle = NULL,
  plot_caption = NULL,
  legend_title = NULL,
  seasonal_color = "#F9A98E",
  seasonal_label = "Saisonnier",
  not_seasonal_color = "#D4E8CE",
  not_seasonal_label = "Non saisonnier"
){

  seasonality_plot <- ggplot(spatial_seasonality_df) +
    geom_sf(aes(fill = as.factor(get(seasonality_colname))))+

    scale_fill_manual(
        name = legend_title,
        # specific labels and colors
        values = c("1" = seasonal_color, "0" = not_seasonal_color),
        labels = c("1" = seasonal_label, "0" = not_seasonal_label)
    ) +
    coord_sf() + # map projection

    # titles
    ggplot2::labs(
      title = plot_title,
      subtitle = plot_subtitle,
      caption = plot_caption
    ) +

    guides(fill=guide_legend(nrow = 2)) +
    
    # map theme
    ggplot2::theme_void() +
    ggplot2::theme(
        plot.title = ggplot2::element_text(
            face = "bold", size = 10,
            margin = ggplot2::margin(b = 4)
        ),
        plot.subtitle = ggplot2::element_text(
            size = 8, colour = "grey40",
            margin = ggplot2::margin(b = 8)
        ),
        plot.caption = ggplot2::element_text(
            size = 8,
            colour = "grey55",
            hjust = 1,
            margin = ggplot2::margin(t = 8)
        ),
        legend.position = "right",
        legend.text = ggplot2::element_text(size = 8),
        plot.margin = ggplot2::margin(10, 10, 10, 10)
    )
  
  return(seasonality_plot)
}


#' @description
#' plot the start month of the high case/rainfall season with custom color palette
#'
#' @param plot_data a data frame with spatial data and a column for the start month of the season
#' @param season_start_month_col column that gives the start month values
#' @param color_vector vector of colors to be used for January-December
#' @param plot_title map title (defaults to NULL)
#' @param plot_subtitle map subtitle (defaults to NULL)
#' @param plot_caption map caption (defaults to NULL)
#' @param legend_title title of the legend (defaults to NULL)
#' @param missing_label label for the missing values (non seasonal areas)
#'
#' @return a ggplot object or NULL if season_start_month_col is not found in plot_data
make_season_start_month_plot <- function(
    plot_data,
    season_start_month_col,
    color_vector,
    color_labels,
    plot_title = NULL,
    plot_subtitle = NULL,
    plot_caption = NULL,
    legend_title = NULL,
    missing_label = "Non saisonnier"
) {
  
  # validate inputs
  stopifnot(
    "seasonality start plot_data must be an sf object" = inherits(plot_data, "sf"),
    "seasonality start month column must be a single string" = is.character(season_start_month_col) && length(season_start_month_col) == 1,
    "seasonality start month column not found in plot_data" = season_start_month_col %in% names(plot_data),
    "seasonality start color_vector must have exactly 12 elements" = length(color_vector) == 12,
    "seasonality start color_labels must have exactly 12 elements" = length(color_labels) == 12
  )
 
  month_vals <- plot_data[[season_start_month_col]]
  valid_vals <- month_vals[!is.na(month_vals)]
  if (!all(valid_vals %in% 1:12)) {
    stop("Column '", season_start_month_col,
        "' contains values outside 1–12: ",
        paste(sort(unique(valid_vals[!valid_vals %in% 1:12])), collapse = ", "))
  }
 
  # make a factor column with ordered levels 1–12 for the months
  # NA stays NA so it gets the na.value color in scale_fill_manual
  plot_data <- plot_data |>
    dplyr::mutate(
      .month_factor = factor(.data[[season_start_month_col]], levels = 1:12)
    )
 
  # make color / label scales including only the months that are in the data
  present_months <- sort(unique(as.integer(
    levels(droplevels(plot_data$.month_factor))
  )))
  present_months <- present_months[present_months %in% 1:12]
 
  scale_values <- stats::setNames(color_vector[present_months],
                                  as.character(present_months))
  scale_breaks <- as.character(present_months)
  scale_labels <- color_labels[present_months]
 
  # make the plot
  season_first_month <- ggplot2::ggplot(data = plot_data) +
 
    # geo layer
    ggplot2::geom_sf(
      ggplot2::aes(fill = .month_factor),
      colour = "white",
      linewidth = 0.15
    ) +
 
    # discrete color scale with NA handling
    ggplot2::scale_fill_manual(
      values = scale_values,
      breaks = scale_breaks,
      labels = scale_labels,
      na.value = "#D3D3D3",
      name = legend_title,
      guide = ggplot2::guide_legend(
        title = legend_title,
        override.aes = list(colour = "white", linewidth = 0.3),
        label.position = "right",
        keywidth = ggplot2::unit(0.9, "lines"),
        keyheight = ggplot2::unit(0.9, "lines")
      )
    ) +
 
    # titles
    ggplot2::labs(
      title = plot_title,
      subtitle = plot_subtitle,
      caption = plot_caption
    ) +
 
    # map theme
    ggplot2::theme_void() +
    ggplot2::theme(
        plot.title = ggplot2::element_text(
            face = "bold", size = 10,
            margin = ggplot2::margin(b = 4)
        ),
        plot.subtitle = ggplot2::element_text(
            size = 8, colour = "grey40",
            margin = ggplot2::margin(b = 8)
        ),
        plot.caption = ggplot2::element_text(
            size = 8,
            colour = "grey55",
            hjust = 1,
            margin = ggplot2::margin(t = 8)
        ),
        legend.position = "right",
        legend.text = ggplot2::element_text(size = 8),
        plot.margin = ggplot2::margin(10, 10, 10, 10)
    )
 
  # append NA note to caption if any missing values exist in the data, so they also appear in the legend
  if (anyNA(month_vals)) {
    season_first_month <- season_first_month + ggplot2::labs(
      caption = paste0(plot_caption,
        if (nchar(plot_caption) > 0) "\n" else "",
        "\u25A0 : ", missing_label)
      )
  }
 
  return(season_first_month)
}


#' @description
#' map the duration of seasonality (in how many months x% of annual rain/cases fall)
#'
#' @param spatial_seasonality_df sf data with spatial and seasonality columns
#' @param seasonality_duration_colname column name (string) for seasonality duration (number of months)
#' @param plot_title map title (defaults to NULL)
#' @param plot_subtitle map subtitle (defaults to NULL)
#' @param plot_caption map caption (defaults to NULL)
#' @param legend_title title of the legend (defaults to NULL)
#' @param color_vector vector of colors for the plot
#' @param none_label legend label when there is no seasonality
#' 
#' @return ggplot object
make_season_duration_plot <- function(
  spatial_seasonality_df,
  seasonality_duration_colname,
  plot_title = NULL,
  plot_subtitle = NULL,
  plot_caption = NULL,
  legend_title = NULL,
  color_vector = c("#FDDECE", "#F07A58", "#A8381E"),
  none_label="Pas saisonnier"
){
  
  # get the possible values of the duration; decreasing order, so that the most intense color is the shortest (most dramatic) type of seasonality
  unique_values <- sort(unique(as.character(spatial_seasonality_df[[seasonality_duration_colname]])), decreasing = TRUE)

  # check that they matche the number of colors
  if (length(color_vector) < length(unique_values)) {
    stop("The number of colors provided in 'color_vector' must be at least equal to the number of unique values in the data.")
  }

  # map colors to values
  color_mapping <- setNames(color_vector[1:length(unique_values)], unique_values)

  duration_plot <- ggplot(spatial_seasonality_df) +
    geom_sf(aes(fill = as.character(get(seasonality_duration_colname)))) +
    coord_sf() + # map projection
    scale_fill_manual(
      name = legend_title,
      values = color_mapping,
      labels = function(x) {
        ifelse(is.na(x) | x == "Inf", none_label, x) # custom labels
      },
      na.value="#D3D3D3"
    ) +
    
    # titles
    ggplot2::labs(
      title = plot_title,
      subtitle = plot_subtitle,
      caption = plot_caption
    ) +
    
    guides(fill = guide_legend(nrow = 2)) +

    # map theme
    ggplot2::theme_void() +
    ggplot2::theme(
        plot.title = ggplot2::element_text(
            face = "bold", size = 10,
            margin = ggplot2::margin(b = 4)
        ),
        plot.subtitle = ggplot2::element_text(
            size = 8, colour = "grey40",
            margin = ggplot2::margin(b = 8)
        ),
        plot.caption = ggplot2::element_text(
            size = 8,
            colour = "grey55",
            hjust = 1,
            margin = ggplot2::margin(t = 8)
        ),
        legend.position = "right",
        legend.text = ggplot2::element_text(size = 8),
        plot.margin = ggplot2::margin(10, 10, 10, 10)
    )

  return(duration_plot)
}



#' @description
#' case/rainfall proportion plot with custom color palette
#' 
#' @param plot_data a data frame containing the spatial data and a column for rainfall proportion
#' @param proportion_colname column containing the proportion of rainfall during the seasonal block
#' @param plot_title map title (defaults to NULL)
#' @param plot_subtitle map subtitle (defaults to NULL)
#' @param plot_caption map caption (defaults to NULL)
#' @param legend_title title of the legend (defaults to NULL)
#' @param proportion_colname column of the cases/rainfall proportion values; default is RAIN_PROPORTION
#'
#' @return a ggplot object or NULL if proportion_col is not found in plot_data
make_season_proportion_plot <- function(
    plot_data,
    proportion_colname,
    plot_title,
    plot_subtitle,
    plot_caption,
    legend_title,
    color_vector = c(
        "#C8DDD9",
        "#9DBFBB",
        "#5E9490",
        "#2E6460",
        "#264A48"
    )
) {
 
  #validate inputs
  if (!inherits(plot_data, "sf")) {
    stop("plot_data must be an sf object.")
  }
 
  if (!proportion_colname %in% names(plot_data)) {
    warning("The cases/rainfall proportion column not found in plot_data. Returning NULL.")
    return(NULL)
  }
 
  prop_vals <- plot_data[[proportion_colname]]
 
  if (!is.numeric(prop_vals)) {
    stop("The cases/rainfall proportion column must be numeric.")
  }
 
  # bin into five ordered categories
  bin_breaks <- c(-Inf, 0.20, 0.40, 0.60, 0.80, Inf)
  bin_labels <- c("<20%", "20 - 40%", "40 - 60%", "60 - 80%", ">80%")
 
  plot_data <- plot_data |>
    dplyr::mutate(
      .prop_rescaled = prop_vals,
      .rain_category = cut(
        .prop_rescaled,
        breaks         = bin_breaks,
        labels         = bin_labels,
        include.lowest = TRUE,
        right          = FALSE
      )
    )
 
  # make plot
  season_proportion_plot <- ggplot2::ggplot(data = plot_data) +
 
    # geo layer
    ggplot2::geom_sf(
      ggplot2::aes(fill = .rain_category),
      colour    = "white",
      linewidth = 0.15
    ) +
 
    # discrete colors and handle missings
    ggplot2::scale_fill_manual(
      values   = color_vector,
      breaks   = bin_labels,
      na.value = "#D3D3D3",
      name     = legend_title,
      drop     = TRUE,
      guide    = ggplot2::guide_legend(
        title          = NULL,
        override.aes   = list(colour = "white", linewidth = 0.3),
        label.position = "right",
        keywidth       = ggplot2::unit(0.9, "lines"),
        keyheight      = ggplot2::unit(0.9, "lines")
      )
    ) +
 
    # titles
    ggplot2::labs(
      title    = plot_title,
      subtitle = plot_subtitle,
      caption  = plot_caption
    ) +
 
    # map theme
    ggplot2::theme_void() +
    ggplot2::theme(
        plot.title = ggplot2::element_text(
            face = "bold", size = 10,
            margin = ggplot2::margin(b = 4)
        ),
        plot.subtitle = ggplot2::element_text(
            size = 8, colour = "grey40",
            margin = ggplot2::margin(b = 8)
        ),
        plot.caption = ggplot2::element_text(
            size = 8,
            colour = "grey55",
            hjust = 1,
            margin = ggplot2::margin(t = 8)
        ),
        legend.position = "right",
        legend.text = ggplot2::element_text(size = 8),
        plot.margin = ggplot2::margin(10, 10, 10, 10)
    )
 
  # add NA note to caption if any missing values in data
  if (anyNA(prop_vals)) {
    season_proportion_plot <- season_proportion_plot + ggplot2::labs(
      caption = paste0(data_source,
        if (nchar(data_source) > 0) "\n" else "",
        "\u25A0 Non saisonnier")
    )
  }
 
  return(season_proportion_plot)
}


#' @description
#' make a choropleth map from an sf object
#'
#' @param spatial_df sf object containing geometries and the variable map
#' @param target_colname name of the column to use for the fill aesthetic
#' @param map_colors vector of colors to scale_fill_manual with names that match the levels of the target variable
#' @param plot_title plot title
#' @param legend_title legend title
#' @return map
make_output_plot <- function(spatial_df, target_colname, map_colors, plot_title, legend_title){
  output_plot <- ggplot(spatial_df) +
    geom_sf(aes(fill = get(target_colname)))+
    coord_sf() +
    scale_fill_manual(
      legend_title,
      values=map_colors
    ) +
    guides(fill=guide_legend(nrow = 2)) +
    theme_void() +
    theme(
      plot.title = element_text(
        family = "Helvetica",
        # face = "bold",
        hjust = 0.5
      ),
      legend.position = "bottom", legend.key.width = unit(2,"cm"),
      legend.text=element_text(
        family = "Helvetica",
        size=10
      )
    ) +
    labs(title=plot_title)

  print(output_plot)
  return(output_plot)
 }