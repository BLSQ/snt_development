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

#' @description
#' cut numeric values into categorical bins based on a vector of breaks
#' 
#' return a data.table with the new category column
#' 
#' @param input_dt input data.table/frame with the data to process
#' @param input_colname the numeric column to categorize
#' @param output_colname nthe new column to create with category labels
#' @param input_breaks vector with the bin boundaries
#' @param num_decimals number of decimal places for rounding in labels (defaults to 1)
#' @param suffix string to add to the end of each label (defaults to "%" for percentages)
#' 
#' @returns data.table with original columns plus new category column
cut_to_categories <- function(input_dt, input_colname, output_colname, input_breaks, num_decimals = 1, suffix = "%") {

  if (!input_colname %in% names(input_dt)) {
      stop(paste("Input numeric column to cut not found in the input data"))
  }
    
  if (length(input_breaks) < 2) {
      stop("At least 2 breaks required to cut the input data")
  }
  
  # sort the input breaks
  breaks <- sort(input_breaks)

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


#' @description
#' choropleth map with unified SNT style
#'
#' @param input_data sf object with the geometries and data to plot
#' @param target_colname name of the numeric column to map
#' @param low_color color for the low end of the gradient (defaults to "#f7fbff").
#' @param high_color color for the high end of the gradient (defaults to "#08306b").
#' @param na_color color for missing values
#' @param plot_title map title (defaults to NULL)
#' @param plot_subtitle map subtitle (defaults to NULL)
#' @param plot_caption map caption (defaults to NULL)
#' @param legend_title title of the legend (defaults to NULL)
#'
#' @return ggplot object
make_snt_choropleth_map <- function(
    input_data,
    target_colname,
    low_color = "#f7fbff",
    high_color = "#08306b",
    na_color = "#D3D3D3",
    plot_title = NULL,
    plot_subtitle = NULL,
    plot_caption = NULL,
    legend_title = NULL
) {
  # validate inputs
  if (!inherits(input_data, "sf")) {
    stop("The data to plot must be an sf object.")
  }
  if (!target_colname %in% names(input_data)) {
    stop("The column to plot is not part of the data.")
  }
  if (!is.numeric(input_data[[target_colname]])) {
    stop("The column to plot must be numeric.")
  }
 
  # plot
  output_plot <- ggplot(data = input_data) +
    geom_sf(
      aes(fill = .data[[target_colname]]),
      color = "white",
      linewidth = 0.2
    ) +
    
    scale_fill_gradient(
      low = low_color,
      high = high_color,
      name = legend_title,
      na.value = na_color
    ) +
    
    # titles
    labs(
      title = plot_title,
      subtitle = plot_subtitle,
      caption = plot_caption
    ) +
 
    # map theme
    theme_void() +
    theme(
        plot.title = element_text(
            face = "bold", size = 10,
            margin = margin(b = 4)
        ),
        plot.subtitle = element_text(
            size = 8, colour = "grey40",
            margin = margin(b = 8)
        ),
        plot.caption = element_text(
            size = 8,
            colour = "grey55",
            hjust = 1,
            margin = margin(t = 8)
        ),
        legend.position = "right",
        legend.text = element_text(size = 8),
        plot.margin = margin(10, 10, 10, 10)
    )

  return(output_plot)
}


#' make percentage map (choropleth)
#'
#' @param map_data spatial df with attribute data
#' @param target_colname column to plot
#' @param plot_title map title (defaults to NULL)
#' @param plot_subtitle map subtitle (defaults to NULL)
#' @param plot_caption map caption (defaults to NULL)
#' @param legend_title title of the legend (defaults to NULL)
#' @param low_color color for the low end of the gradient (defaults to white)
#' @param high_color color for the high end of the gradient (defaults to very dark green)
#' @param na_color color for missing values (defaults to "#D3D3D3")
#' @param scale_range vector of minimum and maximum values for the percentage (defaults to 0-100)
#' @return ggplot object of the map
make_pct_choropleth_map <- function(map_data, target_colname, plot_title = NULL, plot_subtitle = NULL, plot_caption = NULL, legend_title = NULL, low_color = "white", high_color = "#1B3150", na_color = "#D3D3D3", scale_range = c(0, 100)) {
  
   # validate inputs
  if (!inherits(map_data, "sf")) {
    stop("The data to plot must be an sf object.")
  }
  if (!target_colname %in% names(map_data)) {
    stop("The column to plot is not part of the data.")
  }
  if (!is.numeric(map_data[[target_colname]])) {
    stop("The column to plot must be numeric.")
  }

  if(min(map_data[[target_colname]]) < 0 | max(map_data[[target_colname]]) > 100){
    stop("The column to plot is not a correct percentage.")
  }
  
  plot <- ggplot(map_data) +
    geom_sf(
      aes(fill = .data[[target_colname]]),
      color = "white",
      linewidth = 0.2
    ) +
    coord_sf() +
    scale_fill_gradient(
      limits = scale_range,
      low = low_color,
      high = high_color,
      name = legend_title,
      na.value = na_color
    ) +
    
    # titles
    labs(
      title    = plot_title,
      subtitle = plot_subtitle,
      caption  = plot_caption
    ) +
 
    # map theme
    theme_void() +
    theme(
        plot.title = element_text(
            face = "bold", size = 10,
            margin = margin(b = 4)
        ),
        plot.subtitle = element_text(
            size = 8, colour = "grey40",
            margin = margin(b = 8)
        ),
        plot.caption = element_text(
            size = 8,
            colour = "grey55",
            hjust = 1,
            margin = margin(t = 8)
        ),
        legend.position = "right",
        legend.text = element_text(size = 8),
        plot.margin = margin(10, 10, 10, 10)
    )
  
  return(plot)
}

#' make permil map (choropleth)
#'
#' @param map_data spatial df with attribute data
#' @param target_colname column to plot
#' @param scale_range the minimum and maximum values of the scale (defaults to 0-200)
#' @param plot_title map title (defaults to NULL)
#' @param plot_subtitle map subtitle (defaults to NULL)
#' @param plot_caption map caption (defaults to NULL)
#' @param legend_title title of the legend (defaults to NULL)
#' @param low_color color for the low end of the gradient (defaults to white)
#' @param high_color color for the high end of the gradient (defaults to very dark green)
#' @param na_color color for missing values (defaults to "#D3D3D3")
#' @return ggplot object of the map
make_permil_choropleth_map <- function(map_data, target_colname, scale_range = c(0, 200), plot_title = NULL, plot_subtitle = NULL, plot_caption = NULL, legend_title = NULL, low_color = "white", high_color = "#1B3150", na_color = "#D3D3D3") {
  
   # validate inputs
  if (!inherits(map_data, "sf")) {
    stop("The data to plot must be an sf object.")
  }
  if (!target_colname %in% names(map_data)) {
    stop("The column to plot is not part of the data.")
  }
  if (!is.numeric(map_data[[target_colname]])) {
    stop("The column to plot must be numeric.")
  }

  plot <- ggplot(map_data) +
    geom_sf(
      aes(fill = .data[[target_colname]]),
      color = "white",
      linewidth = 0.2
    ) +
    coord_sf() +
    scale_fill_gradient(
      limits = scale_range,
      low = low_color,
      high = high_color,
      name = legend_title,
      na.value = na_color
    ) +
    
    # titles
    labs(
      title    = plot_title,
      subtitle = plot_subtitle,
      caption  = plot_caption
    ) +
 
    # map theme
    theme_void() +
    theme(
        plot.title = element_text(
            face = "bold", size = 10,
            margin = margin(b = 4)
        ),
        plot.subtitle = element_text(
            size = 8, colour = "grey40",
            margin = margin(b = 8)
        ),
        plot.caption = element_text(
            size = 8,
            colour = "grey55",
            hjust = 1,
            margin = margin(t = 8)
        ),
        legend.position = "right",
        legend.text = element_text(size = 8),
        plot.margin = margin(10, 10, 10, 10)
    )
  
  return(plot)
}


#' @description
#' categorical map with unified SNT style
#'
#' @param input_data sf object with the geometries and data to plot
#' @param target_colname name of the factor column to map
#' @param color_vector vector of colors to use
#' @param na_color color for missing values
#' @param plot_title map title (defaults to NULL)
#' @param plot_subtitle map subtitle (defaults to NULL)
#' @param plot_caption map caption (defaults to NULL)
#' @param legend_title title of the legend (defaults to NULL)
#'
#' @return ggplot object
make_snt_categorical_map <- function(
  input_data,
  target_colname,
  color_vector,
  na_color = "#D3D3D3",
  plot_title = NULL,
  plot_subtitle = NULL,
  plot_caption = NULL,
  legend_title = NULL
) {
 
  # validation
 
  if (!is.data.frame(input_data)) {
    stop("Data to plot must be a sf data frame.")
  }
 
  if (!target_colname %in% names(input_data)) {
    stop("Target column should be part of the data to plot.")
  }
  
  if(!is.factor(input_data[[target_colname]])){
	stop("Target column should be an ordered factor.")
  }
 
  if (!is.character(color_vector)) {
    stop("The vector of colors is not valid.")
  }
 
  output_plot <- ggplot(data = input_data) +
    geom_sf(aes(fill = .data[[target_colname]]), color = "white", linewidth = 0.2) +
    scale_fill_manual(
      values  = color_vector,
      na.value = na_color,
      name    = legend_title,
      drop    = FALSE # show all levels
    ) +
    
    # titles
    labs(
      title = plot_title,
      subtitle = plot_subtitle,
      caption = plot_caption
    ) +
 
    # map theme
    theme_void() +
    theme(
        plot.title = element_text(
            face = "bold", size = 10,
            margin = margin(b = 4)
        ),
        plot.subtitle = element_text(
            size = 8, colour = "grey40",
            margin = margin(b = 8)
        ),
        plot.caption = element_text(
            size = 8,
            colour = "grey55",
            hjust = 1,
            margin = margin(t = 8)
        ),
        legend.position = "right",
        legend.text = element_text(size = 8),
        plot.margin = margin(10, 10, 10, 10)
    )
  
  return(output_plot)
}

#' @description create a reporting rate scatter plot with ggplot2 faceted by year
#' @param df_to_plot data frame containing the plotting data
#' @param admin_colname column with administrative unit identifier
#' @param year_colname column with year values used in faceting
#' @param month_colname column with month values on x axis
#' @param value_colname column with numeric values on y axis
#' @param category_colname column with categorical variable mapped to color
#' @param break_vector numeric vector of breaks for y axis ticks
#' @param plot_palette named list or vector of colors for categories
#' @param none_color hex color string for missing values
#' @param plot_title optional plot title
#' @param plot_subtitle optional plot subtitle
#' @param plot_caption optional plot caption
#' @param x_title optional x axis label
#' @param y_title optional y axis label
#' @returns a ggplot object displaying scattered points by month and value colored by category and faceted by year
make_reporting_rate_scatterplot <- function(
  df_to_plot, admin_colname, year_colname, month_colname, value_colname, category_colname, break_vector, plot_palette, none_color = "#D3D3D3", plot_title = NULL, plot_subtitle = NULL, plot_caption = NULL, x_title = NULL, y_title = NULL
){

  output_plot <- ggplot(data = df_to_plot) +
    geom_point(
      aes(
        x = get(month_colname),
        y = get(value_colname),
        group = get(admin_colname),
        color = get(category_colname))
      ) + 
    facet_grid(~get(year_colname)) + 
    scale_color_manual(
      values = plot_palette,
      na.value = none_color
      ) +
    scale_x_continuous(breaks = seq(1, 12, 1)) +
    scale_y_continuous(
      breaks = round(break_vector, 1),
  
      limits = c(0, max(df_to_plot[[value_colname]], na.rm = TRUE) + 0.1)
    ) +
    labs(
      title = plot_title,
      subtitle = plot_subtitle,
      caption = plot_caption,
      x = x_title,
      y =  y_title
      ) +
    theme_minimal() +
    theme(
      plot.title = element_text(
        face = "bold", size = 10,
        margin = margin(b = 4)
      ),
      plot.subtitle = element_text(
        size = 8, colour = "grey40",
        margin = margin(b = 8)
      ),
      plot.caption = element_text(
        size = 8,
        colour = "grey55",
        hjust = 1,
        margin = margin(t = 8)
      ),
      axis.title.x = element_text(
        colour = "grey55",
        size = 8
      ),
      axis.title.y = element_text(
        colour = "grey55",
        size = 8
      ),
      legend.position = "none",
      legend.title = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      strip.placement = "outside",
      strip.text = element_text(face = "bold", size = 8, colour = "grey40",),
      plot.margin = margin(10, 10, 10, 10)
    )

  return(output_plot)
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
#' @param scale_constant scale for the divisor, to limit the heights of the ridges
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
  scale_constant = 2,
  ridge_color = "#008080",
  reorder_fun = mean
) {

  # prep vectors of breaks and labels
  x_breaks <- dt[get(month_colname) == 1, get(x_colname)]
  x_labels <- dt[get(month_colname) == 1, get(year_colname)]

  # make the plot
  ridge_plot <- ggplot(
    dt,
    aes(
      x = .data[[x_colname]],
      y = fct_reorder(.data[[y_colname]], .data[[height_colname]], reorder_fun),
      height = .data[[height_colname]] / (max(.data[[height_colname]]) * scale_constant),
      group = .data[[y_colname]]
    )
  ) +
  geom_ridgeline(
    alpha = 0.4, scale = 4.5, linewidth = 0.2,
    fill = ridge_color, color = "white"
  ) +
  scale_x_discrete(
    breaks = x_breaks,
    labels = x_labels
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
    plot.title = element_text(
        face = "bold", size = 10,
        margin = margin(b = 4)
    ),
    plot.subtitle = element_text(
        size = 8, colour = "grey40",
        margin = margin(b = 8)
    ),
    plot.caption = element_text(
        size = 8,
        colour = "grey55",
        hjust = 1,
        margin = margin(t = 8)
    ),
    plot.margin = margin(10, 10, 10, 10)
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
    labs(
      title = plot_title,
      subtitle = plot_subtitle,
      caption = plot_caption
    ) +

    guides(fill=guide_legend(nrow = 2)) +
    
    # map theme
    theme_void() +
    theme(
        plot.title = element_text(
            face = "bold", size = 10,
            margin = margin(b = 4)
        ),
        plot.subtitle = element_text(
            size = 8, colour = "grey40",
            margin = margin(b = 8)
        ),
        plot.caption = element_text(
            size = 8,
            colour = "grey55",
            hjust = 1,
            margin = margin(t = 8)
        ),
        legend.position = "right",
        legend.text = element_text(size = 8),
        plot.margin = margin(10, 10, 10, 10)
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
  season_first_month <- ggplot(data = plot_data) +
 
    # geo layer
    geom_sf(
      aes(fill = .month_factor),
      colour = "white",
      linewidth = 0.15
    ) +
 
    # discrete color scale with NA handling
    scale_fill_manual(
      values = scale_values,
      breaks = scale_breaks,
      labels = scale_labels,
      na.value = "#D3D3D3",
      name = legend_title,
      guide = guide_legend(
        title = legend_title,
        override.aes = list(colour = "white", linewidth = 0.3),
        label.position = "right",
        keywidth = unit(0.9, "lines"),
        keyheight = unit(0.9, "lines")
      )
    ) +
 
    # titles
    labs(
      title = plot_title,
      subtitle = plot_subtitle,
      caption = plot_caption
    ) +
 
    # map theme
    theme_void() +
    theme(
        plot.title = element_text(
            face = "bold", size = 10,
            margin = margin(b = 4)
        ),
        plot.subtitle = element_text(
            size = 8, colour = "grey40",
            margin = margin(b = 8)
        ),
        plot.caption = element_text(
            size = 8,
            colour = "grey55",
            hjust = 1,
            margin = margin(t = 8)
        ),
        legend.position = "right",
        legend.text = element_text(size = 8),
        plot.margin = margin(10, 10, 10, 10)
    )
 
  # append NA note to caption if any missing values exist in the data, so they also appear in the legend
  if (anyNA(month_vals)) {
    season_first_month <- season_first_month + labs(
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
    labs(
      title = plot_title,
      subtitle = plot_subtitle,
      caption = plot_caption
    ) +
    
    guides(fill = guide_legend(nrow = 2)) +

    # map theme
    theme_void() +
    theme(
        plot.title = element_text(
            face = "bold", size = 10,
            margin = margin(b = 4)
        ),
        plot.subtitle = element_text(
            size = 8, colour = "grey40",
            margin = margin(b = 8)
        ),
        plot.caption = element_text(
            size = 8,
            colour = "grey55",
            hjust = 1,
            margin = margin(t = 8)
        ),
        legend.position = "right",
        legend.text = element_text(size = 8),
        plot.margin = margin(10, 10, 10, 10)
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
  season_proportion_plot <- ggplot(data = plot_data) +
 
    # geo layer
    geom_sf(
      aes(fill = .rain_category),
      colour    = "white",
      linewidth = 0.15
    ) +
 
    # discrete colors and handle missings
    scale_fill_manual(
      values   = color_vector,
      breaks   = bin_labels,
      na.value = "#D3D3D3",
      name     = legend_title,
      drop     = TRUE,
      guide    = guide_legend(
        title          = NULL,
        override.aes   = list(colour = "white", linewidth = 0.3),
        label.position = "right",
        keywidth       = unit(0.9, "lines"),
        keyheight      = unit(0.9, "lines")
      )
    ) +
 
    # titles
    labs(
      title    = plot_title,
      subtitle = plot_subtitle,
      caption  = plot_caption
    ) +
 
    # map theme
    theme_void() +
    theme(
        plot.title = element_text(
            face = "bold", size = 10,
            margin = margin(b = 4)
        ),
        plot.subtitle = element_text(
            size = 8, colour = "grey40",
            margin = margin(b = 8)
        ),
        plot.caption = element_text(
            size = 8,
            colour = "grey55",
            hjust = 1,
            margin = margin(t = 8)
        ),
        legend.position = "right",
        legend.text = element_text(size = 8),
        plot.margin = margin(10, 10, 10, 10)
    )
 
  # add NA note to caption if any missing values in data
  if (anyNA(prop_vals)) {
    season_proportion_plot <- season_proportion_plot + labs(
      caption = paste0(data_source,
        if (nchar(data_source) > 0) "\n" else "",
        "\u25A0 Non saisonnier")
    )
  }
 
  return(season_proportion_plot)
}


#' Make confidence interval plots (horizontal bar chart with error bars)
#'
#' @param df_to_plot data.frame with columns for the administrative unit, point estimate and confidence intervals
#' @param admin_colname column for the administrative unit identifiers (used as x-axis)
#' @param point_estimation_colname column with the point estimates for the main bars
#' @param ci_lower_colname column with the lower bound values for the confidence intervals
#' @param ci_upper_colname column with the upper bound values for the confidence intervals
#' @param bar_color color for the bars (defaults to muted teal-blue)
#' @param errorbar_color color for the error bars (defaults to periwinkle blue)
#' @param plot_title plot title (defaults to NULL)
#' @param plot_subtitle plot subtitle (defaults to NULL)
#' @param plot_caption plot caption (defaults to NULL)
#' @param x_title x-axis label (defaults to NULL)
#' @param y_title y-axis label (defaults to NULL)
#' 
#'
#' @return ggplot2 bar chart with confidence interval error bars (also printed)
#'
make_ci_plot <- function(
  df_to_plot, admin_colname, point_estimation_colname, ci_lower_colname, ci_upper_colname,
  bar_color = "#5B8CBE", errorbar_color = "#2D5F8A", plot_title = NULL, plot_subtitle = NULL, plot_caption = NULL, x_title = NULL, y_title = NULL
){
  
  # main plot
  ci_plot <- ggplot(data = df_to_plot)
  ci_plot <- ci_plot + geom_bar(aes(x=get(admin_colname), y=get(point_estimation_colname)), fill = bar_color, alpha = 0.7, width = 0.6, stat="identity")

  # add the error bars
  ci_plot <- ci_plot + geom_errorbar(
    aes(x=get(admin_colname), ymin = get(ci_lower_colname), ymax = get(ci_upper_colname)),
    width = 0.25,
    color = errorbar_color,
    linewidth = 1
  )
  ## Uncomment below to add value label text for the lower bound
  # + ci_plot <- ci_plot + geom_text(aes(
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

  # spacing
  ci_plot <- ci_plot +
    scale_x_discrete(expand = expansion(mult = c(0.05, 0.05))) +
    scale_y_continuous(expand = expansion(mult = c(0.02, 0.05))) +

  # titles
  labs(
      title = plot_title,
      subtitle = plot_subtitle,
      caption = plot_caption,
      x = x_title,
      y = y_title
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
    plot.title = element_text(
        face = "bold", size = 10,
        margin = margin(b = 4)
    ),
    plot.subtitle = element_text(
        size = 8, colour = "grey40",
        margin = margin(b = 8)
    ),
    plot.caption = element_text(
        size = 8,
        colour = "grey55",
        hjust = 1,
        margin = margin(t = 8)
    ),
    plot.margin = margin(10, 10, 10, 10)
  )

  ci_plot <- ci_plot + coord_flip()
  
  return(ci_plot)
}