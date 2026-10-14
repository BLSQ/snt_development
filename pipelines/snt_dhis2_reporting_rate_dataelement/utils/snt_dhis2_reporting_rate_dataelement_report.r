# Helpers for the reporting rate (data element) report

#' Get the Routine Data Type
#'
#' Extracts the routine data type from the routine filename saved in the
#' pipeline parameters, as the last underscore-separated part of the name
#' without extension: "routine" (raw), "imputed" or "removed".
#'
#' @param input_filename Character. Routine filename, e.g. "NER_routine_outliers_imputed.parquet".
#' @return Character. Lowercase routine data type.
#'
#' @export
get_routine_type <- function(input_filename){
    
    filename_without_extension <- tools::file_path_sans_ext(input_filename)
    filename_components <- strsplit(filename_without_extension, c("_", "."))[[1]]
    result <- tolower(filename_components[length(filename_components)])
    
    return(result)
}


#' Get the Facility Denominator Type
#'
#' Extracts which facilities (FOSA) are expected to report from the
#' denominator method saved in the pipeline parameters, as the second
#' underscore-separated part of the method name: "active" or "open".
#'
#' @param input_string Character. Denominator method, e.g. "ROUTINE_ACTIVE_FACILITIES".
#' @return Character. Lowercase denominator type.
#'
#' @export
get_fosa_denominator <- function(input_string){

    string_components <- strsplit(input_string, c("_", "."))[[1]]

    result <- tolower(string_components[2])
    
    return(result)
}


#' Make Reporting Rate Scatter Plot
#'
#' Builds a ggplot2 scatter plot of reporting rate values by month, with one
#' point per administrative unit, colored by category and faceted by year.
#'
#' @param df_to_plot Data frame. Data to plot.
#' @param admin_colname Character. Column with the administrative unit identifier.
#' @param year_colname Character. Column with the year values used for faceting.
#' @param month_colname Character. Column with the month values on the x axis.
#' @param value_colname Character. Column with the numeric values on the y axis.
#' @param category_colname Character. Column with the category mapped to color.
#' @param break_vector Numeric. Breaks used for the y axis ticks.
#' @param plot_palette Character. Named vector (or list) of colors for the categories.
#' @param none_color Character. Hex color for missing values. Default: "#D3D3D3".
#' @param plot_title Character. Optional plot title.
#' @param plot_subtitle Character. Optional plot subtitle.
#' @param plot_caption Character. Optional plot caption.
#' @param x_title Character. Optional x axis label.
#' @param y_title Character. Optional y axis label.
#' @return ggplot. Scatter plot of values by month, colored by category and faceted by year.
#'
#' @export
make_reporting_rate_scatterplot <- function(
  df_to_plot, admin_colname, year_colname, month_colname, value_colname, category_colname, break_vector, plot_palette, none_color = "#D3D3D3", plot_title = NULL, plot_subtitle = NULL, plot_caption = NULL, x_title = NULL, y_title = NULL
){

  output_plot <- ggplot(data = df_to_plot) +
    geom_point(
      aes(
        x = get(month_colname),
        y = get(value_colname),
        group = get(admin_colname),
        color = get(category_colname)
      )
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
      axis.text = element_text(size = 8),
      legend.position = "none",
      legend.title = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      strip.placement = "outside",
      strip.text = element_text(size = 8, colour = "grey40"),
      plot.margin = margin(10, 10, 10, 10)
    )

  return(output_plot)
}


#' Make Reporting Rate Heatmap
#'
#' Builds a ggplot2 heatmap with one tile per administrative unit and month,
#' filled by category and faceted by year. The y axis shows the administrative
#' unit names.
#'
#' @param df_to_plot Data frame. Data to plot.
#' @param admin_colname Character. Column with the administrative unit identifier on the y axis.
#' @param admin_labels Character. Column with the administrative unit names used as y axis labels.
#' @param year_colname Character. Column with the year values used for column faceting.
#' @param month_colname Character. Column with the month values on the x axis.
#' @param category_colname Character. Column with the category mapped to fill color.
#' @param plot_palette Character. Named vector (or list) of colors for the categories.
#' @param none_color Character. Hex color for missing values. Default: "#D3D3D3".
#' @param plot_title Character. Optional plot title.
#' @param plot_subtitle Character. Optional plot subtitle.
#' @param plot_caption Character. Optional plot caption.
#' @param legend_title Character. Optional legend title.
#' @param x_title Character. Optional x axis label.
#' @param y_title Character. Currently unused: the y axis title is hidden.
#' @return ggplot. Heatmap of categories by month and administrative unit, faceted by year.
#'
#' @export
make_reporting_rate_heatmap <- function(
    df_to_plot, admin_colname, admin_labels, year_colname, month_colname, category_colname, plot_palette, none_color = "#D3D3D3", plot_title = NULL, plot_subtitle = NULL, plot_caption = NULL, legend_title = NULL, x_title = NULL, y_title = NULL
){

  # build a lookup: admin_colname value -> admin_labels value
  label_lookup <- df_to_plot %>%
    dplyr::distinct(.data[[admin_colname]], .data[[admin_labels]]) %>%
    tibble::deframe()

ggplot(data = df_to_plot) +
  geom_tile(
      aes(x = get(month_colname),
                y = fct_rev(get(admin_colname)),
                fill = get(category_colname)
               ), 
                color = "white",
                 show.legend = TRUE
                 ) +
  scale_fill_manual(
      values = plot_palette,
      na.value = none_color,
      name = legend_title
    ) +
  scale_x_continuous(breaks = seq(1, 12, 1)) +
  scale_y_discrete(labels = label_lookup) +
  labs(
      title = plot_title,
      subtitle = plot_subtitle,
      caption = plot_caption,
      x = x_title
      ) +
  facet_grid(cols = vars(get(year_colname)), 
    scales = "free_y", space = "free_y",
    switch = "y") +
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
    legend.position = "bottom",
    legend.key.height = unit(0.25, "cm"),
    axis.text.x = element_text(size = 7),
    axis.title.y = element_blank(),
    axis.text.y = element_text(size = 5),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    strip.placement = "outside",    
    strip.text = element_text(face = "bold", size = 8, colour = "grey40"),
    plot.margin = margin(10, 10, 10, 10)
  ) +
  guides(fill = guide_legend(nrow = 1))
}


#' Make Monthly Reporting Rate Map
#'
#' Builds ggplot2 maps of the reporting rate category per administrative unit,
#' with one map per month (columns) and year (rows).
#'
#' @param df_to_plot Data frame. Data to plot, including a `geometry` column.
#' @param admin_colname Character. Column with the administrative unit identifier. Currently unused.
#' @param year_colname Character. Column with the year values used for row faceting.
#' @param month_colname Character. Column with the month values used for column faceting.
#' @param category_colname Character. Column with the category mapped to fill color.
#' @param plot_palette Character. Named vector (or list) of colors for the categories.
#' @param none_color Character. Hex color for missing values.
#' @param plot_title Character. Plot title.
#' @param plot_subtitle Character. Plot subtitle.
#' @param plot_caption Character. Plot caption.
#' @return ggplot. Maps of the category, faceted by year and month.
#'
#' @export
make_monthly_reporting_map <- function(
    df_to_plot, admin_colname, year_colname, month_colname, category_colname, plot_palette, none_color, plot_title, plot_subtitle, plot_caption
){
  output_plot <- ggplot(data = df_to_plot) +
    geom_sf(
      aes(fill = get(category_colname), geometry = geometry),
      color = "white",
      size = 0.01) +
    scale_fill_manual(
      values = plot_palette,
      na.value = none_color
    ) +
    
    labs(
      title = plot_title,
      subtitle = plot_subtitle,
      caption = plot_caption
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
      legend.title = element_blank(),
      legend.position = "bottom",
      panel.grid = element_blank(),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      strip.placement = "outside",
      strip.text = element_text(face = "bold", size = 8, colour = "grey40"),
      plot.margin = margin(10, 10, 10, 10)
    ) +
    
    # facet by month and year   
    facet_grid(
      rows = vars(get(year_colname)), 
      cols = vars(get(month_colname)),
      switch = "both") +
      guides(fill = guide_legend(nrow = 1))

  return(output_plot)
}


#' Make Yearly Reporting Rate Map
#'
#' Builds ggplot2 maps of the reporting rate category per administrative unit,
#' with one map per year (columns).
#'
#' @param df_to_plot Data frame. Data to plot, including a `geometry` column.
#' @param admin_colname Character. Column with the administrative unit identifier. Currently unused.
#' @param year_colname Character. Column with the year values used for column faceting.
#' @param category_colname Character. Column with the category mapped to fill color.
#' @param plot_palette Character. Named vector (or list) of colors for the categories.
#' @param none_color Character. Hex color for missing values.
#' @param plot_title Character. Plot title.
#' @param plot_subtitle Character. Plot subtitle.
#' @param plot_caption Character. Plot caption.
#' @return ggplot. Maps of the category, faceted by year.
#'
#' @export
make_yearly_reporting_map <- function(
    df_to_plot, admin_colname, year_colname, category_colname, plot_palette, none_color, plot_title, plot_subtitle, plot_caption
){
  output_plot <- ggplot(data = df_to_plot) +
    geom_sf(
      aes(fill = get(category_colname), geometry = geometry),
      color = "white",
      size = 0.01) +
    scale_fill_manual(
      values = plot_palette,
      na.value = none_color
    ) +
    
    labs(
      title = plot_title,
      subtitle = plot_subtitle,
      caption = plot_caption
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
      legend.title = element_blank(),
      legend.position = "bottom",
      panel.grid = element_blank(),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      strip.placement = "outside",
      strip.text = element_text(face = "bold", size = 8, colour = "grey40"),
      plot.margin = margin(10, 10, 10, 10)
    ) +
    
    # facet by year only (columns)
    facet_grid(
      cols = vars(get(year_colname)),
      switch = "both") +
      guides(fill = guide_legend(nrow = 1))
  return(output_plot)
}
