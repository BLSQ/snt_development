# Helpers for the reporting rate (data element) report

#' extract the routine data type from the saved pipeline parameters (raw = routine / imputed / removed)
get_routine_type <- function(input_filename){
    
    filename_without_extension <- tools::file_path_sans_ext(input_filename)
    filename_components <- strsplit(filename_without_extension, c("_", "."))[[1]]
    result <- tolower(filename_components[length(filename_components)])
    
    return(result)
}


#' extract the FOSAs expected to submit reports from the saved pipeline parameters (actives / ouvertes)
get_fosa_denominator <- function(input_string){

    string_components <- strsplit(input_string, c("_", "."))[[1]]

    result <- tolower(string_components[2])
    
    return(result)
}


#' @description make reporting rate scatter plot with ggplot2 faceted by year
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
#' @returns a ggplot object with scattered points by month and value colored by category and faceted by year
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


#' @description make reporting rate heatmap with ggplot2 tiles faceted by year
#' @param df_to_plot data frame containing the plotting data
#' @param admin_colname column with administrative unit identifier on y axis
#' @param admin_labels column with the administrative unit names for y axis labels
#' @param year_colname column with year values used in column faceting
#' @param month_colname column with month values on x axis
#' @param category_colname column with categorical variable mapped to fill color
#' @param plot_palette named list or vector of colors for categories
#' @param none_color hex color string for missing values
#' @param plot_title optional plot title
#' @param plot_subtitle optional plot subtitle
#' @param plot_caption optional plot caption
#' @param legend_title optional legend title
#' @param x_title optional x axis label
#' @param y_title optional y axis label
#' @returns a ggplot object with heat tiles by month and admin unit colored by category and faceted by year
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


#' @description make reporting rate maps faceted by year and month
#' @param df_to_plot data frame containing the plotting data
#' @param admin_colname column with administrative unit identifier
#' @param year_colname column with year values used in faceting
#' @param month_colname column with month values on x axis
#' @param category_colname column with categorical variable mapped to color
#' @param plot_palette named list or vector of colors for categories
#' @param none_color hex color string for missing values
#' @param plot_title optional plot title
#' @param plot_subtitle optional plot subtitle
#' @param plot_caption optional plot caption
#' @returns a ggplot object with maps of the category variable, faceted by year and month
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


#' @description make reporting rate maps faceted by year
#' @param df_to_plot data frame containing the plotting data
#' @param admin_colname column with administrative unit identifier
#' @param year_colname column with year values used in faceting
#' @param category_colname column with categorical variable mapped to color
#' @param plot_palette named list or vector of colors for categories
#' @param none_color hex color string for missing values
#' @param plot_title optional plot title
#' @param plot_subtitle optional plot subtitle
#' @param plot_caption optional plot caption
#' @returns a ggplot object with maps of the category variable, faceted by year
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


summarise_reporting_rate_by_year <- function(
  input_df,
  output_df = NULL,
  admin2_id_colname = "ADM2_ID",
  admin2_name_colname = "ADM2_NAME",
  admin1_id_colname = "ADM1_ID",
  admin1_name_colname = "ADM1_NAME",
  year_colname = "YEAR",
  reporting_rate_colname = "REPORTING_RATE") {

  result <- input_df %>%
    group_by(
      geometry,
      .data[[admin2_id_colname]],
      .data[[admin2_name_colname]],
      .data[[admin1_name_colname]],
      .data[[year_colname]]
    ) %>%
    summarise(
      "{reporting_rate_colname}" := mean(.data[[reporting_rate_colname]], na.rm = TRUE),
      .groups = "drop"
    )

  return(result)
}