# Helpers for the access to healthcare pipeline report

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