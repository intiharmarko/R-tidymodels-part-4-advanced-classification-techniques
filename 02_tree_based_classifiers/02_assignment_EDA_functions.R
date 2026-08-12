#' Assignment - Bank Marketing - EDA - Functions


#' Load packages
library(tidyverse)


#' Missing rows % count
#'
#' This function draws a plot showing percentage of missing rows per each variable (column).
#' 
#' @param df_ A data frame for given data.
#'
#' @return NULL
#'
rows_miss_count <- function(df_){
  
  map(.x = df_,
      .f = ~sum(is.na(.))) %>% 
    unlist() %>% 
    tibble(col = names(.),
           nr_missing = .,
           per_missing = round(nr_missing / nrow(df) * 100, 2)) %>% 
    ggplot(aes(y = col,
               x = per_missing,
               color = per_missing)) +
    geom_point(size = 4) +
    scale_x_continuous(breaks = seq(0,100,10),
                       limits = c(0, 100)) +
    scale_color_viridis_c(option = "magma") +
    labs(title = "Percentage of missing rows by column",
         color = "% missing rows:") +
    xlab("Percentage of missing rows") +
    ylab("Column") +
    theme_minimal(base_size = 16)
}



#' Count number of categories (levels) of categorical columns
#'
#' This function counts number of different categories of each factor, and draws 
#' a bar plot of factors and their number of different categories. 
#' We can control max number of levels to be shown for factor variables on the plot.
#' 
#' @param df_ A data frame for given data.
#' @param n_max An integer - max number of levels / categories for factor variable to be drawn on plot.
#'
#' @return NULL
#'
nr_categories_count <- function(df_,
                                n_max = nrow(df_)){
  
  # convert characters to factors and count categories
  df.counts <- df_ %>% 
    mutate(across(.cols = where(is.factor), 
                  .fns = as.factor)) %>% 
    select(where(is.factor)) %>% 
    map_int(., nlevels) %>%
    enframe(name = "col",
            value = "n") %>%
    arrange(n)
  
  # draw categories counts
  df.counts %>% 
    filter(n <= n_max) %>% 
    ggplot(aes(x = n,
               y = fct_inorder(col))) +
    geom_col(color = "black",
             fill = "gray80") +
    ggtitle("Factor features - unique levels count") +
    xlab("Number of unique levels (categories)") +
    ylab("Feature") +
    theme_minimal(base_size = 16)
}  


#' Target variable distribution
#'
#' This function draws distribution of target variable 
#' "y" (bank term deposit subscription) - bar plot.
#' 
#' @param df_ A data frame for given data.
#'
#' @return NULL
#'
target_var_distr <- function(df_){
  
  df_ %>% 
    count(y) %>% 
    mutate(per = round(n / sum(n) * 100, 2),
           label = paste0(n, " | ", per, "%")) %>% 
    ggplot(aes(x = y,
               y = n,
               label = label,
               fill = y)) +
    geom_col(color = "black") +
    geom_text(size = 6) +
    scale_fill_manual(values = c("brown2", "gray70")) +
    ggtitle("Target varible (bank term deposit) barplot") +
    xlab("Term deposit (1 - subscribed, 0 - not subscribed)") +
    ylab("Client count") +
    labs(fill = "Term deposit:") +
    theme_minimal(base_size = 16)
}


#' Plot bar plot - factor variable distribution
#'
#' This function draws distribution of selected factor variable using bar plot.
#' 
#' @param df_ A data frame for given data.
#' @param x_var A string - name of variable we are plotting.
#' @param x_font_size An integer indicating the size of fonts on x axis (ticks).
#'
#' @return NULL
#'
plot_fct_dist <- function(df_, 
                          x_var,
                          x_font_size = 12) {
  
  # calculate counts per each level
  plot_data <- df_ %>% 
    count(.data[[x_var]], 
          name = "n") %>% 
    arrange(desc(n)) %>% 
    mutate(per = n / sum(n),
           per_csum = cumsum(per))
  
  print(plot_data)
  
  # draw plot
  ggplot(plot_data,
         aes(x = fct_inorder(.data[[x_var]]), 
             y = n)) +
    geom_col(color = "black", 
             fill = "gray80") +
    labs(title = paste0("Factor feature (", x_var, ") distribution – bar plot"),
         x = NULL,
         y = "Counts") +
    theme_minimal(base_size = 16)
}


#' Plot histogram - numeric variable distribution
#'
#' This function draws distribution of selected numeric variable using histogram.
#' 
#' @param df_ A data frame for given data.
#' @param x_var A string - name of variable we are plotting.
#' @param bins_ An integer - number of bins in histogram.
#'
#' @return NULL
#'
plot_histogram <- function(df_, 
                           x_var, 
                           bins_ = 30) {
  
  ggplot(df_, 
         aes(x = .data[[x_var]])) + 
    geom_histogram(fill = "gray90",
                   color = "black",
                   bins = bins_) +
    ggtitle(paste0("Numeric feature (", x_var, ") distribution -  histrogram")) +
    xlab("") +
    ylab("Counts") +
    theme_minimal(base_size = 16)
}


#' Plot density plot - numeric variable distribution
#'
#' This function draws distribution of selected numeric variable using density plot.
#' 
#' @param df_ A data frame for given data.
#' @param x_var A string - name of variable we are plotting.
#'
#' @return NULL
#'
plot_density <- function(df_, 
                         x_var) {
  
  ggplot(df_, 
         aes(x = .data[[x_var]])) + 
    geom_density(fill = "gray90",
                 color = "black") +
    ggtitle(paste0("Numeric feature (", x_var, ") distribution -  density plot")) +
    xlab("") +
    ylab("Density") +
    theme_minimal(base_size = 16)
}


#' Plot target variable VS factor feature distribution - bar plot
#'
#' This function draws distribution of selected target factor column 
#' ("y" - subscribed to bank term deposit) and factor 
#' feature variable using bar plot. 
#' 
#' @param df_ A data frame for given data.
#' @param x_var A string - name of factor feature variable we are plotting.
#' @param x_font_size An integer indicating the size of fonts on x axis (ticks).
#'
#' @return NULL
#'
plot_tar_cat_feat_dist <- function(df_, 
                                   x_var,
                                   x_font_size = 8) {
  
  df_ %>% 
    ggplot(aes(x = .data[[x_var]],
               fill = y)) +
    geom_bar(position = "fill",
             color = "black") +
    scale_y_continuous(labels = scales::percent) +
    scale_fill_manual(values = c("brown2", "gray70")) +
    xlab(paste0("Categorical feature (", x_var, ") - category")) +
    ylab("Percentage of clients") +
    ggtitle(paste0("Target varible (y - bank term deposit) VS ", x_var, " - bar plot")) +
    labs(fill = "Bank term deposit:") +
    theme_minimal(base_size = 16)
}  


#' Plot target variable VS numeric feature distribution - box plot
#'
#' This function draws distribution of selected target factor column 
#' ("y" - subscribed to bank term deposit) and numeric 
#' feature variable using box plot. 
#' 
#' @param df_ A data frame for given data.
#' @param y_var A string - name of numeric feature variable we are plotting.
#' @param x_font_size An integer indicating the size of fonts on x axis (ticks).
#'
#' @return NULL
#'
plot_tar_num_feat_dist <- function(df_, 
                                   y_var,
                                   x_font_size = 8) {
  
  df_ %>% 
    ggplot(aes(x = y,
               y = .data[[y_var]],
               fill = y)) +
    geom_boxplot() +
    scale_fill_manual(values = c("brown2", "gray70")) +
    xlab("Bank term deposit subscription (1 - yes, 0 - no)") +
    ylab(paste0("Numeric feature (", y_var, ") value")) +
    ggtitle(paste0("Target varible (bank term deposit) VS ", y_var, " - box plot")) +
    labs(fill = "Bank term deposit:") +
    theme_minimal(base_size = 16) 
}  



#' Plot target variable VS numeric feature distribution - bar plot
#'
#' This function draws distribution of selected target factor column 
#' ("y" - subscribed to bank term deposit) and numeric 
#' feature variable, for which we first create counts of each numeric value. 
#' 
#' @param df_ A data frame for given data.
#' @param y_var A string - name of numeric feature variable we are plotting.
#' @param x_font_size An integer indicating the size of fonts on x axis (ticks).
#'
#' @return NULL
#'
plot_tar_num_feat_bar_dist <- function(df_, 
                                       y_var,
                                       x_font_size = 8) {
  
  df %>% 
    group_by(y, .data[[y_var]]) %>% 
    summarise(n = n(), 
              .groups = "drop") %>% 
    ggplot(aes(x = .data[[y_var]],
               y = n,
               fill = y)) +
    geom_col(color = "black") +
    facet_grid(rows = vars(y), 
               scales = "free") +
    scale_fill_manual(values = c("brown2", "gray70")) +
    xlab(paste0("Feature - ", y_var)) +
    ylab("Counts") +
    ggtitle(paste0("Target varible (bank term deposit) VS ", y_var, "(counts) - bar plot")) +
    labs(fill = "Bank term deposit:") +
    theme_minimal(base_size = 16)
}  
