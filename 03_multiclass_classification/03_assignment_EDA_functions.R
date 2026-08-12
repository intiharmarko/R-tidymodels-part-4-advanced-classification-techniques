#' Assignment - Multiclass Classification with Dry Bean Dataset - EDA - Functions



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
#' "class" (dry bean type) - bar plot.
#' 
#' @param df_ A data frame for given data.
#'
#' @return NULL
#'
target_var_distr <- function(df_){
  
  df_ %>% 
    count(class) %>% 
    mutate(per = round(n / sum(n) * 100, 2),
           label = paste0(n, " | ", per, "%")) %>% 
    ggplot(aes(x = class,
               y = n,
               label = label,
               fill = class)) +
    geom_col(color = "black", 
             show.legend = F) +
    geom_text(size = 6) +
    scale_color_viridis_d(option = "magma") +
    ggtitle("Target varible (dry bean class) barplot") +
    xlab("Dry bean class") +
    ylab("Bean count") +
    labs(fill = "Dry bean class:") +
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



#' Plot target variable VS numeric feature distribution - box plot
#'
#' This function draws distribution of selected target factor column 
#' ("class" - dry bean class) and numeric 
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
    ggplot(aes(x = class,
               y = .data[[y_var]],
               fill = class)) +
    geom_boxplot() +
    scale_fill_viridis_d(option = "magma") +
    xlab("Dry bean class") +
    ylab(paste0("Numeric feature (", y_var, ") value")) +
    ggtitle(paste0("Target varible (dry bean class) VS ", y_var, " - box plot")) +
    labs(fill = "Dry bean class:") +
    theme_minimal(base_size = 16) 
}  



#' Plot correlation heatmap - numeric variables
#'
#' This function firsts calculates Pearson correlation coefficient for all pairs
#' of numeric variables, and then draws heatmap.
#' 
#' @param data A data frame for given data.
#'
#' @return NULL
#'
plot_corr_heatmap <- function(data = df){
  
  # compute correlation matrix (Person coefficient)
  cor_matrix <- df %>% select(where(is.numeric)) %>%
    cor(method = "pearson", use = "complete.obs")
  
  # convert to long format for ggplot
  cor_long <- cor_matrix %>%
    as.data.frame() %>%
    rownames_to_column("var1") %>%
    pivot_longer(-var1, names_to = "var2", values_to = "correlation")
  
  # plot the heat map
  ggplot(cor_long, 
         aes(x = var1, 
             y = var2, 
             fill = correlation)) +
    geom_tile(color = "white") +
    geom_text(aes(label = round(correlation, 2)), 
              size = 3) +
    scale_fill_gradient2(low = "blue", 
                         mid = "white", 
                         high = "red",
                         midpoint = 0, 
                         limit = c(-1, 1), 
                         name = "Pearson corr.:") +
    xlab("") +
    ylab("") +
    ggtitle("Correlation Heatmap") +
    coord_fixed() +
    theme_minimal(base_size = 16) +
    theme(axis.text.x = element_text(angle = 90, 
                                     hjust = 1),
          legend.title = element_text(size = 12),
          legend.text = element_text(size = 10))
}

