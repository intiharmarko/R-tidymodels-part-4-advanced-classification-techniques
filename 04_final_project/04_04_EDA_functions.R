#' 4 Tidymodels Capstone Project: Forest Cover Classification - EDA - Functions


#' Load packages
library(tidyverse)



#' Column types count
#'
#' This function draws a bar plot counting columns by column type, 
#' also target column is highlighted.
#' 
#' @param df.cols_ A data frame of columns with column types added.
#'
#' @return NULL
#'
col_type_count <- function(df.cols_ = df.cols){
  
  df.cols %>% 
    group_by(role, type) %>% 
    count() %>% 
    ungroup() %>% 
    ggplot(aes(x = type,
               y = n,
               fill = role)) +
    geom_col(color = "black") +
    labs(title = "Varible types count",
         fill = "Variable role:") +
    xlab("Column type") +
    ylab("Count") +
    scale_fill_manual(values = c("grey75", "brown1")) +
    theme_minimal(base_size = 16)
}



#' Missing rows % count
#'
#' This function draws a plot showing percentage of missing rows per each variable (column).
#' 
#' @param df A data frame for given data.
#'
#' @return NULL
#'
rows_miss_count <- function(df = df.train){
  
  map(.x = df,
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



#' Target variable distribution
#'
#' This function draws distribution of target variable 
#' "cover type" (class) - bar plot.
#' 
#' @param df A data frame for given data.
#'
#' @return NULL
#'
target_var_distr <- function(df){
  
  df %>% 
    count(cover_type) %>% 
    mutate(per = round(n / sum(n) * 100, 2),
           label = paste0(n, " | ", per, "%")) %>% 
    ggplot(aes(x = cover_type,
               y = n,
               label = label,
               fill = cover_type)) +
    geom_col(color = "black", 
             show.legend = F) +
    geom_text(size = 6) +
    scale_fill_brewer(palette = "Greens") +
    ggtitle("Target varible (forest cover type) barplot") +
    xlab("Forest cover type (class)") +
    ylab("Count") +
    labs(fill = "Cover type:") +
    theme_minimal(base_size = 16)
}



#' Plot histogram - numeric variable distribution
#'
#' This function draws distribution of selected numeric variable using histogram.
#' 
#' @param data A data frame for given data.
#' @param x_var A string - name of variable we are plotting.
#' @param bins_ An integer - number of bins in histogram.
#'
#' @return NULL
#'
plot_histogram <- function(data = df.train, 
                           x_var, 
                           bins_ = 30) {
  
  ggplot(data, 
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
#' @param data A data frame for given data.
#' @param x_var A string - name of variable we are plotting.
#'
#' @return NULL
#'
plot_density <- function(data = df.train, 
                         x_var) {
  
  ggplot(data, 
         aes(x = .data[[x_var]])) + 
    geom_density(fill = "gray90",
                 color = "black") +
    ggtitle(paste0("Numeric feature (", x_var, ") distribution -  density plot")) +
    xlab("") +
    ylab("Density") +
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
  cor_matrix <- data %>% select(where(is.numeric)) %>%
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



#' Plot bar plot - factor variable distribution
#'
#' This function draws distribution of selected factor variable using bar plot.
#' 
#' @param df A data frame for given data.
#' @param x_var A string - name of variable we are plotting.
#' @param x_font_size An integer indicating the size of fonts on x axis (ticks).
#'
#' @return NULL
#'
plot_fct_dist <- function(data = df.train, 
                          x_var,
                          x_font_size = 14) {
  
  # calculate counts per each level
  plot_data <- data %>% 
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
    theme_minimal(base_size = 16) +
    theme(axis.text.x = element_text(angle = 90, 
                                     size = x_font_size))
}



#' Plot target variable VS numeric feature distribution - box plot
#'
#' This function draws distribution of selected target factor column 
#' ("cover type" - forest  cover type) and numeric 
#' feature variable using box plot. 
#' 
#' @param df A data frame for given data.
#' @param y_var A string - name of numeric feature variable we are plotting.
#' @param x_font_size An integer indicating the size of fonts on x axis (ticks).
#'
#' @return NULL
#'
plot_tar_num_feat_dist <- function(df, 
                                   y_var,
                                   x_font_size = 8) {
  
  df %>% 
    ggplot(aes(x = cover_type,
               y = .data[[y_var]],
               fill = cover_type)) +
    geom_boxplot() +
    scale_fill_brewer(palette = "Greens") +
    xlab("Forest cover type") +
    ylab(paste0("Numeric feature (", y_var, ") value")) +
    ggtitle(paste0("Target varible (forest cover type) VS ", y_var, " - box plot")) +
    labs(fill = "Cover type:") +
    theme_minimal(base_size = 16) 
}  



#' Plot target variable VS factor feature distribution - heat map
#'
#' This function draws distribution of selected factor feature and numeric 
#' target variable variable using heat map. 
#' 
#' @param data A data frame for given data.
#' @param y_var A string - name of factor feature variable we are plotting.
#'
#' @return NULL
#'
plot_tar_fct_feat_heatmap <- function(data,
                                      y_var) {
  
  
  data %>%
    count(cover_type, .data[[y_var]]) %>%
    ggplot(aes(x = cover_type, 
               y = .data[[y_var]], 
               fill = n)) +
    geom_tile(color = "black") +
    scale_fill_viridis_c(option = "viridis") +
    xlab("Forest cover type (class)") +
    ylab(paste0("Factor feature (", y_var, ")")) +
    ggtitle(paste0("Cover type VS ", y_var, " - heat map")) +
    labs(fill = "Count:") +
    theme_minimal(base_size = 14)
}  



#' Plot target variable VS factor feature distribution - bar plot
#'
#' This function draws distribution of selected factor feature and numeric 
#' target variable variable using box plot. 
#' 
#' @param data A data frame for given data.
#' @param x_var A string - name of factor feature variable we are plotting.
#'
#' @return NULL
#'
plot_tar_fct_feat_bar <- function(data, 
                                  x_var) {
  
  data %>% 
    ggplot(aes(x = .data[[x_var]], 
               fill = cover_type)) +
    geom_bar(position = "fill", 
             color = "black") + 
    scale_fill_brewer(palette = "Greens") +
    xlab(paste0("Factor feature (", x_var, ")")) +
    ylab("Percentage of cases (%)") +
    ggtitle(paste0("Cover type VS ", x_var, " - bar plot")) +
    labs(fill = "Cover type:") +
    theme_minimal(base_size = 14)
}  

