#' 4 Tidymodels Capstone Project: Forest Cover Classification - Data pre-process - Functions


#' Load packages
library(tidyverse)
library(tidymodels)



#' Pre-process data
#'
#' This function converts target column into factor type.
#' 
#' @param df A data frame holding the data.
#'
#' @return df A data frame with parsed target column.
#'
preproc_data <- function(df = df.train){
  
  df <- df %>% 
    mutate(cover_type = as.factor(cover_type))
  
  return(df)
}



#' Split train data
#'
#' This function randomly splits (stratified sampling applied) the train data into:
#' - training set
#' - validate set
#' You need to provide a data frame and set's proportions.
#' 
#' @param df A data frame for selected data.
#' @param p_t A numerical value - proportion of data assigned to training set (between 0 and 1 - must be less than 1!) - (default value 8/10).
#'
#' @return data_split.list A list of data frame split into sets.
#'
split_train_data <- function(df,
                             p_t = 8/10){
  
  # training VS validate
  split_init  <- initial_split(df,             
                               prop = p_t, 
                               strata = cover_type) 
  
  df.training <- training(split_init)
  df.validate <- testing(split_init) 
  
  # print data set dimension sizes
  print("Data split into:")
  print(paste0("training set: ", nrow(df.training), " rows | ", round(nrow(df.training) / nrow(df) * 100, 2), " % of all rows"))
  print(paste0("validate set: ", nrow(df.validate), " rows | ", round(nrow(df.validate) / nrow(df) * 100, 2), " % of all rows"))

  # create list
  data_split.list <- list(training = df.training,
                          validate =  df.validate)
  
  return(data_split.list)
}



#' Create smaller training data sample
#'
#' This function generates smaller training data sample.
#' Also it considers target column for stratification sampling
#' 
#' @param df A data frame holding the data.
#' @param p A float - proportion of rows sampled.
#'
#' @return df.sample Sampled data frame
#'
sample_data <- function(df = df.train,
                        p){
  
  # sample data - stratified sampling
  data_split <- initial_split(df,
                              prop   = p,        
                              strata = cover_type)  
  
  return(training(data_split))
}



#' Create columns list
#'
#' This function stores column names with column types into table.
#' Also target column is labelled. 
#' 
#' @param df  A data frame for given data.
#' @param tar A string holding the name of target column in df.
#'
#' @return df.cols A data frame of columns with column types added.
#'
create_cols_list <- function(df = df.train,
                             tar = "cover_type"){
  
  df.cols <- map(.x = df, 
                 .f = class) %>% 
    unlist() %>% 
    tibble(col = names(.),
           type = .) %>% 
    mutate(position = row_number(),
           role = if_else(col == tar, 
                          "target", 
                          "feature")) %>% 
    arrange(desc(role), position)
  
  return(df.cols)
}



#' Merge flag columns into single column
#'
#' This function merges selected binary flag columns into single 
#' factor column 
#' 
#' @param data  A data frame for given data.
#' @param cols_name A string holding the name of binary flag column (name prefix!).
#'
#' @return data A data frame with additional merged column.
#'
merge_flags <- function(data, cols_name){
  
  data <- data %>%
    mutate(!!cols_name := select(., starts_with(cols_name)) %>%
             as.matrix() %>%
             max.col(ties.method = "first") %>%
             factor())
  
  return(data)
}

