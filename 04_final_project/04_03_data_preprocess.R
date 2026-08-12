# 4 Tidymodels Capstone Project: Forest Cover Classification - Data pre-process

rm(list = ls())
graphics.off()


# Load packages
library(tidyverse)
library(janitor)


# Load functions
source("./04_03_data_preprocess_functions.R")



# Data

## Forest cover - train data
df <- read_csv(file = "./data/train.csv", col_names = T)

## pre-process df
df.train <- preproc_data(df)

## split train data
## - split into two datasets
##   - training (EDA, model training phase 1,2)       - 80% of train data
##   - validate (model training phase 2-validation) - 20% of train data
set.seed(1123)

df.split.list <- split_train_data(df = df.train, p_t = 8/10)
df.training   <- df.split.list$training
df.validate   <- df.split.list$validate


## sample train data 
## - faster EDA 
## - faster model training (phase 1)
## - (optional usage!)
## -  we create two samples
##   - small ~ 10 % rows (used for EDA)
##   - tiny  ~ 1 % rows  (used for training phase 1)
set.seed(1123)

df.s.training <- sample_data(df = df.training, p = 0.1)
df.t.training <- sample_data(df = df.training, p = 0.01)


## check data

### view df
View(df.train)

### dimensions
df.train %>% ncol()
df.train %>% nrow()

### show column names
df.train %>% colnames()


## create column list
## - column names
## - column types
df.cols <- create_cols_list() 
