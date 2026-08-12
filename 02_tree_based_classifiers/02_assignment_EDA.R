# Assignment - Bank Marketing - EDA

rm(list = ls())
graphics.off()

# Load packages
library(tidyverse)

# Load functions
source("./02_assignment_EDA_functions.R")
source("./02_assignment_functions.R")



# Data

## Bank Marketing data
df <- read_csv("./data/bank.csv", col_names = T)

## data cleaning
df <- clean_df()



# EDA

## Initial EDA 

### df size
nrow(df)
ncol(df)

### column types
df %>% glimpse()

### check missing values
rows_miss_count(df)

### count number of categories / levels - categorical (factors) features 
nr_categories_count(df)


## Single variable distribution

### target column ("y" - bank term deposit) distribution
target_var_distr(df)

### feature column ("age") distribution
plot_histogram(df, "age")
plot_density(df, "age")

### feature column ("job") distribution
plot_fct_dist(df, "job")

### feature column ("marital") distribution
plot_fct_dist(df, "marital")

### feature column ("education") distribution
plot_fct_dist(df, "education")

### feature column ("default") distribution
plot_fct_dist(df, "default")

### feature column ("balance") distribution
plot_histogram(df, "balance")
plot_density(df, "balance")

### feature column ("housing") distribution
plot_fct_dist(df, "housing")

### feature column ("contact") distribution
plot_fct_dist(df, "contact")

### feature column ("day") distribution
plot_histogram(df, "day")
plot_density(df, "day")

### feature column ("month") distribution
plot_histogram(df, "month")
plot_density(df, "month")

### feature column ("duration") distribution
plot_histogram(df, "duration")
plot_density(df, "duration")

### feature column ("campaign") distribution
plot_histogram(df, "campaign")
plot_density(df, "campaign")

### feature column ("pdays") distribution
plot_histogram(df, "pdays")
plot_density(df, "pdays")

### feature column ("previous") distribution
plot_histogram(df, "previous")
plot_density(df, "previous")

### feature column ("poutcome") distribution
plot_fct_dist(df, "poutcome")

### feature column ("contacted_prev_camp") distribution
plot_fct_dist(df, "contacted_prev_camp")



## Features VS target variable

### "y" ~ "age" distribution
plot_tar_num_feat_dist(df, "age")

### "y" ~ "job" distribution
plot_tar_cat_feat_dist(df, "job")

### "y" ~ "marital" distribution
plot_tar_cat_feat_dist(df, "marital")

### "y" ~ "education" distribution
plot_tar_cat_feat_dist(df, "education")

### "y" ~ "default" distribution
plot_tar_cat_feat_dist(df, "default")

### "y" ~ "balance" distribution
plot_tar_num_feat_dist(df, "balance")

### "y" ~ "housing" distribution
plot_tar_cat_feat_dist(df, "housing")

### "y" ~ "loan" distribution
plot_tar_cat_feat_dist(df, "loan")

### "y" ~ "contact" distribution
plot_tar_cat_feat_dist(df, "contact")

### "y" ~ "duration" distribution
plot_tar_num_feat_dist(df, "duration")

### "y" ~ "poutcome" distribution
plot_tar_cat_feat_dist(df, "poutcome")

### y" ~ "contacted_prev_camp"  distribution
plot_tar_cat_feat_dist(df, "contacted_prev_camp")

### y" ~ "day"  distribution
plot_tar_num_feat_bar_dist(df, "day")

### y" ~ "month"  distribution
plot_tar_num_feat_bar_dist(df, "month")

### y" ~ "campaign"  distribution
plot_tar_num_feat_bar_dist(df, "campaign")

### y" ~ "previous"  distribution
plot_tar_num_feat_bar_dist(df, "previous")
