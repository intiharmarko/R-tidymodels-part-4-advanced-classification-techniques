# Assignment - Multiclass Classification with Dry Bean Dataset - EDA

rm(list = ls())
graphics.off()

# Load packages
library(tidyverse)

# Load functions
source("./03_assignment_EDA_functions.R")
source("./03_assignment_functions.R")



# Data (Dry Beans Data)

## file path
f_path <- "./data/dry_beans.csv"

## load data & apply initial cleaning
df <- import_data(f_path)



# EDA

## Initial EDA 

### df size
nrow(df)
ncol(df)

### column types
df %>% glimpse()

### check missing values
rows_miss_count(df)


## Single variable distribution

### target column ("y" - class) distribution
target_var_distr(df)

### feature column ("area") distribution
plot_histogram(df, "area")
plot_density(df, "area")

### feature column ("perimeter") distribution
plot_histogram(df, "perimeter")
plot_density(df, "perimeter")

### feature column ("major_axis_length") distribution
plot_histogram(df, "major_axis_length")
plot_density(df, "major_axis_length")

### feature column ("minor_axis_length") distribution
plot_histogram(df, "minor_axis_length")
plot_density(df, "minor_axis_length")

### feature column ("aspect_ration") distribution
plot_histogram(df, "aspect_ration")
plot_density(df, "aspect_ration")

### feature column ("eccentricity") distribution
plot_histogram(df, "eccentricity")
plot_density(df, "eccentricity")

### feature column ("convex_area") distribution
plot_histogram(df, "convex_area")
plot_density(df, "convex_area")

### feature column ("equiv_diameter") distribution
plot_histogram(df, "equiv_diameter")
plot_density(df, "equiv_diameter")

### feature column ("extent") distribution
plot_histogram(df, "extent")
plot_density(df, "extent")

### feature column ("solidity") distribution
plot_histogram(df, "solidity")
plot_density(df, "solidity")

### feature column ("roundness") distribution
plot_histogram(df, "roundness")
plot_density(df, "roundness")

### feature column ("compactness") distribution
plot_histogram(df, "compactness")
plot_density(df, "compactness")

### feature column ("shape_factor1") distribution
plot_histogram(df, "shape_factor1")
plot_density(df, "shape_factor1")

### feature column ("shape_factor2") distribution
plot_histogram(df, "shape_factor2")
plot_density(df, "shape_factor2")

### feature column ("shape_factor3") distribution
plot_histogram(df, "shape_factor3")
plot_density(df, "shape_factor3")

### feature column ("shape_factor4") distribution
plot_histogram(df, "shape_factor4")
plot_density(df, "shape_factor4")



## Features VS target variable

### "class" ~ "area" distribution
plot_tar_num_feat_dist(df, "area")

### "class" ~ "perimeter" distribution
plot_tar_num_feat_dist(df, "perimeter")

### "class" ~ "major_axis_length" distribution
plot_tar_num_feat_dist(df, "major_axis_length")

### "class" ~ "minor_axis_length" distribution
plot_tar_num_feat_dist(df, "minor_axis_length")

### "class" ~ "aspect_ration" distribution
plot_tar_num_feat_dist(df, "aspect_ration")

### "class" ~ "eccentricity" distribution
plot_tar_num_feat_dist(df, "eccentricity")

### "class" ~ "convex_area" distribution
plot_tar_num_feat_dist(df, "convex_area")

### "class" ~ "equiv_diameter" distribution
plot_tar_num_feat_dist(df, "equiv_diameter")

### "class" ~ "extent" distribution
plot_tar_num_feat_dist(df, "extent")

### "class" ~ "solidity" distribution
plot_tar_num_feat_dist(df, "solidity")

### "class" ~ "roundness" distribution
plot_tar_num_feat_dist(df, "roundness")

### "class" ~ "compactness" distribution
plot_tar_num_feat_dist(df, "compactness")

### "class" ~ "shape_factor1" distribution
plot_tar_num_feat_dist(df, "shape_factor1")

### "class" ~ "shape_factor2" distribution
plot_tar_num_feat_dist(df, "shape_factor2")

### "class" ~ "shape_factor3" distribution
plot_tar_num_feat_dist(df, "shape_factor3")

### "class" ~ "shape_factor4" distribution
plot_tar_num_feat_dist(df, "shape_factor4")



# Correlation between numeric variables
# - calculate correlation between each pair of numeric features 
# - and plot correlation heatmap
plot_corr_heatmap()

