# 4 Tidymodels Capstone Project: Forest Cover Classification - EDA


# Load packages
library(tidyverse)
library(GGally)

# Load functions
source("./04_04_EDA_functions.R")
source("./04_03_data_preprocess_functions.R")



# Initial EDA

## figure: column types count
col_type_count()

## figure: check missing values
rows_miss_count(df = df.training)



# Target variable ("cover_type") distribution

## distribution target variable
target_var_distr(df = df.s.training)



# Numeric features distribution (not binary flag columns!)

## "elevation"  distribution
plot_histogram(data = df.s.training, x_var = "elevation")
plot_density(data = df.s.training,   x_var = "elevation")

## "aspect"  distribution
plot_histogram(data = df.s.training, x_var = "aspect")
plot_density(data = df.s.training,   x_var = "aspect")

## "slope"  distribution
plot_histogram(data = df.s.training, x_var = "slope")
plot_density(data = df.s.training,   x_var = "slope")

## "horizontal_distance_to_roadways"  distribution
plot_histogram(data = df.s.training, x_var = "horizontal_distance_to_roadways")
plot_density(data = df.s.training,   x_var = "horizontal_distance_to_roadways")

## "vertical_distance_to_hydrology"  distribution
plot_histogram(data = df.s.training, x_var = "vertical_distance_to_hydrology")
plot_density(data = df.s.training,   x_var = "vertical_distance_to_hydrology")

## "hillshade_9am"  distribution
plot_histogram(data = df.s.training, x_var = "hillshade_9am")
plot_density(data = df.s.training,   x_var = "hillshade_9am")

## "hillshade_noon"  distribution
plot_histogram(data = df.s.training, x_var = "hillshade_noon")
plot_density(data = df.s.training,   x_var = "hillshade_noon")

## "hillshade_3pm"  distribution
plot_histogram(data = df.s.training, x_var = "hillshade_3pm")
plot_density(data = df.s.training,   x_var = "hillshade_3pm")

## "horizontal_distance_to_fire_points"  distribution
plot_histogram(data = df.s.training, x_var = "horizontal_distance_to_fire_points")
plot_density(data = df.s.training,   x_var = "horizontal_distance_to_fire_points")


## Correlation heat-map (only selected numeric features)
plot_corr_heatmap(data = df.s.training %>% select(2:11))



# Binary integer features distribution
# - we will merge flags into factor feature
# - we have two different groups of flag features (soil type & wilderness_area)
# - based on the description: only single flag can have value 1 per given row

## check that only one flag is active
df.s.training %>%
  select(starts_with("soil_type")) %>%
  mutate(row_sum = rowSums(.)) %>%
  count(row_sum)

df.s.training %>%
  select(starts_with("wilderness")) %>%
  mutate(row_sum = rowSums(.)) %>%
  count(row_sum)


## "soil_type_*" flags distribution (converted to factor)

### first create factor feature "soil_type"
df.s.training <- merge_flags(data = df.s.training, cols_name = "soil_type")

### draw plot
plot_fct_dist(data = df.s.training, x_var = "soil_type")


## "wilderness_area*" flags distribution (converted to factor)

### first create factor feature "wilderness_area"
df.s.training <- merge_flags(data = df.s.training, cols_name = "wilderness_area")

### draw plot
plot_fct_dist(data = df.s.training, x_var = "wilderness_area")



# Numeric features VS target variable - box plot

## "elevation" ~ "cover type" distribution
plot_tar_num_feat_dist(df = df.s.training, y_var = "elevation")

## "aspect" ~ "cover type" distribution
plot_tar_num_feat_dist(df = df.s.training, y_var = "aspect")

## "slope" ~ "cover type" distribution
plot_tar_num_feat_dist(df = df.s.training, y_var = "slope")

## "horizontal_distance_to_roadways" ~ "cover type" distribution
plot_tar_num_feat_dist(df = df.s.training, y_var = "horizontal_distance_to_roadways")

## "vertical_distance_to_hydrology" ~ "cover type" distribution
plot_tar_num_feat_dist(df = df.s.training, y_var = "vertical_distance_to_hydrology")

## "hillshade_9am" ~ "cover type" distribution
plot_tar_num_feat_dist(df = df.s.training, y_var = "hillshade_9am")

## "hillshade_noon" ~ "cover type" distribution
plot_tar_num_feat_dist(df = df.s.training, y_var = "hillshade_noon")

## "hillshade_3pm" ~ "cover type" distribution
plot_tar_num_feat_dist(df = df.s.training, y_var = "hillshade_3pm")

## "horizontal_distance_to_fire_points" ~ "cover type" distribution
plot_tar_num_feat_dist(df = df.s.training, y_var = "horizontal_distance_to_fire_points")

## pair plot
ggpairs(data = df.t.training, columns = 1:11)



# Binary integer features VS target variable

## "soil_type_*" flags distribution (converted to factor)
plot_tar_fct_feat_heatmap(data = df.s.training, y_var = "soil_type")
plot_tar_fct_feat_bar(data = df.s.training,     x_var = "soil_type")

## "wilderness_area*" flags distribution (converted to factor)
plot_tar_fct_feat_heatmap(data = df.s.training, y_var = "wilderness_area")
plot_tar_fct_feat_bar(data = df.s.training, x_var = "wilderness_area")



# Drop merged flags column at the end (two columns)
df.s.training <- df.s.training %>% select(-c("soil_type", "wilderness_area"))

