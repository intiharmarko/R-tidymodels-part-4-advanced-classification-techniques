# Assignment - Bank Marketing

rm(list = ls())
graphics.off()

# Load packages
library(tidyverse)
library(tidymodels)
library(workflowsets)
library(future)
library(doFuture)
library(janitor)

library(kknn)
library(rpart)
library(ranger)
library(xgboost)
library(bonsai)
library(lightgbm)

# Load functions
source("./02_assignment_functions.R")



# Data (Bank Marketing data)

##data import
df <- read_csv("./data/bank.csv", col_names = T)

## data cleaning
df <- clean_df()

## split data
##
## - train    70%
##   - CV (k = 10)
## - validate 20%
## - test     10%
set.seed(1123)

df.split.list <- split_data(df, p_t = 7/10, p_v = 2/3)
df.train      <- df.split.list$train
df.validate   <- df.split.list$validate
df.test       <- df.split.list$test

## set up CV for hyperpar. tuning
## - we will use k=10 fold CV
cv_folds <- vfold_cv(df.train, v = 10)


# Model training

## Model specifications
## - specify model specification
## - create recipe
## - create workflow
##
## - algorithms / models used and list of hyper-parameters we tune per model:
##   - logistic regression (penalized) - (engine - "glmnet")
##     - penalty
##     - mixture
##   - KNN (engine - "kknn")
##     - neighbors
##   - Decision tree (engine - "rpart")
##     - min_n
##     - tree_depth
##     - cost_complexity
##   - Random forest (engine - "ranger")
##     - mtry
##     - min_n
##     - trees
##   - XGBoost (engine - "xgboost")
##     - trees
##     - tree_depth
##     - learn_rate
##   - LigthGBM (engine - "ligthgbm")
##     - trees
##     - tree_depth
##     - learn_rate

### logistic regression
log_mod <- logistic_reg(mode = "classification", 
                        penalty = tune(),
                        mixture = tune()) %>%
  set_engine("glmnet") 

### KNN model
knn_mod <- nearest_neighbor(mode = "classification",
                            neighbors = tune()) %>%
  set_engine("kknn")

### Decision tree model
tree_mod <- decision_tree(mode = "classification",
                          min_n = tune(),
                          tree_depth = tune(),
                          cost_complexity = tune()) %>%
  set_engine("rpart")

### Random Forest model
rf_mod <- rand_forest(mode = "classification",
                      mtry = tune(),
                      min_n = tune(),
                      trees = tune()) %>%
  set_engine("ranger")

### XGBoost model
xgb_mod <- boost_tree(mode = "classification",
                      trees = tune(),
                      tree_depth = tune(),
                      learn_rate = tune()) %>%
  set_engine("xgboost")

### LigthGBM model
lgbm_mod <- boost_tree(mode = "classification",
                       trees = tune(),
                       tree_depth = tune(),
                       learn_rate = tune()) %>%
  set_engine("lightgbm")

### List of models specs.
mod_list <- list(log  = log_mod,
                 knn  = knn_mod,
                 tree = tree_mod,
                 rf   = rf_mod,
                 xgb  = xgb_mod,
                 lgbm = lgbm_mod)


## Recipes
## - some recipes specifics are per algorithm
##   - no pre-processing needed: decision trees & random forests
##   - encoding categorical features needed: logistic regression, KNN, XGBoost & LightGBM
##   - normalizing numeric features needed: KNN
##   - removing features with zero variance (problem will occur later near ZV numeric features): KNN

### basic recipe - no pre-processing
rec_basic <- recipe(formula = y ~ .,
                    data = df.train)

### recipe with dummy features 
rec_dummy <- rec_basic %>% 
  step_dummy(all_nominal_predictors()) 

### recipe with dummy features + removed zero variance features + normalized numeric features 
rec_norm_nzv_dummy <- rec_dummy %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())

### list of recipes
rec_list <- list(log  = rec_dummy,
                 knn  = rec_norm_nzv_dummy,
                 tree = rec_basic,
                 rf   = rec_basic,
                 xgb  = rec_dummy,
                 lgbm = rec_dummy)


## Workflows
## - match model specifications with recipes
## - create workflow set
wflow_set <- workflow_set(
  preproc = rec_list,
  models = mod_list,
  cross = F) %>% 
  mutate(wflow_id = names(mod_list))


## Finalize model's parameters (features) list
## - we need to provide features list
## - for each combination of model and used recipe
## - for tuning step
par_fin_list <- finalize_model_params()


## hyperpar. tuning
## - execute tuning using CV data
## - tuning is executed using Bayesian optimization 
## - Bayes optimization tuning specs:
##   - 5 initial points 
##   - max 20 iterations
##   - stop if no improvements after 5 iterations
## - we will tune using parallel computing on multiple CPU-s
## - we must register backend for foreach  
## - we must specify cores for parallel computing
registerDoFuture()                                        # parallel backend for foreach
plan(multisession, workers = parallel::detectCores() - 1) # use multi-cores
future::nbrOfWorkers()                                    # check how many workers are active

set.seed(235)

tune_rez <- tune_bayes_custom(iter = 20, 
                              initial = 5, 
                              no_improve = 5)


## collect out-of-fold predictions metrics
## - extracted from assessment folds
## - each workflow / model included
## - we collect on two levels:
##   - per each assessment fold
##   - aggregate over all assessment folds
tune_rez_met_fold <- collect_metrics_l(sum_met = FALSE)
tune_rez_met_aggr <- collect_metrics_l(sum_met = TRUE)


## select top model candidate
## - for each workflow (model + recipe)
## - we choose best performing model candidate
## - based on highest ROC AUC metric
top_mod_candidates <- extract_best_models()


## select top model (per each algorithm!) & finalize it
## - model with highest AUC
## - and selected values for hyperpar.
## - finalize workflow with best model
## - train model (model fit) on whole train data
## - for all selected algorithms
mod_fit_list <- fit_best_models()


## visualize tuning results
## - figure 1: ROC AUC per each top candidate 
##             over all assessment folds
## - figure 2: multiple classification metrics 
##             aggregated over all assessment folds
##             per each top candidate
visualize_tune_rez()



# Model Validation - Select best performing algorithm 
# - evaluate each models performance
# - on validation data (AUC metrics)
# - first predict output on validation data
# - then extract AUC
# - visualize AUC & select best performing algorithm

## predict and prepare the prediction data frames
df.validate_pred <- map(mod_fit_list, ~ predict(.x, 
                                                df.validate, 
                                                type = "prob") %>% 
                          bind_cols(df.validate) %>% 
                          select(y, .pred_1)) %>% 
  bind_rows(., .id = "wflow_id")

## calculate AUC for each model
df.validate_AUC <- df.validate_pred %>% 
  group_by(wflow_id) %>% 
  roc_auc(truth = y, 
          .pred_1) %>% 
  ungroup() 

## compare results on validate data set
## - use ROC AUC
## - compare top candidates
levels.wflow   <- c("log", "knn", "tree", "rf", "xgb", "lgbm")

df.validate_AUC %>%  
  mutate(wflow_id = factor(wflow_id, levels = levels.wflow)) %>% 
  ggplot(aes(x = wflow_id,
             y = .estimate,
             label = round(.estimate, 2),
             fill = wflow_id)) +
  geom_col(color = "black", 
           show.legend = F) +
  geom_text(size = 7) +
  scale_y_continuous(breaks = seq(0,1,0.05)) +
  scale_fill_viridis_d() +
  xlab("Workflow (model)") +
  ylab("ROC AUC") +
  ggtitle("ROC AUC on validate data set for best model candidate per workflow") +
  labs(subtitle = "Showing only best performing model candidate per each workflow (classifier - recipe)\nValidation dataset was used.") +
  theme_minimal(base_size = 16)


## final model selection
## - model with highest ROC AUC on validation set

### get final model workflow id
fin_model_wflow_id <- df.validate_AUC %>%  
  arrange(desc(.estimate)) %>% 
  slice_head(n = 1) %>% 
  pull(wflow_id)

### final model and its hyperparametrs
fin_model_wflow_id
top_mod_candidates[[fin_model_wflow_id]]

### store final model fit
final_mod_fit <- mod_fit_list[[fin_model_wflow_id]]



# Model testing
# - we will use final selected model
# - then we will generate predictions on test set
# - and check ROC AUC

## generate predictions
df.pred.test <- predict(final_mod_fit, 
                        df.test, 
                        type = "prob") %>% 
  bind_cols(df.test) %>% 
  dplyr::select(y, .pred_1)

## calculate test ROC AUC
df.pred.test %>% 
  roc_auc(truth = y, .pred_1)
