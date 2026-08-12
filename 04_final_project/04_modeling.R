# 4 Tidymodels Capstone Project: Forest Cover Classification - Modeling

graphics.off()

# Load packages
library(tidyverse)
library(tidymodels)
library(workflowsets)
library(future)
library(doFuture)
library(themis)

library(ranger)
library(xgboost)
library(bonsai)
library(lightgbm)
library(bonsai)

# Load functions
source("./04_05_modeling_functions.R")


# Data

## Set up CV
## - we will use k=5 fold CV
## - stratified sampling applied
set.seed(1123)

cv_folds <- vfold_cv(df.t.training, 
                     v = 5, 
                     strata = cover_type)



# Model training - Phase I (Model candidate discovery)
# 
# - Goal: find the best hyperparameters per each workflow
#
# - Selected models:
#   - penalized logistic regression (engine: glmnet)
#   - random forest                 (engine: ranger)
#   - XGBoost                       (engine: xgboost)
#   - LightGBM                      (engine: lightgbm)
#
# - Selected resampling strategies:
#   - no resampling
#   - downsample    (under_ratio = 5)
#
# - Approach:
#  - optimize using macro metrics:
#  - macro F1 (primary metric)
#  - balanced accuracy (secondary metric)
#  - Hand–Till ROC AUC (secondary metric)
#
# - Outcome:
#  - for each workflow:
#    - best hyperparameters found cheaply
#    - best resampling strategy
#    - preliminary ranking
#    - but not final model training!


## model specifications
##
## - algorithms / models used and list of hyper-parameters we tune per model:
##   - logistic regression (penalized) - (engine - "glmnet")
##     - penalty
##     - mixture
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

### logistic regression - penalized
log_mod <- multinom_reg(mode = "classification", 
                        penalty = tune(),
                        mixture = tune()) %>%
  set_engine("glmnet") 

### random Forest model
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

### list of models specs.
mod_list <- list(log  = log_mod,
                 rf   = rf_mod,
                 xgb  = xgb_mod,
                 lgbm = lgbm_mod)


## recipes
## - base recipe (formula + some pre-processing steps):
##   - impute missing numeric features with median values & removing zero variance features
##   - normalizing numeric features
## - re-sampling technique (class imbalance) based recipes
##   - no-sampling 
##   - down-sampling  (under_ratio = 5)
##     - fixed minority proportion (no tuning)
## - we store recipes into a list

### basic recipe - formula & impute missing
rec_basic <- recipe(formula = cover_type ~ ., data = df.t.training) %>% 
  step_impute_median(all_numeric_predictors()) %>% 
  step_zv(all_predictors())

### recipe with normalization applied
rec_norm <- rec_basic %>% 
  step_normalize(all_numeric_predictors()) 

### recipe with down-sampling applied & variation with normalization
rec_down <- rec_basic %>% 
  step_downsample(cover_type, under_ratio = 5, skip = TRUE)

rec_norm_down <- rec_norm %>% 
  step_downsample(cover_type, under_ratio = 5, skip = TRUE)

### list of recipes
rec_list <- list(base       = rec_basic,
                 norm       = rec_norm,
                 down       = rec_down,
                 down_norm  = rec_norm_down)


## workflows
## - match model specifications with recipes
## - not all possible matches (only selected ones)!
##   - glmnet: only normalized variants
##   - tree-based: only non-normalized variants
## - create workflow set
wflow_set <- workflow_set(
  preproc = rec_list,
  models = mod_list,
  cross = T) %>% 
  filter(str_detect(wflow_id, 
                    pattern = "^(norm|down_norm)_log$|^(base|down)_(rf|xgb|lgbm)$"))


## finalize model's parameters (features) list
## - we need to provide features list
## - for each combination of model and used recipe
## - for tuning step
par_fin_list <- finalize_wflow_params(df.train_ = df.t.training)

## selected metrics (we collect during tuning)
multi_class_metrics <- metric_set(
  f_meas,        # F1 (macro by default for multiclass)
  bal_accuracy,  # balanced accuracy
  roc_auc,       # Hand–Till by default for multiclass
  precision,     # macro by default
  recall,        # macro by default
  accuracy       # overall accuracy
)


## hyperpar. tuning
## - execute tuning using date based cross-validation data
## - tuning is executed using Bayesian optimization 
## - Bayes optimization tuning specs:
##   - 8 initial points 
##   - max 25 iterations
##   - stop if no improvements after 10 iterations
## - we will tune using parallel computing on multiple CPU-s
## - we must register backend for foreach  
## - we must specify cores for parallel computing

registerDoFuture()                                        # parallel backend for foreach
plan(multisession, workers = parallel::detectCores() - 1) # use multi-cores
future::nbrOfWorkers()                                    # check how many workers are active

set.seed(235)

tune_rez <- tune_bayes_custom(iter = 25,      
                              initial = 8,    
                              no_improve = 10) 


save_WS_snapshot("./data/tuning_rez.RData")
load_WS_snapshot("./data/tuning_rez.RData")


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


## visualize tuning / training results
## - figure 1: ROC AUC per each top candidate 
##             over all assessment folds
## - figure 2: multiple classification metrics 
##             aggregated over all assessment folds
##             per each top candidate
visualize_tune_rez()



# Model training - Phase IIA (Validation: training → validate)
# 
# - Goal: find the single best model workflow.
#
# - Seps:
#   - 1) For each workflow candidate from Phase 1:
#       - finalize hyperparameters
#       - fit on full train (not tiny!)
#       - predict on validate
#   - 2) Compute the same macro metrics (F1 score).
#   - 3) Rank all workflows on the unseen validation data.
#
# - Outcome:
#  - identify the global winner (final selected workflow):
#    - best algorithm
#    - best hyperparameters
#    - best resampling strategy

## finalize each workflow and fit best model
## - finalize workflow with best model
## - train model (model fit) on training data (not full train data)
## - for all selected algorithms
mod_fit_list <- fit_best_models(df.train_ = df.training)

save_WS_snapshot("./data/final_models_retrained.RData")
load_WS_snapshot("./data/final_models_retrained.RData")


## predict and prepare the prediction data frames
df.validate_pred <- predict_multi_mod(mod_fit_list, df.validate)


## compare results on validate data set
## - figure: F1 score per each final model (validate data)
visualize_validate_rez()


## final model selection
## - model with highest ROC AUC on validation set

### get final model workflow id
fin_model_wflow_id <- df.validate_pred %>% 
  group_by(wflow_id) %>% 
  f_meas(truth = cover_type, 
         .pred_class) %>% 
  ungroup() %>%  
  arrange(desc(.estimate)) %>% 
  slice_head(n = 1) %>% 
  pull(wflow_id)

### final model and its hyperparametrs
fin_model_wflow_id
top_mod_candidates[[fin_model_wflow_id]]

### store final model fit
final_mod_fit <- mod_fit_list[[fin_model_wflow_id]]



# Model training - Phase IIB (Final training: train data)
# 
# - Goal: train final model using all available data before the test set.
#
# - Seps:
#   - 1) Combine data:
#       - train = training + validate
#   - 2) Finalize workflow.
#   - 3) Fit on train data (full train data).
#
# - Outcome:
#  - Fully trained final model - before testing.

## finalize final selected model's workflow and fit model on full train data
## - finalize workflow with best model
## - train model (model fit) on whole train data
## - for all selected algorithms
final_mod_fit <- final_mod_fit %>% 
finalize_workflow(top_mod_candidates[[fin_model_wflow_id]]) %>%
  fit(data = df.train)

save_WS_snapshot("./data/final_training_rez.RData")
load_WS_snapshot("./data/final_training_rez.RData")



# Model Testing 
# - best performing algorithm is tested on unseen test data
# - import test data 
# - then apply data pre-processing step
# - use model to predict cover_type on test data
# - then extract selected classification metrics
# - visualize results

## load data pre-processing functions
source("./04_03_data_preprocess_functions.R")


## load data - test data
df.test <- read_csv(file = "./data/test.csv", col_names = T)

### pre-process df
df.test <- preproc_data(df = df.test)


## predict and prepare the prediction data frames
df.test_pred <- predict_single_mod(final_mod_fit, df.test)

## visualize model testing results
## - figure 1: selected metrics for final model on test data
## - figure 2/3: confusion matrix (absolute / relative values) for final model on test data
visualize_test_rez()

