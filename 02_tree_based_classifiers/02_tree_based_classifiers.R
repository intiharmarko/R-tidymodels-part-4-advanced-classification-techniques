# 2 Tree-Based Classifiers

rm(list = ls())
graphics.off()

# Install packages
#install.packages("rpart.plot")
#install.packages("ranger")

# Load packages
library(tidyverse)
library(tidymodels)
library(rpart)
library(rpart.plot)
library(ranger)
library(xgboost)
library(bonsai)
library(lightgbm)
library(future)
library(doFuture)



# 2.4 Decision Tree in tidymodels

## data - diabetes
df.train <- MASS::Pima.tr
df.test  <- MASS::Pima.te


## recode response variable
## - convert values to 1/0
## - put response column at the beginning of the df
## - we will use a function for the task
recode_response <- function(df){
  df <- df %>% 
    mutate(type = str_to_lower(type),
           type = if_else(type == "yes", 1, 0),
           type = factor(type, levels = c(1, 0))) %>% 
    as_tibble()
  
  return(df)
}

df.train <- recode_response(df.train)
df.test  <- recode_response(df.test)


## set up CV for hyperpar. tuning
## - we will use k=10 fold CV
## - we don't have severely imbalanced data (so stratification is not needed)
cv_folds <- vfold_cv(df.train, v = 10)



## Model Training

### model specification
### - "rpart" algorithm
### - we tune (hyperpar.):
###   - min_n ~ min number of samples per leaf 
###   - tree_depth ~ max number of tree's splitting levels
###   - cost_complexity ~ alpha parameter that controls tree complexity via pruning
tree_mod <- decision_tree(mode = "classification",
                          min_n = tune(),
                          tree_depth = tune(),
                          cost_complexity = tune()) %>%
  set_engine("rpart")


### recipe
### - we predict 'type' variable
### - we impute potential missing values
tree_rec <- recipe(type ~ ., data = df.train) %>%
  step_impute_median(all_numeric_predictors())

### workflow 
tree_wflow <- workflow() %>%
  add_recipe(tree_rec) %>%
  add_model(tree_mod)


### prepare grid for tuning
### - we will use regular grid
tree_grid_regular <- grid_regular(
  cost_complexity(),
  tree_depth(range = c(1L, 12L)),
  min_n(range = c(2L, 40L)),
  levels = 7
)


### classification metrics to be collected
class_metrics <- metric_set(
  accuracy, bal_accuracy,     
  precision, recall, f_meas,           
  roc_auc, pr_auc,           
  mcc, j_index
)


### hyperpar. tuning
### - execute tuning using CV data split & grid
### - we will tune using parallel computing on multiple CPU-s
### - we must register backend for foreach  
### - we must specify cores for parallel computing
registerDoFuture()                                        # parallel backend for foreach
plan(multisession, workers = parallel::detectCores() - 1) # use multi-cores
future::nbrOfWorkers()                                    # check how many workers are active

set.seed(235)

tree_tune_rez <- tune_grid(tree_wflow,
                           resamples = cv_folds,
                           grid = tree_grid_regular,
                           metrics = class_metrics,
                           control = control_grid(verbose = TRUE,
                                                  parallel_over = "everything",
                                                  save_pred = T)
)


### select best model 
### - model with highest AUC (assessment folds)
### - and selected value for neighbors hyperpar.
tree_mod_best <- select_best(tree_tune_rez, metric = "roc_auc")

### check training AUC 
### - filter best model predictions (assessment folds)
### - show AUC per each assessment fold 
### - and AUC calculated using all assessment folds
pred_tune_rez <- collect_predictions(tree_tune_rez)

best_AUC_folds <-inner_join(x = pred_tune_rez,
                            y = tree_mod_best %>% select(.config),
                            by = ".config") %>%
  group_by(id) %>% 
  roc_auc(type, .pred_1) %>% 
  ungroup()

best_AUC <-inner_join(x = pred_tune_rez,
                      y = tree_mod_best %>% select(.config),
                      by = ".config") %>% 
  roc_auc(type, .pred_1) 

best_AUC_folds
best_AUC


### finalize model
### - finalize workflow with best model
### - train model (model fit) on whole train data
tree_wflow_fin <- finalize_workflow(tree_wflow, tree_mod_best)
tree_mod_fit <- fit(tree_wflow_fin, df.train)


### visualize final tree
final_tree_fit <- extract_fit_parsnip(tree_mod_fit)$fit
rpart.plot(final_tree_fit,         # your fitted rpart model
           main = "Final Decision Tree — Pima Diabetes Classification",
           type = 3,               # draw split labels and class probabilities (or mean if regression) *at the nodes*
           fallen.leaves = TRUE,   # leaves are placed at the bottom
           digits = 3,             # number of digits to display (3 is enough for clarity)
           roundint = FALSE,       # do not round numbers to integers (better for continuous predictions)
           box.palette = "RdYlGn", # nice color gradient from red to green (works for both regression and classification)
           shadow.col = "gray",    # subtle shadow under boxes for better contrast
           branch.lty = 1,         # solid branches (line type = 1), standard choice
           extra = 101,            # show extra information at nodes (predicted value + percentage)
           cex = 0.7               # text shrinkage if tree is large
)



## Model Testing / Validation

### evaluate model performance
### - on test data (AUC metric)
### - first predict output on test data
### - then extract AUC
df.test_tree_pred <- predict(tree_mod_fit, 
                             df.test, type = "prob") %>% 
  bind_cols(df.test) %>% 
  select(type, .pred_1)

test_AUC_tree <- pr_auc(data = df.test_tree_pred, 
                       truth = type, 
                       .pred_1) 
test_AUC_tree



# 2.5 Random Forest Classifier

## Model Training

### model specification
### - "ranger" algorithm
### - we tune (hyperpar.):
###   - mtry ~ number of predictors (features) sampled at each split
###   - min_n ~ min number of observations per node
###   - trees ~ number of trees in the forest
rf_mod <- rand_forest(mode = "classification",
                      mtry = tune(),
                      min_n = tune(),
                      trees = tune()) %>%
  set_engine("ranger")


### recipe
### - we predict 'type' variable
### - we impute potential missing values
rf_rec <- recipe(type ~ ., data = df.train) %>%
  step_impute_median(all_numeric_predictors())

### workflow 
rf_wflow <- workflow() %>%
  add_recipe(rf_rec) %>%
  add_model(rf_mod)


### prepare grid for tuning
### - we will use regular grid
### - update mtry hyperpar. with number of predictors (max 7, min 2)
param_set <- extract_parameter_set_dials(rf_wflow) %>%
  update(mtry = mtry(c(2L, 7L)))

rf_grid_regular <-  grid_regular(param_set, 
                                 levels = c(mtry = 6, 
                                            min_n = 7, 
                                            trees = 7))


### classification metrics to be collected
class_metrics <- metric_set(
  accuracy, bal_accuracy,     
  precision, recall, f_meas,           
  roc_auc, pr_auc,           
  mcc, j_index
)


### hyperpar. tuning
### - execute tuning using CV data split & grid
### - we will tune using parallel computing on multiple CPU-s
### - we must register backend for foreach  
### - we must specify cores for parallel computing
registerDoFuture()                                        # parallel backend for foreach
plan(multisession, workers = parallel::detectCores() - 1) # use multi-cores
future::nbrOfWorkers()                                    # check how many workers are active

set.seed(235)

rf_tune_rez <- tune_grid(rf_wflow,
                         resamples = cv_folds,
                         grid = rf_grid_regular,
                         metrics = class_metrics,
                         control = control_grid(verbose = TRUE,
                                                parallel_over = "everything",
                                                save_pred = T)
)


### select best model 
### - model with highest AUC (assessment folds)
### - and selected value for neighbors hyperpar.
rf_mod_best <- select_best(rf_tune_rez, metric = "roc_auc")

### check training AUC 
### - filter best model predictions (assessment folds)
### - show AUC per each assessment fold 
### - and AUC calculated using all assessment folds
pred_tune_rez <- collect_predictions(rf_tune_rez)

best_AUC_folds <-inner_join(x = pred_tune_rez,
                            y = rf_mod_best %>% select(.config),
                            by = ".config") %>%
  group_by(id) %>% 
  roc_auc(type, .pred_1) %>% 
  ungroup()

best_AUC <-inner_join(x = pred_tune_rez,
                      y = rf_mod_best %>% select(.config),
                      by = ".config") %>% 
  roc_auc(type, .pred_1) 

best_AUC_folds
best_AUC


### finalize model
### - finalize workflow with best model
### - train model (model fit) on whole train data
rf_wflow_fin <- finalize_workflow(rf_wflow, rf_mod_best)
rf_mod_fit   <- fit(rf_wflow_fin, df.train)



## Model Testing / Validation

### evaluate model performance
### - on test data (AUC metric)
### - first predict output on test data
### - then extract AUC
df.test_rf_pred <- predict(rf_mod_fit, 
                           df.test, type = "prob") %>% 
  bind_cols(df.test) %>% 
  select(type, .pred_1)

test_AUC_rf <- roc_auc(data = df.test_rf_pred, 
                       truth = type, 
                       .pred_1) 
test_AUC_rf



# 2.7 Gradient Boosted Trees

## Model Training

### model specifications
### - XGBoost / LigthGBM
###   - trees ~ number of trees (learners)
###   - tree_depth ~ max depth of each tree
###   - learn_rate ~ learning rate in boosting step
xgb_mod <- boost_tree(mode = "classification",
                      trees = tune(),
                      tree_depth = tune(),
                      learn_rate = tune()) %>%
  set_engine("xgboost")

lgbm_mod <- boost_tree(mode = "classification",
                       trees = tune(),
                       tree_depth = tune(),
                       learn_rate = tune()) %>%
  set_engine("lightgbm")


### recipe
### - we predict 'type' variable
### - we impute potential missing values
xgb_lgbm_rec <- recipe(type ~ ., data = df.train) %>%
  step_impute_median(all_numeric_predictors())


### workflows 
xgb_wflow <- workflow() %>%
  add_recipe(xgb_lgbm_rec) %>%
  add_model(xgb_mod)

lgbm_wflow <- workflow() %>%
  add_recipe(xgb_lgbm_rec) %>%
  add_model(lgbm_mod)


### prepare grid for tuning
### - we will use regular grid
xgb_lgbm_grid_regular <-  grid_regular(trees(),
                                       tree_depth(),
                                       learn_rate(), 
                                       levels = 7)


### classification metrics to be collected
class_metrics <- metric_set(
  accuracy, bal_accuracy,     
  precision, recall, f_meas,           
  roc_auc, pr_auc,           
  mcc, j_index
)


### hyperpar. tuning
### - execute tuning using CV data split & grid
### - we will tune using parallel computing on multiple CPU-s
### - we must register backend for foreach  
### - we must specify cores for parallel computing
registerDoFuture()                                        # parallel backend for foreach
plan(multisession, workers = parallel::detectCores() - 1) # use multi-cores
future::nbrOfWorkers()                                    # check how many workers are active

set.seed(235)

xgb_tune_rez <- tune_grid(xgb_wflow,
                          resamples = cv_folds,
                          grid = xgb_lgbm_grid_regular,
                          metrics = class_metrics,
                          control = control_grid(verbose = TRUE,
                                                 parallel_over = "everything",
                                                 save_pred = T)
)

lgbm_tune_rez <- tune_grid(lgbm_wflow,
                           resamples = cv_folds,
                           grid = xgb_lgbm_grid_regular,
                           metrics = class_metrics,
                           control = control_grid(verbose = TRUE,
                                                  parallel_over = "everything",
                                                  save_pred = T)
)



### select best model 
### - model with highest AUC (assessment folds)
### - and selected value for neighbors hyperpar.
xgb_mod_best  <- select_best(xgb_tune_rez,  metric = "roc_auc")
lgbm_mod_best <- select_best(lgbm_tune_rez, metric = "roc_auc")

### check training AUC 
### - filter best model predictions (assessment folds)
### - and AUC calculated using all assessment folds
xgb_pred_tune_rez  <- collect_predictions(xgb_tune_rez)
lgbm_pred_tune_rez <- collect_predictions(lgbm_tune_rez)

inner_join(x = xgb_pred_tune_rez,
           y = xgb_mod_best %>% select(.config),
           by = ".config") %>% 
  roc_auc(type, .pred_1) 

inner_join(x = lgbm_pred_tune_rez,
           y = lgbm_mod_best %>% select(.config),
           by = ".config") %>% 
  roc_auc(type, .pred_1) 


### finalize model
### - finalize workflow with best model
### - train model (model fit) on whole train data
xgb_wflow_fin <- finalize_workflow(xgb_wflow, xgb_mod_best)
xgb_mod_fit   <- fit(xgb_wflow_fin, df.train)

lgbm_wflow_fin <- finalize_workflow(lgbm_wflow, lgbm_mod_best)
lgbm_mod_fit   <- fit(lgbm_wflow_fin, df.train)



## Model Testing / Validation

### evaluate model performance
### - on test data (AUC metric)
### - first predict output on test data
### - then extract AUC
df.test_xgb_pred <- predict(xgb_mod_fit, 
                            df.test, type = "prob") %>% 
  bind_cols(df.test) %>% 
  select(type, .pred_1)

df.test_lgbm_pred <- predict(lgbm_mod_fit, 
                             df.test, type = "prob") %>% 
  bind_cols(df.test) %>% 
  select(type, .pred_1)

roc_auc(data = df.test_xgb_pred, truth = type, .pred_1) 
roc_auc(data = df.test_lgbm_pred, truth = type, .pred_1) 
