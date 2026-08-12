# 3 Multiclass Classification

rm(list = ls())
graphics.off()

# Load packages
library(tidyverse)
library(tidymodels)
library(nnet)
library(kknn)
library(rpart)
library(rpart.plot)
library(ranger)
library(xgboost)
library(bonsai)
library(lightgbm)
library(future)
library(doFuture)



# 3.3 Multiclass Logistic Regression

## data - iris
df <- iris

## split data
## - train    80%
## - test     20%
set.seed(1123)

split_init <- initial_split(df, prop = 0.8)
df.train   <- training(split_init)
df.test    <- testing(split_init)



## Model Training

### define model specification
### - softmax (multinomial) logistic regression
### - no hyperpar. to tune
smax_mod <- multinom_reg(mode = "classification") %>%
  set_engine("nnet")

## define recipe
## - we predict "Species" target variable with three classes
## - all numeric predictors — normalization is optional but recommended
smax_rec <- recipe(formula = Species ~ ., data = df.train) %>%
  step_normalize(all_numeric_predictors())

## create workflow 
smax_wflow <- workflow() %>%
  add_recipe(smax_rec) %>%
  add_model(smax_mod)


## model fit
smax_fit <- fit(smax_wflow, data = df.train)
smax_fit


## check training results 
## - first we generate predictions on training data
## - then we collect accuracy & confusion matrix

### predict (class)
df.train_pred <- predict(smax_fit, new_data = df.train) %>%
  bind_cols(df.train %>% select(Species))

### collect metrics
conf_mat(df.train_pred, truth = Species, estimate = .pred_class)
accuracy(df.train_pred, truth = Species, estimate = .pred_class)



## Model Testing
## - predict target on test data
## - collect metrics

### predict (class)
df.test_pred <- predict(smax_fit, new_data = df.test) %>%
  bind_cols(df.test %>% select(Species))

### collect metrics
conf_mat(df.test_pred, truth = Species, estimate = .pred_class)
accuracy(df.test_pred, truth = Species, estimate = .pred_class)

### probabilities (prediction)
predict(smax_fit, new_data = df.test, type = "prob")



# 3.4 Metrics for Multiclass Classification

## metrics set
class_multi_metrics <- metric_set(
  accuracy,
  precision, 
  recall, 
  f_meas
)


## "macro" averaging strategy
macro_metrics_rez <- class_multi_metrics(
  df.test_pred, 
  truth = Species, 
  estimate = .pred_class,
  estimator = "macro"
)

macro_metrics_rez


## "micro" averaging strategy
micro_metrics_rez <- class_multi_metrics(
  df.test_pred, 
  truth = Species, 
  estimate = .pred_class,
  estimator = "micro"
)

micro_metrics_rez


## "macro_weighted" averaging strategy
weighted_metrics_rez <- class_multi_metrics(
  df.test_pred, 
  truth = Species, 
  estimate = .pred_class,
  estimator = "macro_weighted"
)

weighted_metrics_rez


## visualize all metrics
bind_rows(macro_metrics_rez,
          micro_metrics_rez,
          weighted_metrics_rez) %>% 
  distinct() %>% 
  mutate(.metric = factor(.metric, 
                          levels = c("accuracy", "precision", "recall", "f_meas")),
         .estimator = factor(.estimator, 
                          levels = c("multiclass", "macro", "micro", "macro_weighted"))) %>% 
  ggplot(aes(x = .metric,
             y = .estimate,
             fill = .estimator,
             label = round(.estimate, 4))) +
  geom_col(color = "black", 
           show.legend = F) +
  geom_text(size = 6) +
  facet_grid(rows = vars(.estimator)) +
  xlab("Metric") +
  ylab("Metric value") +
  ggtitle("Multiclass metrics - Softmax model (Iris ~ test data)") +
  theme_minimal(base_size = 16)



# 3.5 Multiclass ROC & AUC

## get class probabilities on test set
df.test_pred <- bind_cols(predict(smax_fit, new_data = df.test, type = "prob"),
                          predict(smax_fit, new_data = df.test, type = "class"),
                          df.test) %>%
  rename(.pred_class    = .pred_class,
         .pred_setosa   = .pred_setosa,
         .pred_versi    = .pred_versicolor,
         .pred_virginica = .pred_virginica)


## OvR ROC curves (one-vs-rest) for all three classes
## - roc_curve() with a multiclass outcome 
## - produces one-vs-rest ROC curves 
## - for each class (column .level)
df.test_roc_ovr <- df.test_pred %>%
  roc_curve(truth = Species,
            .pred_setosa:.pred_virginica) # all three class prob columns

df.test_roc_ovr

## plot all three curves
df.test_roc_ovr %>% 
  autoplot()

df.test_roc_ovr %>% 
  mutate(TPR = sensitivity,
         FPR = 1 - specificity) %>% 
  arrange(FPR, TPR) %>% 
  ggplot(aes(x = FPR,
             y = TPR,
             color = .level)) +
  geom_point(size = 5) +
  geom_line(linewidth = 1.2) +
  facet_wrap(vars(.level)) +
  scale_x_continuous(breaks = seq(0,1,0.1), 
                     limits = c(0,1)) +
  scale_y_continuous(breaks = seq(0,1,0.05), 
                     limits = c(0,1)) +
  xlab("False positive rate (FPR) : 1 - specificity") +
  ylab("True positive rate (TPR) : sensitivity") +
  ggtitle("ROC Curve - Softmax regression model") +
  labs(subtitle = "Iris test data",
       color = "Class:") +
  theme_minimal(base_size = 16)

## yardstick doesn’t directly give OvO (one-vs-one)
## ROC curves in one call, but you can build them 
## easily by filtering to two classes and treating 
## it as a binary problem!


## Multiclass AUC
## - yardstick::roc_auc() supports several multiclass estimators:
##   - estimator = "hand_till" → Hand–Till method, a pairwise (OvO-style) multiclass AUC (this is also the default).
##   - estimator = "macro" → uniform class weights 
##   - estimator = "macro_weighted" → weighted by class frequencies 

### Hand–Till (OvO-style multiclass AUC, default)
auc_hand_till <- df.test_pred %>%
  roc_auc(truth = Species,
          .pred_setosa:.pred_virginica, # probs for all levels
          estimator = "hand_till")

auc_hand_till

# Macro AUC (uniform class weights; OvR-based)
auc_macro <- df.test_pred %>%
  roc_auc(truth = Species,
          .pred_setosa:.pred_virginica,
          estimator = "macro")

auc_macro

# Macro-weighted AUC (class-frequency weights; OvR-based)
auc_macro_w <- df.test_pred %>%
  roc_auc(truth = Species,
          .pred_setosa:.pred_virginica,
          estimator = "macro_weighted")

auc_macro_w



# 3.7 Multiclass KNN Classifier

## set up CV for hyperpar. tuning
## - we will use k=10 fold CV
## - we don't have severely imbalanced data (so stratification is not needed)
cv_folds <- vfold_cv(df.train, v = 10)


## Model Training

### define model specification
### - KNN algorithm
### - we tune (hyperpar.):
###   - neighbors ~ K (number of neighbors)
knn_mod <- nearest_neighbor(mode = "classification",
                            neighbors = tune()) %>%
  set_engine("kknn")

### define recipe
### - we predict "Species" target variable with three classes
### - all numeric predictors — normalization is applied
knn_rec <- recipe(formula = Species ~ ., data = df.train) %>%
  step_normalize(all_numeric_predictors())

### create workflow 
knn_wflow <- workflow() %>%
  add_recipe(knn_rec) %>%
  add_model(knn_mod)


### prepare grid for tuning
### - we will use regular grid
### - n=20 different values for K
### - range between: 1 & 20
knn_grid_regular <- grid_regular(
  neighbors(range = c(1,20)),
  levels = 20)

#### classification metrics to be collected
class_metrics <- metric_set(
  accuracy, bal_accuracy,     
  precision, recall, f_meas,           
  roc_auc
)

### hyperpar. tuning
### - execute tuning using CV data split & grid
### - we will tune using parallel computing on multiple CPU-s
registerDoFuture()                                        # parallel backend for foreach
plan(multisession, workers = parallel::detectCores() - 1) # use multi-cores
future::nbrOfWorkers()                                    # check how many workers are active

set.seed(235)

knn_tune_rez <- tune_grid(knn_wflow,
                          resamples = cv_folds,
                          grid = knn_grid_regular,
                          metrics = class_metrics,
                          control = control_grid(verbose = TRUE,
                                                 parallel_over = "everything", 
                                                 save_pred = T)
)


### select best model 
### - model with highest AUC (assessment folds)
### - and selected value for neighbors hyperpar.
knn_mod_best <- select_best(knn_tune_rez, metric = "roc_auc")


### finalize model
### - finalize workflow with best model
### - train model (model fit) on whole train data
knn_wflow_fin <- finalize_workflow(knn_wflow, knn_mod_best)
knn_mod_fit   <- fit(knn_wflow_fin, df.train)



## Model Testing / Validation

### evaluate model performance
### - on test data (AUC metric)
### - first predict output on test data
### - then extract AUC

### get class probabilities on test set
df.test_pred <- bind_cols(predict(knn_mod_fit, new_data = df.test, type = "prob"),
                          predict(knn_mod_fit, new_data = df.test, type = "class"),
                          df.test) %>%
  rename(.pred_class    = .pred_class,
         .pred_setosa   = .pred_setosa,
         .pred_versi    = .pred_versicolor,
         .pred_virginica = .pred_virginica)

### Hand–Till multiclass AUC
df.test_pred %>%
  roc_auc(truth = Species,
          .pred_setosa:.pred_virginica,
          estimator = "hand_till")



# 3.8 Multiclass Tree-Based Models


## Model Training

### define model specifications (4 models)
### - decision tree classifier
### - random forest classifier
### - XGBoost classifier
### - LightGBM classifier
tree_mod <- decision_tree(mode = "classification",
                          min_n = tune(),
                          tree_depth = tune(),
                          cost_complexity = tune()) %>%
  set_engine("rpart")

rf_mod <- rand_forest(mode = "classification",
                      mtry = tune(),
                      min_n = tune(),
                      trees = tune()) %>%
  set_engine("ranger")

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

### define recipe
### - we predict "Species" target variable with three classes
tree_based_rec <- recipe(Species ~ ., data = df.train)

### create workflows
tree_wflow <- workflow() %>%
  add_recipe(tree_based_rec) %>%
  add_model(tree_mod)

rf_wflow <- workflow() %>%
  add_recipe(tree_based_rec) %>%
  add_model(rf_mod)

xgb_wflow <- workflow() %>%
  add_recipe(tree_based_rec) %>%
  add_model(xgb_mod)

lgbm_wflow <- workflow() %>%
  add_recipe(tree_based_rec) %>%
  add_model(lgbm_mod)


### prepare grids for tuning
### - we will use regular grid
### - update mtry hyperpar. with number of predictors (max 7, min 2)
tree_grid_regular <- grid_regular(
  cost_complexity(),
  tree_depth(range = c(1L, 12L)),
  min_n(range = c(2L, 40L)),
  levels = 5
)

param_set <- extract_parameter_set_dials(rf_wflow) %>%
  update(mtry = mtry(c(2L, 4L)))

rf_grid_regular <-  grid_regular(param_set, 
                                 levels = c(mtry = 3, 
                                            min_n = 5, 
                                            trees = 5))

xgb_lgbm_grid_regular <-  grid_regular(trees(),
                                       tree_depth(),
                                       learn_rate(), 
                                       levels = 5)

#### classification metrics to be collected
class_metrics <- metric_set(
  accuracy, bal_accuracy,     
  precision, recall, f_meas,           
  roc_auc
)

### hyperpar. tuning
### - execute tuning using CV data split & grid
### - we will tune using parallel computing on multiple CPU-s
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

rf_tune_rez <- tune_grid(rf_wflow,
                         resamples = cv_folds,
                         grid = rf_grid_regular,
                         metrics = class_metrics,
                         control = control_grid(verbose = TRUE,
                                                parallel_over = "everything",
                                                save_pred = T)
)

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
tree_mod_best <- select_best(tree_tune_rez, metric = "roc_auc")
rf_mod_best   <- select_best(rf_tune_rez,   metric = "roc_auc")
xgb_mod_best  <- select_best(xgb_tune_rez,  metric = "roc_auc")
lgbm_mod_best <- select_best(lgbm_tune_rez, metric = "roc_auc")


### finalize model
### - finalize workflows with best model
### - train models (model fit) on whole train data
tree_wflow_fin <- finalize_workflow(tree_wflow, tree_mod_best)
rf_wflow_fin   <- finalize_workflow(rf_wflow,   rf_mod_best)
xgb_wflow_fin  <- finalize_workflow(xgb_wflow,  xgb_mod_best)
lgbm_wflow_fin <- finalize_workflow(lgbm_wflow, lgbm_mod_best)

tree_mod_fit <- fit(tree_wflow_fin, df.train)
rf_mod_fit   <- fit(rf_wflow_fin,   df.train)
xgb_mod_fit  <- fit(xgb_wflow_fin,  df.train)
lgbm_mod_fit <- fit(lgbm_wflow_fin, df.train)

### visualize final tree
final_tree_fit <- extract_fit_parsnip(tree_mod_fit)$fit
rpart.plot(final_tree_fit,         # your fitted rpart model
           main = "Final Decision Tree — Iris Flowers Classification",
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

### decision tree classifier
df.test_pred <- bind_cols(predict(tree_mod_fit, new_data = df.test, type = "prob"),
                          predict(tree_mod_fit, new_data = df.test, type = "class"),
                          df.test) %>%
  rename(.pred_class     = .pred_class,
         .pred_setosa    = .pred_setosa,
         .pred_versi     = .pred_versicolor,
         .pred_virginica = .pred_virginica)

df.test_pred %>%
  roc_auc(truth = Species,
          .pred_setosa:.pred_virginica,
          estimator = "hand_till")

### random forest classifier
df.test_pred <- bind_cols(predict(rf_mod_fit, new_data = df.test, type = "prob"),
                          predict(rf_mod_fit, new_data = df.test, type = "class"),
                          df.test) %>%
  rename(.pred_class     = .pred_class,
         .pred_setosa    = .pred_setosa,
         .pred_versi     = .pred_versicolor,
         .pred_virginica = .pred_virginica)

df.test_pred %>%
  roc_auc(truth = Species,
          .pred_setosa:.pred_virginica,
          estimator = "hand_till")

### XGBoost classifier
df.test_pred <- bind_cols(predict(xgb_mod_fit, new_data = df.test, type = "prob"),
                          predict(xgb_mod_fit, new_data = df.test, type = "class"),
                          df.test) %>%
  rename(.pred_class     = .pred_class,
         .pred_setosa    = .pred_setosa,
         .pred_versi     = .pred_versicolor,
         .pred_virginica = .pred_virginica)

df.test_pred %>%
  roc_auc(truth = Species,
          .pred_setosa:.pred_virginica,
          estimator = "hand_till")

### LightGBM classifier
df.test_pred <- bind_cols(predict(lgbm_mod_fit, new_data = df.test, type = "prob"),
                          predict(lgbm_mod_fit, new_data = df.test, type = "class"),
                          df.test) %>%
  rename(.pred_class     = .pred_class,
         .pred_setosa    = .pred_setosa,
         .pred_versi     = .pred_versicolor,
         .pred_virginica = .pred_virginica)

df.test_pred %>%
  roc_auc(truth = Species,
          .pred_setosa:.pred_virginica,
          estimator = "hand_till")
