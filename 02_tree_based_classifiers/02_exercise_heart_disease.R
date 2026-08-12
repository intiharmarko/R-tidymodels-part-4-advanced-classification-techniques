# Exercise - Heart Disease

rm(list = ls())
graphics.off()

# Install packages
#install.packages("kmed")

# Load packages
library(tidyverse)
library(tidymodels)
library(ranger)
library(future)
library(doFuture)
library(kmed)



# Data (Heart disease data)

## load data
df <- kmed::heart; ?kmed::heart

## recode response variable
## - class == 0: no heart disease
## - class  > 0: heart disease
df <- df %>% 
  mutate(h_disease = if_else(class == 0, 0, 1),
         h_disease = factor(h_disease, levels = c(1, 0))) %>% 
  select(-class) %>% 
  select(h_disease, everything())


## split data
## - train    80%
## - test     20%
## - we don't have severely imbalanced data (so stratification is not needed)
set.seed(1123)

split_init <- initial_split(df, 
                            prop = 0.8)
df.train <- training(split_init)
df.test  <- testing(split_init)

## set up CV for hyperpar. tuning
## - we will use k=10 fold CV
## - we don't have severely imbalanced data (so stratification is not needed)
cv_folds <- vfold_cv(df.train, v = 10)



# EDA (Quick)

## data structure & size
str(df)
nrow(df)
ncol(df)

## missing rows
map(df, ~sum(is.na(.)))

## target variable distribution
df %>% 
  count(h_disease) %>% 
  mutate(`%` = round(n / sum(n) * 100, 2))

## pair plot
GGally::ggpairs(df)



## Model Training

### model specification
rf_mod <- rand_forest(mode = "classification",
                      mtry = tune(),
                      min_n = tune(),
                      trees = tune()) %>%
  set_engine("ranger")


### recipe
rf_rec <- recipe(h_disease ~ ., data = df.train)

### workflow 
rf_wflow <- workflow() %>%
  add_recipe(rf_rec) %>%
  add_model(rf_mod)


### classification metrics to be collected
class_metrics <- metric_set(
  accuracy, bal_accuracy,     
  precision, recall, f_meas,           
  roc_auc, pr_auc,           
  mcc, j_index
)


### hyperpar. tuning
registerDoFuture()                                        
plan(multisession, workers = parallel::detectCores() - 1) 
future::nbrOfWorkers()                                    

set.seed(235)

#### extract and finalize the parameter set
par_final <- extract_parameter_set_dials(rf_mod) %>%
  finalize(df.train %>% select(-h_disease))

rf_tune_rez <- tune_bayes(
  rf_wflow,
  resamples = cv_folds,
  metrics = class_metrics,
  initial = 5,                              
  iter = 25,                                
  param_info = par_final,                    
  control = control_bayes(verbose = TRUE,
                          parallel_over = "everything", 
                          save_pred = T, 
                          no_improve = 7)
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
  roc_auc(h_disease, .pred_1) %>% 
  ungroup()

best_AUC <-inner_join(x = pred_tune_rez,
                      y = rf_mod_best %>% select(.config),
                      by = ".config") %>% 
  roc_auc(h_disease, .pred_1) 

best_AUC_folds
best_AUC


### finalize model
### - finalize workflow with best model
### - train model (model fit) on whole train data
rf_wflow_fin <- finalize_workflow(rf_wflow, rf_mod_best)
rf_mod_fit   <- fit(rf_wflow_fin, df.train)


## Visualize training results 

### Visualize tuning results
### - draw tuning hyperparameter results
### - each hyperparameter placed in its own facet
### - use facet grid to create facets for each hyperparameter 
### - each hyperparameter value VS AUC
### - use tuning results (during training - CV)
rf_tune_rez %>% 
  collect_metrics() %>% 
  filter(.metric == "roc_auc") %>% 
  select(mtry,
         min_n,
         trees,
         mean, 
         std_err) %>% 
  pivot_longer(cols = mtry:trees,
               names_to = "par",
               values_to = "par_value") %>% 
  ggplot(aes(x = par_value, 
             y = mean)) +
  geom_point(size = 4, 
             color = "brown3") +
  geom_line(color = "gray20", 
            linewidth = 1) +
  geom_errorbar(aes(ymin = mean - std_err, 
                    ymax = mean + std_err), 
                width = 0.5, 
                alpha = 0.5) +
  facet_grid(cols = vars(par),
             scales = "free",
             labeller = "label_both") +
  labs(title = "Random forest classifier tuning: AUC vs. hyperparameter values",
       x = "Hyperparameter value",
       y = "AUC") +
  theme_minimal(base_size = 16)


### selected values for hyperparam.
rf_mod_fit$fit$actions$model$spec


### visualize training AUC
### - show how AUC changes over assessment folds
### - draw average AUC over all assessment folds
best_AUC_folds %>% 
  ggplot(aes(x = as.factor(id),
             y = .estimate)) +
  geom_col(color = "black",
           fill = "gray60") +
  geom_hline(yintercept = best_AUC$.estimate,
             color = "brown3",
             linewidth = 1.5,
             linetype = "dashed") +
  scale_y_continuous(breaks = seq(0, 1, 0.05)) +
  xlab("Assessment fold") +
  ylab("AUC") +
  ggtitle("Random forest classifier: AUC over assessment folds") +
  labs(subtitle = "Dashed line: average AUC over all assessment folds") +
  theme_minimal(base_size = 16)



## Model Testing / Validation

### evaluate model performance
### - on test data (AUC metric)
### - first predict output on test data
### - then extract AUC
df.test_rf_pred <- predict(rf_mod_fit, 
                           df.test, 
                           type = "prob") %>% 
  bind_cols(df.test) %>% 
  select(h_disease, .pred_1)

test_AUC_rf <- roc_auc(data = df.test_rf_pred, 
                       truth = h_disease, 
                       .pred_1) 
test_AUC_rf
