# Exercise - Multiclass Classification

rm(list = ls())
graphics.off()

# Install packages
#install.packages("HDclassif")

# Load packages
library(tidyverse)
library(tidymodels)
library(HDclassif)



# Data (Wine data)

## load data
data(wine); ?wine
df <- wine

## convert response variable to factor
df <- df %>% 
  mutate(class = factor(class, levels = c(1, 2, 3))) 

## add names to columns
df <- df %>% 
  rename(`Alcohol`              = V1,
         `Malic acid`           = V2,
         `Ash`                  = V3,
         `Alcalinity of ash`    = V4,
         `Magnesium`            = V5,
         `Total phenols`        = V6,
         `Flavanoids`           = V7,
         `Nonflavanoid phenols` = V8,
         `Proanthocyanins`      = V9,
         `Color intensity`      = V10,
         `Hue`                  = V11,
         `OD280/OD315`          = V12,
         `Proline`              = V13)


## split data
## - train    80%
## - test     20%
## - we don't have severely imbalanced data (so stratification is not needed)
set.seed(1123)

split_init <- initial_split(df, prop = 0.8)
df.train <- training(split_init)
df.test  <- testing(split_init)

## set up CV for hyperpar. tuning
## - we will use k=5 fold CV
## - we don't have severely imbalanced data (so stratification is not needed)
cv_folds <- vfold_cv(df.train, v = 5)



# EDA (Quick)

## data structure & size
str(df)
nrow(df)
ncol(df)

## missing rows
map(df, ~sum(is.na(.)))

## target variable distribution
df %>% 
  count(class) %>% 
  mutate(`%` = round(n / sum(n) * 100, 2))

## pair plot
GGally::ggpairs(df)



# Model Training

## model specification
## - we will use penalized logistic multiclass regression model
## - we tune penalty and mixture hyperpar.
mod <- multinom_reg(penalty = tune(), 
                    mixture = tune()) %>%
  set_engine("glmnet") %>% 
  set_mode("classification")


## recipe
## - we only normalize all numeric features
rec <- recipe(formula = class ~ ., data = df.train) %>%
  step_normalize(all_numeric_predictors()) 


## create workflow 
wflow <- workflow() %>%
  add_recipe(rec) %>%
  add_model(mod)


## classification metrics (collected during tuning phase)
class_metrics <- metric_set(accuracy, roc_auc)

## regular grid (hyperpar. tuning)
reg_grid <- grid_regular(
  penalty(),
  mixture(),
  levels = c(penalty = 7, 
             mixture = 5))

## hyperpar. tuning
set.seed(1123)

tune_rez <- tune_grid(
  wflow,
  resamples = cv_folds,
  grid      = reg_grid,
  metrics   = class_metrics,
  control   = control_grid(save_pred = TRUE))


## select best model 
## - model with highest AUC (assessment folds)
mod_best <- select_best(tune_rez, metric = "roc_auc")


## finalize model
## - finalize workflow with best model
## - train model (model fit) on whole train data
wflow_fin <- finalize_workflow(wflow, mod_best)
mod_fit   <- fit(wflow_fin, df.train)



# Model Testing

## generate predictions on test data
## - use final model fit
## - and predict classes & class probabilities for test data
df.test_pred <- bind_cols(predict(mod_fit, df.test, type = "prob"),
                          predict(mod_fit, df.test, type = "class"),
                          df.test %>% dplyr::select(`class`))


## calculate metrics

### metrics set
class_multi_metrics <- metric_set(
  accuracy,
  bal_accuracy,
  precision, 
  recall, 
  f_meas
)

### "macro" averaging strategy
macro_metrics_rez <- class_multi_metrics(
  df.test_pred, 
  truth = class, 
  estimate = .pred_class,
  estimator = "macro"
)

### "micro" averaging strategy
micro_metrics_rez <- class_multi_metrics(
  df.test_pred, 
  truth = class,
  estimate = .pred_class,
  estimator = "micro"
)

### "macro_weighted" averaging strategy
weighted_metrics_rez <- class_multi_metrics(
  df.test_pred, 
  truth = class, 
  estimate = .pred_class,
  estimator = "macro_weighted"
)


### Multiclass AUC

#### Hand–Till
auc_hand_till <- df.test_pred %>%
  roc_auc(truth = class,
          .pred_1:.pred_3,
          estimator = "hand_till")

#### Macro AUC
auc_macro <- df.test_pred %>%
  roc_auc(truth = class,
          .pred_1:.pred_3,
          estimator = "macro")

# Macro-weighted AUC
auc_macro_w <- df.test_pred %>%
  roc_auc(truth = class,
          .pred_1:.pred_3,
          estimator = "macro_weighted")


### Visualize metrics

bind_rows(macro_metrics_rez,
          micro_metrics_rez,
          weighted_metrics_rez,
          auc_hand_till,
          auc_macro,
          auc_macro_w) %>% 
  distinct() %>% 
  mutate(.metric = factor(.metric, 
                          levels = c("accuracy", "bal_accuracy", 
                                     "precision", "recall", "f_meas", 
                                     "roc_auc")),
         .estimator = factor(.estimator, 
                             levels = c("multiclass", 
                                        "macro", 
                                        "micro", 
                                        "macro_weighted", 
                                        "hand_till"))) %>% 
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
  ggtitle("Multiclass metrics - Penalized multiclass regression model (wine ~ test data)") +
  theme_minimal(base_size = 16)


### Visualize ROC curves (one-vs-rest) for all three classes
df.test_pred %>%
  roc_curve(truth = class,
            .pred_1:.pred_3) %>% 
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
  ggtitle("ROC Curve - Penalized Multiclass Regression Model") +
  labs(subtitle = "Wine test data",
       color = "Class:") +
  theme_minimal(base_size = 16)
