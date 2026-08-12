#' 4 Tidymodels Capstone Project: Forest Cover Classification - Modeling - Functions


library(tidyverse)
library(tidymodels)
library(workflowsets)



#' Finalize workflow-specific parameter sets
#'
#' This function finalizes the tunable parameter ranges for each workflow
#' contained in a workflow set:
#'  - each workflow is matched with an appropriate recipe 
#'   (either the base recipe or the normalized recipe)
#'  - and the recipe is prepped and baked on the provided training data.  
#'
#' The baked training data is then used to finalize the parameter objects 
#' associated with each workflow. This ensures that the parameter ranges 
#' (e.g., mtry, trees, penalty) reflect the actual set of predictors produced 
#' by preprocessing steps such as imputation, dummy encoding, normalization, 
#' or zero-variance removal.  
#'
#' @param wflow_set_ A workflow set.
#' @param rec_list_ A list of recipes. 
#' @param df.train_ A training data frame.
#'
#' @return par_fin_list A named list of finalized parameter sets, one per each workflow id.
#' 
finalize_wflow_params <- function(wflow_set_ = wflow_set, 
                                  rec_list_ = rec_list, 
                                  df.train_) {
  
  # get workflow ids
  w_ids <- wflow_set_$wflow_id
  
  par_fin_list <- purrr::map(w_ids, function(id) {
    
    # decide which non-sampling recipe to use for this workflow
    # log_* workflows  -> use normalized recipe
    # tree-based       -> use base recipe
    rec_name <- if (stringr::str_detect(id, "_log$")) {
      "norm"      # rec_norm: impute + normalize
    } else {
      "base"      # rec_basic: impute only
    }
    
    # prep & bake that recipe on full training data
    rec   <- recipes::prep(rec_list_[[rec_name]], training = df.train_)
    df_baked <- recipes::bake(rec, new_data = NULL)
    
    # extract workflow + its parameters
    wflow  <- workflowsets::extract_workflow(wflow_set_, id = id)
    params <- workflows::extract_parameter_set_dials(wflow)
    
    dials::finalize(params, df_baked)
  })
  
  names(par_fin_list) <- w_ids
  return(par_fin_list)
}



#' Tune hyperparameters of multiple algorithms with Bayes optimization tuning algorithm
#' 
#' This functions runs tuning hyperparameters of selected models (different algorithms)
#' using Bayesian optimization tuning algorithm. New model candidates are generated
#' based on tuning parameter combinations from previous results (tuning iterations).
#' It is custom made function based on tune_bayes(), which can be applied for multiple
#' modeling paradigms - algorithm types in single run.
#' 
#' @param wflow_set_ A tible of workflows objects per each model.
#' @param par_fin_list_ A list of finalized features per each model.
#' @param cv_folds_ Train data split into folds (Cross-validation object) / also supports date based CV objects.
#' @param iter The maximum number of search iterations.
#' @param initial An initial set of results.
#' @param no_improve The integer cutoff for the number of iterations without better results.
#' 
#' @return tune_rez A list of tuning results per each model. 
#'
tune_bayes_custom <- function(wflow_set_ = wflow_set,
                              par_fin_list_ = par_fin_list,
                              cv_folds_ = cv_folds,
                              metrics_  = multi_class_metrics,
                              iter = 20,
                              initial = 5,
                              no_improve = 5){
  
  # empty list - tuning results
  tune_rez <- list()
  
  # loop over each model id
  for (id in wflow_set_$wflow_id) {
    
    print("")
    print("|--------------------------|")
    print(paste0("Tuning: ", id))
    print(paste0("Started at: ", Sys.time()))
    
    ts <- Sys.time() # track execution time
    
    # extract workflow
    wflow <- extract_workflow(wflow_set_, id)
    
    # extract finalized param set
    par_fin <- par_fin_list_[[id]]
    
    # run Bayesian tuning
    rez <- tune_bayes(
      object = wflow,
      resamples = cv_folds_,
      param_info = par_fin,
      metrics = metrics_,
      iter = iter,
      initial = initial,
      control = control_bayes(verbose = TRUE, 
                              parallel_over = "everything",
                              no_improve = no_improve, 
                              save_pred = TRUE)
    )
    
    # report execution time
    te <- Sys.time()
    t_el <- round(as.numeric(difftime(te, ts, units = "mins")), 2)
    print(paste0("Finished tuning for: ", id, " in ", t_el, " minutes."))
    print(paste0("Ended at: ", Sys.time()))
    
    # store results
    tune_rez[[id]] <- rez
  }
  
  return(tune_rez)
}



#' Save R's work space snapshot
#' 
#' This functions stores current R's work space on PC disk. 
#' 
#' @param path A string containing path to .RData file on disk.
#' 
#' @return NULL
#'
save_WS_snapshot <- function(path){
  
  save.image(file = path)
  
  cat("\nR's work space image created and stored to:")
  cat("\n", path)
}



#' Load stored R's work space snapshot
#' 
#' This functions loads stored R's work space from disk to R session. 
#' 
#' @param path A string containing path to .RData file on disk.
#' 
#' @return NULL
#'
load_WS_snapshot <- function(path){
  
  load(file = path, envir = .GlobalEnv)
  
  cat("\nR's work space image loaded from:")
  cat("\n", path)
}



#' Collect metrics (tuning results)
#' 
#' This functions collects out-of-fold predictions
#' - extracted from assessment folds
#' - each workflow / model included
#' - probability predictions
#' - we collect on two levels:
#'   - per each assessment fold
#'   - aggregate over all assessment folds
#' 
#' @param tune_rez_ A list of tuning results.
#' @param sum_met A logical of if metrics are summarized or not.
#' 
#' @return col_met.l A list of collected metrics per each model.
#'
collect_metrics_l <- function(tune_rez_ = tune_rez,
                              sum_met = F){
  
  # list for collected metrics
  col_met.l  <- list()
  
  for (model in names(tune_rez_)){
    
    # collect metrics for selected model
    col_met_ <- collect_metrics(tune_rez_[[model]], 
                                summarize = sum_met) %>% 
      mutate(wflow_id = model) %>% 
      select(wflow_id, everything())
    
    # store model's metrics into main df
    col_met.l[[model]] <- col_met_
  }
  
  return(col_met.l)
}



#' Select best model candidate.
#' 
#' This functions selects best model candidate per each model (algorithm type) using F1 score. 
#' 
#' @param tune_rez_ A list of tuning results per each model. 
#' @param wflow_set_ A tibble of workflows objects per each model.
#' @param df.train_ A training data df.
#' 
#' @return mod_fit_list A list of fitted models (best candidates per algorithm) on training data.
#'
extract_best_models <- function(tune_rez_ = tune_rez,
                                wflow_set_ = wflow_set){
  
  mod_fit_list  <- map(names(tune_rez_), function(id) {
    
    # extract best model candidate per given algorithm
    best_par <- select_best(tune_rez_[[id]], metric = "f_meas")
  })
  
  # assign model's names to list of results
  names(mod_fit_list) <- names(tune_rez_)
  
  return(mod_fit_list)
}



#' Finalize each workflow and fit best model.
#' 
#' This functions first selects best model candidate per each model (algorithm type) based on F1 score,
#' and then fits best model candidate using complete training data. 
#' 
#' @param tune_rez_ A list of tuning results per each model. 
#' @param wflow_set_ A tibble of workflows objects per each model.
#' @param df.train_ A training data df.
#' 
#' @return mod_fit_list A list of fitted models (best candidates per algorithm) on training data.
#'
fit_best_models <- function(tune_rez_ = tune_rez,
                            wflow_set_ = wflow_set,
                            df.train_ = df.training){
  
  mod_fit_list  <- map(names(tune_rez_), function(id) {
    
    # extract best model candidate per given algorithm
    best_par <- select_best(tune_rez_[[id]], metric = "f_meas")
    
    # fit final model
    extract_workflow(wflow_set_, id) %>%
      finalize_workflow(best_par) %>% 
      fit(data = df.train_)
  })
  
  # assign model's names to list of results
  names(mod_fit_list) <- names(tune_rez_)
  
  return(mod_fit_list)
}



#' Visualize tuning results
#' 
#' This function creates two plots:
#' - figure 1: F1 score per each top candidate 
#'             over all assessment folds
#' - figure 2: multiple classification metrics 
#'             aggregated over all assessment folds
#'             per each top candidate
#' 
#' @param tune_rez_met_fold_ A list of collected metrics (assessment fold level).
#' @param tune_rez_met_aggr_ A list of collected metrics (aggregated over all assessment folds).
#' @param top_mod_candidates_ A list of top model candidate per each workflow.
#' 
#' @return Null
#'
visualize_tune_rez <- function(tune_rez_met_fold_ = tune_rez_met_fold,
                               tune_rez_met_aggr_ = tune_rez_met_aggr,
                               top_mod_candidates_ = top_mod_candidates){

  # categories (for figures)
  levels.wflow   <- c("base_rf", "base_xgb", "base_lgbm", "norm_log", "down_rf", "down_xgb", "down_lgbm", "down_norm_log")
  levels.metrics <- c("f_meas", "precision", "recall", "roc_auc", "bal_accuracy", "accuracy")
  
  
  # figure 1:
  # - first collect F1 score only for top candidates
  # - metrics on fold level
  # - then we draw fold id on x-axis
  # - and ROC AUC on y-axis
  # - per each top model candidate
  
  # prepare data
  df.plot <- tibble()
  
  for(model in names(tune_rez_met_fold_)){
    
    print(model)
    
    df.plot_ <- inner_join(x = tune_rez_met_fold_[[model]],
                           y = top_mod_candidates_[[model]] %>% select(.config),
                           by = ".config") %>% 
      filter(.metric == "f_meas") %>% 
      mutate(wflow_id = factor(wflow_id, levels = levels.wflow),
             id = as.factor(id)) %>% 
      select(wflow_id, id, .metric, .estimator, .estimate, .config)
    
    df.plot <- rbind(df.plot, df.plot_)
  }
  
  # create plot
  p1 <- df.plot %>%  
    ggplot(aes(x = id,
               y = .estimate,
               color = wflow_id,
               group = wflow_id)) +
    geom_line(linewidth = 1) +
    geom_point(size = 5) +
    scale_color_viridis_d() +
    xlab("Assessment fold ID") +
    ylab("F1 score") +
    ggtitle("F1 score over assessment folds for best model candidate per workflow") +
    labs(subtitle = "Showing only best performing model candidate per each workflow (classifier - recipe)",
         color = "Workflow (model):") +
    theme_minimal(base_size = 16)
  
  
  # figure 2:
  # - first collect all classification metrics only for top candidates
  # - metrics aggregated over all assessment folds
  # - then we draw metric on x-axis
  # - and metric value on y-axis
  # - for each top model candidate
  
  # prepare data
  df.plot <- tibble()
  
  for(model in names(tune_rez_met_aggr_)){
    
    df.plot_ <- inner_join(x = tune_rez_met_aggr_[[model]],
                           y = top_mod_candidates_[[model]] %>% select(.config),
                           by = ".config") %>% 
      mutate(wflow_id = factor(wflow_id, levels = levels.wflow),
             .metric = factor(.metric, levels = levels.metrics)) %>% 
      select(wflow_id, .metric, .estimator, mean, n, std_err, .config)
    
    df.plot <- rbind(df.plot, df.plot_)
  }
  
  # create plot
  p2 <- df.plot %>% 
    ggplot(aes(x = .metric,
               y = mean,
               label = round(mean, 4),
               fill = wflow_id)) +
    geom_col(color = "black") +
    geom_text(size = 5) +
    facet_grid(rows = vars(wflow_id)) +
    scale_fill_viridis_d() +
    xlab("Classification metric") +
    ylab("Aggregated mean value (selected metric)") +
    ggtitle("Aggregated classification metrics for best model candidate per workflow") +
    labs(subtitle = "Showing only best performing model candidate per each workflow (classifier - recipe).\nEach classification metric is aggregated over all assessments folds (mean values).",
         fill = "Workflow (model):") +
    theme_minimal(base_size = 14)
  
  # show plots
  print(p1)
  print(p2)
}



#' Predict (multi models) - probabilities & class predictions
#' 
#' This functions generates predictions of multiple models stored inside a list,
#' using selected data (validate / test data).
#' 
#' @param mod_fit_list_ A list of trained / fitted models. 
#' @param df_ A data frame - used for predictions.
#' 
#' @return df.pred A data frame with predictions included.
#'
predict_multi_mod <- function(mod_fit_list_,
                              df_){
  
  # predict probabilites
  df.pred_prob <- map(mod_fit_list_, ~ predict(.x, 
                                              df_, 
                                              type = "prob") %>% 
                   bind_cols(df_) %>% 
                   select(cover_type, 
                          starts_with(".pred_"))) %>% 
    bind_rows(., 
              .id = "wflow_id")
  
  # predict classes
  df.pred_class <- map(mod_fit_list_, ~ predict(.x, 
                                               df_, 
                                               type = "class")) %>% 
    bind_rows(., 
              .id = "wflow_id") %>% 
    select(.pred_class)
  
  # merge predictions
  df.pred <- bind_cols(df.pred_prob, df.pred_class)
  
  return(df.pred)
}



#' Visualize validation results
#' 
#' This function creates a plot:
#' - F1 score per each final model 
#' - results for validate data predictions
#'
#' @param df.validate_pred A data frame with predictions included (probabilities only) on validate data.
#' 
#' @return Null
#'
visualize_validate_rez <- function(df.validate_pred_ = df.validate_pred){
  
  # categories (for figures)
  levels.wflow   <- c("base_rf", "base_xgb", "base_lgbm", "norm_log", "down_rf", "down_xgb", "down_lgbm", "down_norm_log")
  
  # calculate F1 score for each model
  df.validate_F1 <- df.validate_pred_ %>% 
    group_by(wflow_id) %>% 
    f_meas(truth = cover_type, 
           .pred_class) %>% 
    ungroup() 
  
  # visualize results
  df.validate_F1 %>%  
    mutate(wflow_id = factor(wflow_id, levels = levels.wflow)) %>% 
    ggplot(aes(x = wflow_id,
               y = .estimate,
               label = round(.estimate, 4),
               fill = wflow_id)) +
    geom_col(color = "black", 
             show.legend = F) +
    geom_text(size = 7) +
    scale_y_continuous(breaks = seq(0,1,0.05)) +
    scale_fill_viridis_d() +
    xlab("Workflow (model)") +
    ylab("F1 score") +
    ggtitle("F1 score on validate data set for best model candidate per workflow") +
    labs(subtitle = "Showing only best performing model candidate per each workflow (classifier - recipe)\nValidation dataset was used.") +
    theme_minimal(base_size = 16)
}



#' Predict (single model) - probabilities & class predictions
#' 
#' This functions generates predictions of a single model,
#' using selected data (validate / test data).
#' 
#' @param mod_fit_ A model fit object. 
#' @param df_ A data frame - used for predictions.
#' 
#' @return df.pred A data frame with predictions included.
#'
predict_single_mod <- function(mod_fit_,
                               df_){
  
  # generate probabilities & class predictions (and merge them)
  df.pred <- predict(mod_fit_, 
                          df_, 
                          type = "class") %>% 
    bind_cols(predict(mod_fit_, 
                      df_, 
                      type = "prob")) %>% 
    bind_cols(df_) %>% 
    select(cover_type, starts_with(".pred_"))
  
  return(df.pred)
}



#' Visualize test results
#' 
#' This function first calculates selected classification metrics 
#' based on model's predictions on test data. Then it creates three different plots:
#' - figure 1: selected classification metrics on test data (final model) 
#' - figure 2: confusion matrix (absolute values) - test data (final model)
#' - figure 3: confusion matrix (relative values) - test data (final model)
#'
#' @param df.test_pred_ A data frame with predictions included (probabilities only) on test data.
#' 
#' @return Null
#'
visualize_test_rez <- function(df.test_pred_ = df.test_pred_){
  
  # figure 1:
  # - first calculate selected classification metrics
  # - then visualize results
  
  # calculate metrics that use hard class predictions
  
  # selected metrics (we collect during tuning)
  multi_class_metrics <- metric_set(
    f_meas,        # F1 (macro by default for multiclass)
    bal_accuracy,  # balanced accuracy
    precision,     # macro by default
    recall,        # macro by default
    accuracy       # overall accuracy
  )
  
  # calculate metrics
  test_metrics_class <- df.test_pred %>%
    multi_class_metrics(truth = cover_type, estimate = .pred_class)
  
  
  # calculate ROC AUC (Hand–Till, uses probabilities)
  test_metrics_auc <- df.test_pred %>%
    roc_auc(truth = cover_type,
            .pred_1:.pred_7)
  
  # metrics levels
  levels.metrics <- c("f_meas", "precision", "recall", "roc_auc", "bal_accuracy", "accuracy")
  
  # create plot
  p1 <- rbind(test_metrics_class,
        test_metrics_auc) %>% 
    mutate(.metric = factor(.metric, levels = levels.metrics)) %>% 
    ggplot(aes(x = .metric,
               y = .estimate,
               label = round(.estimate, 2))) +
    geom_col(color = "black", 
             fill = "gray50") +
    geom_text(size = 8) +
    xlab("Classification metric") +
    ylab("Value") +
    ggtitle("Classification metrics - final model's predictions") +
    labs(subtitle = "Test data") +
    theme_minimal(base_size = 14)
  
  
  
  # figure " & 3:
  # - first calculate values for confusion matrix
  # - then visualize it
  
  # confusion matrix calculation
  cm_tidy <- df.test_pred %>%
    conf_mat(truth = cover_type, estimate = .pred_class) %>%
    tidy() %>%
    separate(name, into = c("prefix", "prediction", "truth"), sep = "_") %>%
    mutate(truth = factor(truth),
           prediction = factor(prediction)) %>%
    select(prediction, truth, value) %>% 
    mutate(correct = factor(if_else(truth == prediction, 1, 0)),
           value_per = round(value / sum(value) * 100, 2))
  
  # visualize absolute values
  p2 <- cm_tidy %>% 
    ggplot(aes(x = truth,
               y = fct_rev(prediction),
               fill = correct)) +
    geom_tile(color = "black") +
    geom_text(aes(label = value)) +
    scale_fill_manual(values = c("brown1", "#98FB98")) + 
    xlab("Actual - Cover type class") +
    ylab("Predicted - Cover type class") +
    ggtitle("Confusion Matrix - Absolute values") +
    labs(subtitle = "Final model's predictions - test data",
         fill = "Correct prediction:") +
    theme_minimal(base_size = 16)
  
  # visualize relative values
  p3 <- cm_tidy %>% 
    ggplot(aes(x = truth,
               y = fct_rev(prediction),
               fill = correct)) +
    geom_tile(color = "black") +
    geom_text(aes(label = value_per)) +
    scale_fill_manual(values = c("brown1", "#98FB98")) + 
    xlab("Actual - Cover type class") +
    ylab("Predicted - Cover type class") +
    ggtitle("Confusion Matrix - Relative values (%)") +
    labs(subtitle = "Final model's predictions - test data",
         fill = "Correct prediction:") +
    theme_minimal(base_size = 16)
  
  # show plots
  print(p1)
  print(p2)
  print(p3)
}

