#' Assignment - Multiclass Classification with Dry Bean Dataset - Functions



#' Import & clean data
#' 
#' This functions:
#' - first imports .csv file
#' - and applies cleaning (clean column names, and modify target column)
#' 
#' @param path_file A string holding path to .csv file.
#'
#' @return df_ A data frame - imported.
#'
import_data <- function(path_file){
  
  # target column levels
  tar_lev <- c("barbunya", "bombay", "cali", "dermason", "horoz", "seker", "sira")
  
  # data import
  df_ <- read_csv(f_path, col_names = T) %>% 
    janitor::clean_names()
  
  # data cleaning
  df_ <- df_ %>% 
    select(class, everything()) %>% 
    mutate(class = str_to_lower(class),
           class = as.factor(class))
  
  return(df_)
}



#' Sample data
#' 
#' This functions:
#' - decreased df size
#' - by randomly sampling smaller portion of rows
#' 
#' @param df_ A data frame to be sampled.
#' @param p A proportion of rows kept after data sampling.
#'
#' @return df_ A data frame - sampled.
#'
sample_data <- function(df_ = df,
                        p = 0.1){
  
  df_ <- df_ %>% 
    sample_frac(size = p, 
                replace = F)
  
  return(df_)
}



#' Split data
#'
#' This function randomly splits (stratified sampling considered) the data into:
#' - train set
#' - validate set
#' - test set
#' You need to provide a data frame and set's proportions.
#' 
#' @param df_ A data frame for selected data.
#' @param p_t A numerical value - proportion of data assigned to train set (between 0 and 1 - must be less than 1!) - (default value 7/10).
#' @param p_v A numerical value - proportion of test set data assigned to validate set (between 0 and 1 - must be greater than 0!) - (default value 2/3).
#' @param tar_var A string - name of target column (for stratified sampling).
#'
#' @return data_split.list A list of data frame split into sets.
#'
split_data <- function(df_,
                       p_t = 7/10,
                       p_v = 2/3,
                       tar_var){
  
  # train VS validate + test split
  split_init  <- initial_split(df_,           # initial split train VS validate + test   
                               prop = p_t, 
                               strata = all_of(tar_var)) 
  df.train    <- training(split_init)        # train data
  df.val_test <- testing(split_init)         # validate + test data
  
  # validate VS test split
  split_val_tes <- initial_split(df.val_test,  # split validate VS test
                                 prop = p_v,
                                 strata = all_of(tar_var)) 
  df.validate   <- training(split_val_tes)     # validate data
  df.test       <- testing(split_val_tes)      # test data
  
  # print data set dimension sizes
  print("Data split into:")
  print(paste0("train set: ",    nrow(df.train),    " rows | ", round(nrow(df.train) / nrow(df) * 100, 2),    " % of all rows"))
  print(paste0("validate set: ", nrow(df.validate), " rows | ", round(nrow(df.validate) / nrow(df) * 100, 2), " % of all rows"))
  print(paste0("test set: ",     nrow(df.test),     " rows | ", round(nrow(df.test) / nrow(df) * 100, 2),     " % of all rows"))
  
  # create list
  data_split.list <- list(train = df.train,
                          validate =  df.validate,
                          test = df.test)
  
  return(data_split.list)
}



#' Finalize model parameter list
#'
#' This function matches selected recipe with selected model and finalizes features (parameter) list.
#' For the hyperparameters tuning phase.
#' 
#' @param mod_list_ A list of model specification.
#' @param rec_list_ A list of recipes used for each model.
#' @param df.train_ A training data df.
#'
#' @return par_fin_list A list of finalized features per each model.
#'
finalize_model_params <- function(mod_list_ = mod_list,
                                  rec_list_ = rec_list,
                                  df.train_ = df.train){
  
  # finalize model's parameter list (matching model and recipe)
  par_fin_list <- map(names(mod_list_), function(id) {
    
    # first prep and bake the corresponding recipe
    rec <- prep(rec_list_[[id]], training = df.train_)
    df.train_baked <- bake(rec, new_data = NULL)
    
    # extract and finalize model parameters using baked data
    extract_parameter_set_dials(mod_list[[id]]) %>%
      finalize(df.train_baked)
  })
  
  # assign model's names to list of results
  names(par_fin_list) <- names(mod_list)
  
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
#' @param cv_folds_ Train data split into folds (Cross-validation object).
#' @param iter The maximum number of search iterations.
#' @param initial An initial set of results.
#' @param no_improve The integer cutoff for the number of iterations without better results.
#' 
#' @return tune_rez A list of tuning results per each model. 
#'
tune_bayes_custom <- function(wflow_set_ = wflow_set,
                              par_fin_list_ = par_fin_list,
                              cv_folds_ = cv_folds,
                              iter = 20,
                              initial = 5,
                              no_improve = 5){
  
  # classification metrics to be collected
  class_metrics <- metric_set(
    accuracy, bal_accuracy,     
    precision, recall, f_meas,           
    roc_auc, pr_auc,           
    mcc, j_index
  )
  
  # empty list - tuning results
  tune_rez <- list()
  
  # wflow ids 
  # - remove not penalized logistic regression - no tuning!
  ids <- wflow_set_$wflow_id %>% setdiff("log")
  
  # loop over each model id
  for (id in ids) {
    
    print("")
    print("|--------------------------|")
    print(paste0("Tuning: ", id))
    
    ts <- Sys.time() # track execution time
    
    # extract workflow
    wflow <- extract_workflow(wflow_set_, id)
    
    # extract finalized param set
    par_fin <- par_fin_list_[[id]]
    
    # run Bayesian tuning
    rez <- tune_bayes(
      object = wflow,
      resamples = cv_folds_,
      metrics = class_metrics,
      param_info = par_fin,
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
    
    # store results
    tune_rez[[id]] <- rez
  }
  
  return(tune_rez)
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
#' This functions selects best model candidate per each model (algorithm type). 
#' 
#' @param tune_rez_ A list of tuning results per each model. 
#' @param wflow_set_ A tibble of workflows objects per each model.
#' @param df.train_ A training data df.
#' 
#' @return mod_fit_list A list of fitted models (best candidates per algorithm) on training data.
#'
extract_best_models <- function(tune_rez_ = tune_rez,
                                wflow_set_ = wflow_set,
                                df.train_ = df.train){
  
  mod_fit_list  <- map(names(tune_rez_), function(id) {
    
    # extract best model candidate per given algorithm
    best_par <- select_best(tune_rez_[[id]], metric = "roc_auc")
  })
  
  # assign model's names to list of results
  names(mod_fit_list) <- names(tune_rez_)
  
  return(mod_fit_list)
}



#' Finalize each workflow and fit best model.
#' 
#' This functions first selects best model candidate per each model (algorithm type),
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
                            df.train_ = df.train){
  
  mod_fit_list  <- map(names(tune_rez_), function(id) {
    
    # extract best model candidate per given algorithm
    best_par <- select_best(tune_rez_[[id]], metric = "roc_auc")
    
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
#' - figure 1: ROC AUC per each top candidate 
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
  levels.wflow   <- c("log", "logp", "knn", "tree", "rf", "xgb", "lgbm")
  levels.metrics <- c("roc_auc", "pr_auc", "accuracy", "bal_accuracy", "precision", "recall", "f_meas", "j_index", "mcc")
  
  
  # figure 1:
  # - first collect ROC AUC only for top candidates
  # - metrics on fold level
  # - then we draw fold id on x-axis
  # - and ROC AUC on y-axis
  # - per each top model candidate
  
  ## prepare data
  df.plot <- tibble()
  
  for(model in names(tune_rez_met_fold_)){
    
    print(model)
    
    df.plot_ <- inner_join(x = tune_rez_met_fold_[[model]],
                           y = top_mod_candidates_[[model]] %>% select(.config),
                           by = ".config") %>% 
      filter(.metric == "roc_auc") %>% 
      mutate(wflow_id = factor(wflow_id, levels = levels.wflow),
             id = as.factor(id)) %>% 
      select(wflow_id, id, .metric, .estimator, .estimate, .config)
    
    df.plot <- rbind(df.plot, df.plot_)
  }
  
  ## create plot
  p1 <- df.plot %>%  
    ggplot(aes(x = id,
               y = .estimate,
               color = wflow_id,
               group = wflow_id)) +
    geom_line(linewidth = 1) +
    geom_point(size = 5) +
    scale_color_viridis_d() +
    xlab("Assessment fold ID") +
    ylab("ROC AUC") +
    ggtitle("ROC AUC over assessment folds for best model candidate per workflow") +
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
    theme_minimal(base_size = 16)
  
  # show plots
  print(p1)
  print(p2)
}
