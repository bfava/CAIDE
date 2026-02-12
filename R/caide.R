# =============================================================================
# CAIDE: Covariate-Adjusted Inference on the Distribution of Treatment Effects
# =============================================================================
#
# This file implements the CAIDE estimator for bounding the fraction of units
# with treatment effects below (or above) a threshold delta in randomized
# experiments. The method leverages pre-treatment covariates and machine
# learning to sharpen classical Makarov bounds.
#
# Main references:
#   Fava, B. (2024). "Predicting the Distribution of Treatment Effects via
#   Covariate-Adjustment, with an Application to Microcredit."
#   arXiv:2407.14635. https://arxiv.org/abs/2407.14635
#
# Two main estimators are provided:
#   - caide_cf()  : Cross-fitting estimator (recommended)
#   - caide_ss()  : Sample-splitting estimator
#
# Required packages:
#   tidyverse, rsample, doMC, foreach, ranger
#   Optional (for additional ML models): mlr3, mlr3learners, mlr3tuning,
#   mlr3tuningspaces, mlr3extralearners, qrnn
#
# =============================================================================


# ---- Conditional CDF Estimation ---------------------------------------------

#' Estimate conditional CDFs F(Y | X) using machine learning
#'
#' Fits a conditional distribution model on a training set and returns
#' estimated CDF functions evaluated at each observation in df_new.
#'
#' @param df     Training data frame. Must contain columns 'Y' (outcome),
#'               'D' (treatment), 'pX' (propensity score), plus covariates.
#' @param df_new Evaluation data frame with the same columns as df.
#' @param model  Character string specifying the ML model to use:
#'               - 'quant_rf': Quantile random forest (recommended; uses ranger)
#'               - 'quant_nn': Quantile neural network (uses qrnn)
#'               - 'none': No covariate adjustment (returns NA placeholders)
#'               - Any mlr3 regression learner ID (e.g., 'regr.ranger',
#'                 'regr.xgboost', 'regr.svm', 'regr.nnet', 'regr.glmnet').
#'                 These use a location-shift model: F(Y|X) = F_eps(Y - mu(X)).
#' @param quants_seq Numeric vector of quantile levels in [0,1] for quantile
#'                   regression models (e.g., seq(0, 1, length.out = 101)).
#' @param .tune  Logical. If TRUE, tunes hyperparameters of mlr3 learners via
#'               random search with cross-validation. Default is FALSE.
#'
#' @return A list of functions, one per row of df_new. Each function maps a
#'         scalar y to the estimated CDF value P(Y <= y | X = x_i).
calc_cdf <- function(df, df_new, model, quants_seq, .tune = FALSE) {
  if (model == 'quant_rf') {
    # --- Quantile Random Forest via ranger ---
    library(ranger)
    rf <- ranger(Y ~ ., data = df, quantreg = TRUE,
                 num.trees = 10^3, keep.inbag = TRUE)
    quants <- predict(rf, data = df_new, type = "quantiles",
                      quantiles = quants_seq)$predictions
    cdfs <- apply(quants, 1, function(.x)
      approxfun(.x, quants_seq, yleft = 0, yright = 1, ties = "ordered"))

  } else if (model == 'none') {
    # --- No covariate adjustment ---
    cdfs <- rep(list(NA), nrow(df_new))

  } else if (model == 'quant_nn') {
    # --- Quantile Neural Network via qrnn ---
    n.hidden <- 3
    iter.max <- 100
    n.trials <- 1
    quants <- matrix(NA_real_, nrow = nrow(df_new), ncol = length(quants_seq))
    for (i in seq_along(quants_seq)) {
      quant <- quants_seq[i]
      qnn <- qrnn::qrnn.fit(
        x = as.matrix(select(df, -Y, -D, -pX)),
        y = as.matrix(df$Y),
        tau = quant, n.hidden = n.hidden,
        iter.max = iter.max, n.trials = n.trials, trace = FALSE
      )
      quants[, i] <- qrnn::qrnn.predict(
        x = as.matrix(select(df_new, -Y, -D, -pX)),
        parms = qnn
      )
    }
    cdfs <- apply(quants, 1, function(.x)
      approxfun(.x, quants_seq, yleft = 0, yright = 1))

  } else {
    # --- mlr3 regression learner with location-shift residual CDF ---
    library(mlr3)
    library(mlr3learners)
    library(mlr3tuning)
    library(mlr3tuningspaces)
    library(mlr3extralearners)
    task <- as_task_regr(df, target = 'Y')
    learner <- lrn(model)

    if (.tune) {
      tuner <- tnr("random_search")
      terminator <- trm("combo",
                         list(trm("run_time", secs = 60),
                              trm("stagnation", iters = 15, threshold = 0.01)),
                         any = TRUE)
      resampling_method <- rsmp("cv", folds = 3)
      measure <- msr("regr.rsq")
      tuning_space <- lts(paste0(model, '.default'))
      model <- auto_tuner(tuner = tuner,
                          learner = learner,
                          resampling = resampling_method,
                          measure = measure,
                          search_space = tuning_space,
                          terminator = terminator)
    } else {
      model <- learner
    }

    model$train(task)
    mu_x <- model$predict_newdata(df_new)$response

    # Estimate residual CDF and apply location shift
    cdf_eps <- ecdf(model$predict(task)$truth - model$predict(task)$response)
    cdfs <- map(mu_x, function(.mu_x) function(x) cdf_eps(x - .mu_x))
  }
  return(cdfs)
}


# ---- Conditional Expectation Estimation --------------------------------------

#' Estimate conditional expectation E[Y | X] using machine learning
#'
#' Fits a regression model on training data and returns predicted conditional
#' expectations on evaluation data. Used in the 'exp' (expectation) mode of
#' CAIDE, which approximates the optimal conditioning variable via E[Y|X].
#'
#' @param df     Training data frame with columns 'Y', 'D', 'pX', and covariates.
#' @param df_new Evaluation data frame with the same columns.
#' @param model  Character string: any mlr3 regression learner ID or 'none'.
#' @param .tune  Logical. If TRUE, tunes hyperparameters. Default is FALSE.
#'
#' @return Numeric vector of length nrow(df_new) with predicted values.
calc_expectation <- function(df, df_new, model, .tune = FALSE) {
  library(mlr3)
  library(mlr3learners)
  library(mlr3tuning)
  library(mlr3tuningspaces)
  library(mlr3extralearners)

  if (model == 'none') {
    return(rep(0, nrow(df_new)))
  }

  task <- as_task_regr(df, target = 'Y')
  learner <- lrn(model)

  if (.tune) {
    tuner <- tnr("random_search")
    terminator <- trm("combo",
                       list(trm("run_time", secs = 60),
                            trm("stagnation", iters = 15, threshold = 0.01)),
                       any = TRUE)
    resampling_method <- rsmp("cv", folds = 3)
    measure <- msr("regr.rsq")
    tuning_space <- lts(paste0(model, '.default'))
    model <- auto_tuner(tuner = tuner,
                        learner = learner,
                        resampling = resampling_method,
                        measure = measure,
                        search_space = tuning_space,
                        terminator = terminator)
  } else {
    model <- learner
  }

  model$train(task)
  # Average predictions at D=0 and D=1 to get E[Y|X] marginalized over D
  df_new0 <- mutate(df_new, D = 0)
  df_new1 <- mutate(df_new, D = 1)
  mu_x <- .5 * (model$predict_newdata(df_new1)$response +
                   model$predict_newdata(df_new0)$response)
  return(mu_x)
}


# ---- Estimand and Variance Functions -----------------------------------------

#' Compute the empirical process theta(t) at threshold t
#'
#' Evaluates the Makarov-type statistic:
#'   theta(t) = mean[ D * 1(Y <= t) / pX - (1-D) * 1(Y <= t - delta) / (1-pX) ]
#'
#' @param t      Scalar threshold.
#' @param Y      Numeric vector of (possibly adjusted) outcomes.
#' @param D      Binary treatment vector (0 or 1).
#' @param pX     Numeric vector of propensity scores.
#' @param .delta Scalar treatment effect threshold.
#'
#' @return Scalar value of theta(t).
theta_t <- function(t, Y, D, pX, .delta) {
  mean(D * (Y <= t) / pX - (1 - D) * (Y <= t - .delta) / (1 - pX))
}


#' Variance estimator when propensity scores are known
#'
#' Computes the sample variance of the influence function divided by n,
#' appropriate for settings with known (or accurately estimated) propensity
#' scores that do not depend on covariates X within strata.
#'
#' @inheritParams theta_t
#' @return Scalar estimated variance.
sigma2_hat_pX_known <- function(t, Y, D, pX, .delta) {
  var(D * (Y <= t) / pX - (1 - D) * (Y <= t - .delta) / (1 - pX)) / length(D)
}


#' Variance estimator when propensity scores are estimated (stratified)
#'
#' Groups observations by their propensity score value and computes a
#' stratified variance estimator. Appropriate for settings where pX varies
#' across strata defined by covariates.
#'
#' @inheritParams theta_t
#' @return Scalar estimated variance.
sigma2_hat_pX_hat <- function(t, Y, D, pX, .delta) {
  g <- as.numeric(as.factor(pX))
  n_g <- table(g)
  n1_g <- unname(table(g[D == 1]))
  n0_g <- unname(table(g[D == 0]))

  var_z1 <- numeric(length(unique(g)))
  var_z0 <- numeric(length(unique(g)))
  for (.g in unique(g)) {
    var_z1[.g] <- var(Y[D == 1 & g == .g] <= t)
    var_z0[.g] <- var(Y[D == 0 & g == .g] <= t)
  }
  return(sum(((n_g / length(D))^2) * ((1 / n1_g) * var_z1 + (1 / n0_g) * var_z0)))
}


# ---- Makarov Bounds Computation ----------------------------------------------

#' Compute Makarov bounds with cross-fitting inference
#'
#' Given adjusted outcomes Y_L (for lower bound) and Y_U (for upper bound),
#' computes Makarov bounds on the fraction of units with treatment effects
#' below delta. Also computes standard errors and the SJLS (simple) estimator.
#'
#' @param Y_L   Numeric vector of adjusted outcomes for the lower bound.
#' @param Y_U   Numeric vector of adjusted outcomes for the upper bound.
#' @param D     Binary treatment vector.
#' @param pX    Numeric vector of propensity scores.
#' @param .delta Scalar treatment effect threshold.
#'
#' @return A list with components:
#'   \item{makarov_lower}{Lower bound on P(Y(1) - Y(0) <= delta)}
#'   \item{makarov_upper}{Upper bound on P(Y(1) - Y(0) <= delta)}
#'   \item{sigma2_L}{Variance estimate for the lower bound (stratified)}
#'   \item{sigma2_U}{Variance estimate for the upper bound (stratified)}
#'   \item{sigma2_L_pXknown}{Variance estimate for lower bound (known pX)}
#'   \item{sigma2_U_pXknown}{Variance estimate for upper bound (known pX)}
#'   \item{vira}{List with SJLS estimator bounds and variances (lower, upper, sigma2L, sigma2U)}
makarov_cf <- function(Y_L, Y_U, D, pX, .delta) {
  # Build evaluation grid from adjusted outcomes
  pre_grid <- sort(c(Y_L + (1 - D) * .delta, Y_U + (1 - D) * .delta))
  grid_diffs <- abs(pre_grid - c(pre_grid[-1], 0))
  grid <- pre_grid - min(grid_diffs[grid_diffs != 0]) / 2

  # Evaluate theta(t) at all grid points for lower and upper bounds
  eval_grid_L <- map_dbl(grid, theta_t, Y = Y_L, D = D, pX = pX, .delta = .delta)
  eval_grid_U <- map_dbl(grid, theta_t, Y = Y_U, D = D, pX = pX, .delta = .delta)
  makarov_lower <- max(eval_grid_L, 0)
  makarov_upper <- 1 + min(eval_grid_U, 0)

  # Variance estimates at the maximizing/minimizing grid points
  sigma2_L <- sigma2_hat_pX_hat(grid[which.max(eval_grid_L)], Y_L, D, pX, .delta)
  sigma2_U <- sigma2_hat_pX_hat(grid[which.min(eval_grid_U)], Y_U, D, pX, .delta)
  sigma2_L_pXknown <- sigma2_hat_pX_known(grid[which.max(eval_grid_L)], Y_L, D, pX, .delta)
  sigma2_U_pXknown <- sigma2_hat_pX_known(grid[which.min(eval_grid_U)], Y_U, D, pX, .delta)

  # SJLS (simple) estimator: evaluate at t = 0
  vira_lower <- theta_t(0, Y_L, D, pX, .delta)
  vira_upper <- 1 + theta_t(0, Y_U, D, pX, .delta)
  vira_sigma2L <- sigma2_hat_pX_known(0, Y_L, D, pX, .delta)
  vira_sigma2U <- sigma2_hat_pX_known(0, Y_U, D, pX, .delta)

  list(makarov_lower = makarov_lower, makarov_upper = makarov_upper,
       sigma2_L = sigma2_L, sigma2_U = sigma2_U,
       sigma2_L_pXknown = sigma2_L_pXknown,
       sigma2_U_pXknown = sigma2_U_pXknown,
       vira = list(lower = vira_lower, upper = vira_upper,
                   sigma2L = vira_sigma2L, sigma2U = vira_sigma2U))
}


#' Compute Makarov bounds without covariates (sharp bounds)
#'
#' Estimates the sharp Makarov bounds using the empirical CDFs of treated
#' and control outcomes evaluated on a random grid. No covariate adjustment.
#'
#' @param Y_L   Numeric vector of outcomes for lower bound (or raw Y).
#' @param Y_U   Numeric vector of outcomes for upper bound (or raw Y).
#' @param D     Binary treatment vector.
#' @param .delta Scalar treatment effect threshold.
#'
#' @return A list with makarov_lower and makarov_upper.
sharp_makarov <- function(Y_L, Y_U, D, .delta) {
  F1_L <- ecdf(Y_L[D == 1])
  F0_L <- ecdf(Y_L[D == 0])
  F1_U <- ecdf(Y_U[D == 1])
  F0_U <- ecdf(Y_U[D == 0])
  diff_L <- function(y) F1_L(y) - F0_L(y - .delta)
  diff_U <- function(y) F1_U(y) - F0_U(y - .delta)
  random_grid <- rnorm(10^4, sd = 18)
  grid_diff_L <- diff_L(random_grid)
  grid_diff_U <- diff_U(random_grid)
  makarov_lower <- max(grid_diff_L, 0)
  makarov_upper <- min(1 + grid_diff_U, 1)
  list(makarov_lower = makarov_lower, makarov_upper = makarov_upper)
}


#' Compute Makarov bounds with sample-splitting inference
#'
#' Computes Makarov bounds and a DKW-type critical value for finite-sample
#' valid confidence intervals via sample splitting.
#'
#' @param Y_L   Numeric vector of adjusted outcomes for the lower bound.
#' @param Y_U   Numeric vector of adjusted outcomes for the upper bound.
#' @param D     Binary treatment vector.
#' @param .delta Scalar treatment effect threshold.
#' @param alpha  Significance level for the confidence interval.
#'
#' @return A list with makarov_lower, makarov_upper, and c_alpha (critical value).
makarov_ss <- function(Y_L, Y_U, D, .delta, alpha) {
  grid <- sort(c(Y_L + (1 - D) * .delta, Y_U + (1 - D) * .delta)) - 10^-2
  F1_L <- ecdf(Y_L[D == 1])
  F0_L <- ecdf(Y_L[D == 0])
  F1_U <- ecdf(Y_U[D == 1])
  F0_U <- ecdf(Y_U[D == 0])
  eval_grid_L <- map_dbl(grid, function(y) F1_L(y) - F0_L(y - .delta))
  eval_grid_U <- map_dbl(grid, function(y) F1_U(y) - F0_U(y - .delta))
  makarov_lower <- max(eval_grid_L, 0)
  makarov_upper <- 1 + min(eval_grid_U, 0)
  c_alpha <- sqrt(log(2 / alpha) / 2) *
    (sqrt(sum(D == 1))^(-1) + sqrt(sum(D == 0))^(-1))

  list(makarov_lower = makarov_lower, makarov_upper = makarov_upper,
       c_alpha = c_alpha)
}


# ---- Optimal Conditioning Variable ------------------------------------------

#' Find the optimal conditioning point s* (both lower and upper)
#'
#' Given estimated conditional CDFs F1(.|X) and F0(.|X), finds the values
#' s*_L and s*_U that maximize/minimize the CDF difference
#' F1(y|X) - F0(y - delta|X) over a random grid. These define the adjusted
#' outcomes Y_L = Y - s*_L and Y_U = Y - s*_U.
#'
#' @param .F1_X  Estimated CDF function for the treated group conditional on X.
#' @param .F0_X  Estimated CDF function for the control group conditional on X.
#' @param .delta Scalar treatment effect threshold.
#' @param .model Character string for the model (returns zero if 'none').
#' @param .scale Numeric scale for the random grid (default: 100).
#'
#' @return A list with s_L (maximizer) and s_U (minimizer).
.calc_s_star <- function(.F1_X, .F0_X, .delta, .model, .scale = 100) {
  if (.model == 'none') return(list(s_L = 0, s_U = 0))

  .diff <- function(y) .F1_X(y) - .F0_X(y - .delta)
  random_grid <- rnorm(10^4, sd = .scale)
  grid_diff <- .diff(random_grid)
  s_L <- random_grid[which.max(grid_diff)]
  s_U <- random_grid[which.min(grid_diff)]
  list(s_L = s_L, s_U = s_U)
}


#' Find the optimal conditioning point s* for one side only
#'
#' Same as .calc_s_star but returns only the maximizer (type='L') or
#' minimizer (type='U'). Used when different models are selected for the
#' lower and upper bounds.
#'
#' @inheritParams .calc_s_star
#' @param type Character: 'L' for lower bound optimizer, 'U' for upper.
#'
#' @return Scalar optimal conditioning value.
.calc_s_star_oneside <- function(.F1_X, .F0_X, .delta, .model, type = 'L', .scale = 100) {
  if (.model == 'none') return(0)
  .diff <- function(y) .F1_X(y) - .F0_X(y - .delta)
  random_grid <- rnorm(10^4, sd = .scale)
  grid_diff <- .diff(random_grid)
  if (type == 'L') {
    return(random_grid[which.max(grid_diff)])
  } else if (type == 'U') {
    return(random_grid[which.min(grid_diff)])
  }
}


# ---- Main Estimators ---------------------------------------------------------

#' CAIDE estimator via cross-fitting (recommended)
#'
#' Implements the CAIDE estimator using K-fold cross-fitting. When multiple
#' models are provided, an inner cross-validation loop selects the best model
#' for the lower and upper bounds separately based on t-statistics.
#'
#' @param Y      Numeric vector of outcomes (length n).
#' @param X      Data frame or matrix of pre-treatment covariates (n x p).
#' @param D      Binary treatment vector (0 or 1, length n).
#' @param pX     Numeric vector of propensity scores (length n). For a
#'               completely randomized experiment, set pX = rep(mean(D), n).
#' @param delta  Scalar threshold for the treatment effect. Default is 0
#'               (tests whether the treatment harms any units).
#' @param K      Integer number of cross-fitting folds. Default is 5.
#' @param K_sub  Integer number of inner CV folds for model selection.
#'               Only used when length(models) > 1. Default is 3.
#' @param alpha  Significance level for inference. Default is 0.05.
#' @param models Character vector of model names to consider. Options include:
#'               - 'quant_rf': Quantile random forest (recommended)
#'               - 'quant_nn': Quantile neural network
#'               - 'none': No covariate adjustment
#'               - Any mlr3 regression learner (e.g., 'regr.ranger',
#'                 'regr.xgboost', 'regr.svm', 'regr.nnet', 'regr.glmnet').
#'               If a single model is provided, model selection is skipped.
#' @param quants_seq Numeric vector of quantile levels for CDF estimation.
#'                   Recommended: seq(0, 1, length.out = 101).
#' @param ncores Integer number of parallel cores. Default is 1.
#' @param mode   Character: 'cdf' for CDF-based conditioning (sharper but
#'               slower) or 'exp' for expectation-based conditioning (faster,
#'               restricted to mlr3 models). Default is 'exp'.
#' @param tune   Logical. If TRUE, tunes ML hyperparameters. Default is FALSE.
#'
#' @return A list with components:
#'   \item{makarov_lower}{Estimated lower bound on P(Y(1) - Y(0) <= delta)}
#'   \item{makarov_upper}{Estimated upper bound on P(Y(1) - Y(0) <= delta)}
#'   \item{sigma2_L}{Variance estimate for the lower bound}
#'   \item{sigma2_U}{Variance estimate for the upper bound}
#'   \item{sigma2_L_pXknown}{Variance for lower bound (known propensity scores)}
#'   \item{sigma2_U_pXknown}{Variance for upper bound (known propensity scores)}
#'   \item{vira}{SJLS estimator (list with lower, upper, sigma2L, sigma2U)}
#'   \item{data}{Tibble of adjusted outcomes, treatment, and propensity scores}
#'   \item{model_selection}{Tibble of selected models per fold}
#'
#' @examples
#' \dontrun{
#' # Simulate data
#' n <- 500
#' X <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
#' D <- rbinom(n, 1, 0.5)
#' Y0 <- X$x1 + rnorm(n)
#' Y1 <- Y0 - 1 + X$x2
#' Y <- D * Y1 + (1 - D) * Y0
#' pX <- rep(0.5, n)
#'
#' # Run CAIDE with cross-fitting
#' result <- caide_cf(Y, X, D, pX,
#'                    delta = 0, K = 5,
#'                    models = 'quant_rf',
#'                    quants_seq = seq(0, 1, length.out = 101),
#'                    mode = 'cdf')
#'
#' cat("Lower bound:", result$makarov_lower, "\n")
#' cat("Upper bound:", result$makarov_upper, "\n")
#' cat("95% CI for lower bound: [",
#'     max(result$makarov_lower - qnorm(0.95) * sqrt(result$sigma2_L), 0),
#'     ",", result$makarov_lower + qnorm(0.95) * sqrt(result$sigma2_L), "]\n")
#' }
caide_cf <- function(Y, X, D, pX, delta = 0, K = 5, K_sub = 3, alpha = .05,
                     models = c('quant_rf', 'regr.ranger'), quants_seq,
                     ncores = 1L, mode = 'exp', tune = FALSE) {
  library(doMC)
  library(rsample)
  library(tidyverse)
  registerDoMC(ncores)

  n <- nrow(Y)
  Y_scale <- (max(Y) - min(Y)) / 20
  df0 <- cbind(Y, D, X, pX)
  splits <- vfold_cv(df0, v = K, strata = 'D')

  # ---- Step 1: Model selection (if multiple models) ----
  if (length(models) > 1) {
    subsplits <- map(splits$splits,
                     . %>% analysis() %>% vfold_cv(v = K_sub, strata = 'D'))
    specifications <- expand.grid(k = 1:K, k_sub = 1:K_sub, model = models,
                                  stringsAsFactors = FALSE)

    loop_res1 <- foreach(s = 1:nrow(specifications),
                         .combine = 'bind_rows') %dopar% {
      k <- specifications$k[s]
      k_sub <- specifications$k_sub[s]
      model <- specifications$model[s]

      split <- subsplits[[k]]$splits[[k_sub]]
      outer_ids <- splits$splits[[k]]$in_id
      out_id <- outer_ids[complement(split)]

      df <- analysis(split)
      df_new <- assessment(split)

      if (mode == 'cdf') {
        cdfs0 <- calc_cdf(df[df$D == 0, ], df_new, model, quants_seq, .tune = tune)
        cdfs1 <- calc_cdf(df[df$D == 1, ], df_new, model, quants_seq, .tune = tune)
        s_star <- map2(cdfs1, cdfs0, .calc_s_star,
                       .delta = delta, .model = model, .scale = Y_scale)
        s_L <- map_dbl(s_star, ~ .x$s_L)
        s_U <- map_dbl(s_star, ~ .x$s_U)
      } else if (mode == 'exp') {
        s_L <- calc_expectation(df, df_new, model, tune)
        s_U <- s_L
      }

      Y_local <- Y[out_id]
      tibble(k = k, k_sub = k_sub, model = model, out_id = out_id,
             s_L = s_L, s_U = s_U, Y = Y_local,
             D = D[out_id], pX = pX[out_id])
    }

    # Select best model per fold based on t-statistics
    best_specifications <- loop_res1 %>%
      arrange(k) %>%
      mutate(Y_L = Y - s_L, Y_U = Y - s_U) %>%
      group_by(k, model) %>%
      nest() %>%
      ungroup() %>%
      mutate(
        data = map(data, function(.x) .x %>%
                     group_by(pX) %>% mutate(pX2 = mean(D)) %>% ungroup()),
        bounds = map(data, function(.x)
          makarov_cf(.x$Y_L, .x$Y_U, .x$D, .x$pX2, delta)),
        t_stat_L = map_dbl(bounds, function(.x)
          ifelse(.x$sigma2_L > 0, .x$makarov_lower / sqrt(.x$sigma2_L), 0)),
        t_stat_U = map_dbl(bounds, function(.x)
          ifelse(.x$sigma2_U > 0, (1 - .x$makarov_upper) / sqrt(.x$sigma2_U), 0))
      ) %>%
      group_by(k) %>%
      summarise(model_L = model[which.max(t_stat_L)],
                model_U = model[which.max(t_stat_U)])
  } else {
    best_specifications <- tibble(k = 1:K,
                                  model_L = rep(models, K),
                                  model_U = rep(models, K))
  }

  # ---- Step 2: Fit ML models and compute adjusted outcomes ----
  loop_res2 <- foreach(s = 1:nrow(best_specifications),
                       .combine = 'bind_rows') %dopar% {
    k <- best_specifications$k[s]
    model_L <- best_specifications$model_L[s]
    model_U <- best_specifications$model_U[s]

    split <- splits$splits[[k]]
    out_id <- complement(split)

    df <- analysis(split)
    df_new <- assessment(split)

    if (mode == 'cdf') {
      if (model_L != model_U) {
        cdfs0_L <- calc_cdf(df[df$D == 0, ], df_new, model_L, quants_seq, .tune = tune)
        cdfs1_L <- calc_cdf(df[df$D == 1, ], df_new, model_L, quants_seq, .tune = tune)
        s_L <- map2_dbl(cdfs1_L, cdfs0_L, .calc_s_star_oneside,
                        .delta = delta, .model = model_L, type = 'L', .scale = Y_scale)

        cdfs0_U <- calc_cdf(df[df$D == 0, ], df_new, model_U, quants_seq, .tune = tune)
        cdfs1_U <- calc_cdf(df[df$D == 1, ], df_new, model_U, quants_seq, .tune = tune)
        s_U <- map2_dbl(cdfs1_U, cdfs0_U, .calc_s_star_oneside,
                        .delta = delta, .model = model_U, type = 'U', .scale = Y_scale)
      } else {
        cdfs0 <- calc_cdf(df[df$D == 0, ], df_new, model_L, quants_seq, .tune = tune)
        cdfs1 <- calc_cdf(df[df$D == 1, ], df_new, model_L, quants_seq, .tune = tune)
        s_star <- map2(cdfs1, cdfs0, .calc_s_star,
                       .delta = delta, .model = model_L, .scale = Y_scale)
        s_L <- map_dbl(s_star, ~ .x$s_L)
        s_U <- map_dbl(s_star, ~ .x$s_U)
      }
    } else if (mode == 'exp') {
      s_L <- calc_expectation(df, df_new, model_L, tune)
      if (model_L != model_U) {
        s_U <- calc_expectation(df, df_new, model_U, tune)
      } else {
        s_U <- s_L
      }
    }

    Y_local <- Y[out_id]
    tibble(k = k, model_L = model_L, model_U = model_U, out_id = out_id,
           s_L = s_L, s_U = s_U, Y = Y_local,
           D = D[out_id], pX = pX[out_id])
  }

  loop_res2$Y_L <- loop_res2$Y - loop_res2$s_L
  loop_res2$Y_U <- loop_res2$Y - loop_res2$s_U

  # ---- Step 3: Compute final bounds ----
  bounds <- makarov_cf(loop_res2$Y_L, loop_res2$Y_U,
                       loop_res2$D, loop_res2$pX, delta)

  c(bounds,
    list(tibble(Y_L = loop_res2$Y_L, Y_U = loop_res2$Y_U,
                D = loop_res2$D, pX = loop_res2$pX)),
    list(best_specifications))
}


#' CAIDE estimator via sample splitting
#'
#' Implements the CAIDE estimator using a single train/test split. Provides
#' finite-sample valid confidence intervals using DKW-type critical values.
#'
#' @inheritParams caide_cf
#' @param K_sub Integer number of inner CV folds for model selection.
#'              Default is 2.
#' @param prop  Proportion of data used for training. Default is 0.5.
#'
#' @return A list with components:
#'   \item{makarov_lower}{Estimated lower bound on P(Y(1) - Y(0) <= delta)}
#'   \item{makarov_upper}{Estimated upper bound on P(Y(1) - Y(0) <= delta)}
#'   \item{c_alpha}{DKW critical value for confidence intervals}
#'   \item{data}{Tibble of adjusted outcomes and treatment assignment}
#'
#' @examples
#' \dontrun{
#' result <- caide_ss(Y, X, D, pX,
#'                    delta = 0, prop = 0.5,
#'                    models = 'quant_rf',
#'                    quants_seq = seq(0, 1, length.out = 101))
#'
#' # Finite-sample valid confidence interval
#' ci_lower <- result$makarov_lower - result$c_alpha
#' ci_upper <- result$makarov_upper + result$c_alpha
#' }
caide_ss <- function(Y, X, D, pX, delta = 0, K_sub = 2L, prop = .5,
                     alpha = .1, models = 'quant_rf', quants_seq,
                     ncores = 1L, tune = FALSE) {
  library(doMC)
  registerDoMC(ncores)

  n <- length(Y)
  Y_scale <- (max(Y) - min(Y)) / 2
  df0 <- cbind(Y, D, X, pX)
  split <- initial_split(df0, prop = prop)

  # ---- Step 1: Model selection (if multiple models) ----
  if (length(models) > 1) {
    subsplits <- vfold_cv(training(split), v = K_sub)
    specifications <- expand.grid(k_sub = 1:K_sub, model = models,
                                  stringsAsFactors = FALSE)

    loop_res1 <- foreach(s = 1:nrow(specifications),
                         .combine = 'bind_rows') %dopar% {
      k_sub <- specifications$k_sub[s]
      model <- specifications$model[s]

      subsplit <- subsplits$splits[[k_sub]]
      outer_ids <- split$in_id
      out_id <- outer_ids[complement(subsplit)]

      df <- analysis(subsplit)
      df_new <- assessment(subsplit)
      cdfs0 <- calc_cdf(df[df$D == 0, ], df_new, model, quants_seq, .tune = tune)
      cdfs1 <- calc_cdf(df[df$D == 1, ], df_new, model, quants_seq, .tune = tune)
      s_star <- map2(cdfs1, cdfs0, .calc_s_star,
                     .delta = delta, .model = model, .scale = Y_scale)
      s_L <- map_dbl(s_star, ~ .x$s_L)
      s_U <- map_dbl(s_star, ~ .x$s_U)

      Y_local <- Y[out_id]
      tibble(k_sub = k_sub, model = model, out_id = out_id,
             s_L = s_L, s_U = s_U, Y = Y_local,
             D = D[out_id], pX = pX[out_id])
    }

    best_specification <- loop_res1 %>%
      mutate(Y_L = Y - s_L, Y_U = Y - s_U) %>%
      group_by(model) %>%
      nest() %>%
      ungroup() %>%
      mutate(
        bounds = map(data, function(.x)
          makarov_cf(.x$Y_L, .x$Y_U, .x$D, .x$pX, delta)),
        t_stat = map_dbl(bounds, function(.x)
          .x$makarov_lower / sqrt(.x$sigma2_L)),
        t_stat = ifelse(is.nan(t_stat), 0, t_stat) +
          rnorm(length(t_stat), sd = 10^-4)
      ) %>%
      filter(t_stat == max(t_stat))
    best_model <- best_specification$model
  } else {
    best_model <- models
  }

  # ---- Step 2: Estimate conditioning variable on test set ----
  df_train <- training(split)
  df_test <- testing(split)
  cdfs0 <- calc_cdf(df_train[df_train$D == 0, ], df_test, best_model,
                     quants_seq, .tune = tune)
  cdfs1 <- calc_cdf(df_train[df_train$D == 1, ], df_test, best_model,
                     quants_seq, .tune = tune)
  s_star <- map2(cdfs1, cdfs0, .calc_s_star,
                 .delta = delta, .model = best_model, .scale = Y_scale)
  s_L <- map_dbl(s_star, ~ .x$s_L)
  s_U <- map_dbl(s_star, ~ .x$s_U)

  # ---- Step 3: Compute bounds on test set ----
  Y_L <- df_test$Y - s_L
  Y_U <- df_test$Y - s_U
  bounds <- makarov_ss(Y_L, Y_U, df_test$D, delta, alpha)

  c(bounds, list(tibble(Y_L = Y_L, Y_U = Y_U, D = df_test$D)))
}


# ---- Utility: Summarize Results ----------------------------------------------

#' Print a summary of CAIDE results
#'
#' @param result Output list from caide_cf() or caide_ss().
#' @param alpha  Significance level for confidence intervals. Default is 0.05.
#' @param method Character: 'cf' for cross-fitting or 'ss' for sample splitting.
#'
#' @return Invisible NULL. Prints results to console.
print_caide_results <- function(result, alpha = 0.05, method = 'cf') {
  cat("==================================================\n")
  cat("CAIDE Results\n")
  cat("==================================================\n\n")
  cat("Bounds on P(Y(1) - Y(0) <= delta):\n")
  cat(sprintf("  Lower bound: %.4f\n", result$makarov_lower))
  cat(sprintf("  Upper bound: %.4f\n", result$makarov_upper))

  if (method == 'cf') {
    se_L <- sqrt(result$sigma2_L)
    se_U <- sqrt(result$sigma2_U)
    z <- qnorm(1 - alpha)
    ci_L_low <- max(result$makarov_lower - z * se_L, 0)
    ci_L_high <- result$makarov_lower + z * se_L
    ci_U_low <- result$makarov_upper - z * se_U
    ci_U_high <- min(result$makarov_upper + z * se_U, 1)

    cat(sprintf("\nStandard errors:\n"))
    cat(sprintf("  SE(lower): %.4f\n", se_L))
    cat(sprintf("  SE(upper): %.4f\n", se_U))
    cat(sprintf("\n%.0f%% Confidence intervals:\n", (1 - alpha) * 100))
    cat(sprintf("  Lower bound CI: [%.4f, %.4f]\n", ci_L_low, ci_L_high))
    cat(sprintf("  Upper bound CI: [%.4f, %.4f]\n", ci_U_low, ci_U_high))

    pval_L <- 1 - pnorm(result$makarov_lower / se_L)
    cat(sprintf("\np-value (H0: lower bound = 0): %.4f\n", pval_L))
  } else if (method == 'ss') {
    ci_lower <- max(result$makarov_lower - result$c_alpha, 0)
    ci_upper <- min(result$makarov_upper + result$c_alpha, 1)
    cat(sprintf("\nDKW critical value: %.4f\n", result$c_alpha))
    cat(sprintf("Confidence interval: [%.4f, %.4f]\n", ci_lower, ci_upper))
  }

  cat("==================================================\n")
  invisible(NULL)
}
