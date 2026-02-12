# =============================================================================
# Example: Monte Carlo Simulation for CAIDE
# =============================================================================
#
# This script demonstrates how to run a Monte Carlo study to evaluate the
# performance of the CAIDE estimator under known data-generating processes.
# It simulates data, computes CAIDE bounds, and compares them to the true
# fraction affected.
#
# This is adapted from the simulation study in Section 5 of:
#   Fava, B. (2024). "Predicting the Distribution of Treatment Effects via
#   Covariate-Adjustment, with an Application to Microcredit."
#   arXiv:2407.14635. https://arxiv.org/abs/2407.14635
#
# Required packages: tidyverse, rsample, ranger, doMC, mvtnorm
# =============================================================================

library(tidyverse)
library(rsample)

# Load CAIDE functions
source("R/caide.R")

# =============================================================================
# Step 1: Define the Data-Generating Process (DGP)
# =============================================================================

set.seed(123)

# Dimensions
d <- 20   # Total covariates generated
p <- 8    # Covariates used in the outcome model
r <- 0.5  # Autocorrelation parameter for covariate covariance

# Outcome model coefficients
beta0 <- 3^(1:-(p - 2))                       # Coefficients for Y(0)
Theta0 <- matrix(c(0.2, -0.2), nrow = p, ncol = p)  # Quadratic terms for Y(0)
for (i in 1:p) for (j in i:p) Theta0[i, j] <- Theta0[j, i]

beta_tau <- rep(1, p)                          # Coefficients for treatment effect
Theta_tau <- matrix(0.2, nrow = p, ncol = p)   # Quadratic terms for treatment effect

# Covariance matrix for covariates (AR(1) structure with block independence)
Sigma <- matrix(NA_real_, nrow = d, ncol = d)
for (i in 1:d) for (j in 1:d) Sigma[i, j] <- r^abs(i - j)
Sigma[1:2, 3:d] <- 0
Sigma[3:d, 1:2] <- 0

# Zero out cross-block interactions in the outcome model
Theta0[1:2, 3:p] <- 0
Theta0[3:p, 1:2] <- 0
Theta_tau[1:2, 3:p] <- 0
Theta_tau[3:p, 1:2] <- 0

# Treatment effect parameters
sigma_eps <- 0   # No additional noise in potential outcomes
tau_fix <- -1    # Fixed component of treatment effect


# =============================================================================
# Step 2: Compute the True Fraction Harmed (for validation)
# =============================================================================

cat("Computing true fraction with Y(1) - Y(0) <= 0...\n")
N_oracle <- 10^5
X_oracle <- mvtnorm::rmvnorm(N_oracle, mean = rep(0, p),
                              sigma = Sigma[c(1:2, (d - p + 3):d),
                                            c(1:2, (d - p + 3):d)])
Y0_oracle <- numeric(N_oracle)
Y1_oracle <- numeric(N_oracle)
for (i in 1:N_oracle) {
  Y0_oracle[i] <- X_oracle[i, ] %*% beta0 +
    t(X_oracle[i, ]) %*% Theta0 %*% X_oracle[i, ]
  Y1_oracle[i] <- Y0_oracle[i] + tau_fix +
    X_oracle[i, ] %*% beta_tau +
    t(X_oracle[i, ]) %*% Theta_tau %*% X_oracle[i, ]
}
theta0 <- mean(Y1_oracle - Y0_oracle <= 0)
cat("True P(Y(1) - Y(0) <= 0) =", round(theta0, 4), "\n\n")


# =============================================================================
# Step 3: Monte Carlo Loop
# =============================================================================

n_sims <- 10     # Number of MC replications (increase for final results)
n_obs <- 500     # Sample size per replication
model <- 'quant_rf'  # ML model for covariate adjustment

# Storage for results
mc_results <- tibble(
  sim = integer(),
  lower_cov = numeric(),   # CAIDE lower bound (with covariates)
  upper_cov = numeric(),   # CAIDE upper bound (with covariates)
  se_lower = numeric(),    # Standard error of lower bound
  se_upper = numeric(),    # Standard error of upper bound
  lower_nocov = numeric(), # Lower bound without covariates
  upper_nocov = numeric()  # Upper bound without covariates
)

cat("Running Monte Carlo simulations...\n")
for (s in 1:n_sims) {
  cat(sprintf("  Simulation %d/%d\n", s, n_sims))

  # --- Simulate data ---
  X <- mvtnorm::rmvnorm(n_obs, mean = rep(0, d), sigma = Sigma)
  X0 <- X[, c(1:2, (d - p + 3):d)]

  Y0 <- numeric(n_obs)
  Y1 <- numeric(n_obs)
  for (i in 1:n_obs) {
    Y0[i] <- X0[i, ] %*% beta0 +
      t(X0[i, ]) %*% Theta0 %*% X0[i, ]
    Y1[i] <- Y0[i] + tau_fix +
      X0[i, ] %*% beta_tau +
      t(X0[i, ]) %*% Theta_tau %*% X0[i, ]
  }

  D <- sample(0:1, n_obs, replace = TRUE)
  Y <- D * Y1 + (1 - D) * Y0
  pX <- rep(mean(D), n_obs)

  # --- CAIDE with covariates ---
  result_cov <- caide_cf(
    Y = Y,
    X = as.data.frame(X[, 1:10]),  # Use first 10 covariates
    D = D,
    pX = pX,
    delta = 0,
    K = 5,
    models = model,
    quants_seq = seq(0, 1, length.out = 101),
    ncores = 1L,
    mode = 'cdf'
  )

  # --- CAIDE without covariates (benchmark) ---
  result_nocov <- caide_cf(
    Y = Y,
    X = as.data.frame(X[, 1:10]),
    D = D,
    pX = pX,
    delta = 0,
    K = 5,
    models = 'none',
    quants_seq = seq(0, 1, length.out = 101),
    ncores = 1L,
    mode = 'cdf'
  )

  mc_results <- mc_results %>%
    add_row(
      sim = s,
      lower_cov = result_cov$makarov_lower,
      upper_cov = result_cov$makarov_upper,
      se_lower = sqrt(result_cov$sigma2_L),
      se_upper = sqrt(result_cov$sigma2_U),
      lower_nocov = result_nocov$makarov_lower,
      upper_nocov = result_nocov$makarov_upper
    )
}


# =============================================================================
# Step 4: Summarize Results
# =============================================================================

cat("\n==================================================\n")
cat("Monte Carlo Results (", n_sims, " replications)\n")
cat("==================================================\n\n")

cat("True theta_0 =", round(theta0, 4), "\n\n")

cat("With Covariates (CAIDE):\n")
cat(sprintf("  Avg lower bound: %.4f\n", mean(mc_results$lower_cov)))
cat(sprintf("  Avg upper bound: %.4f\n", mean(mc_results$upper_cov)))
cat(sprintf("  Avg bound width: %.4f\n",
            mean(mc_results$upper_cov - mc_results$lower_cov)))

# Coverage of 95% confidence intervals
alpha <- 0.05
ci_lower_lb <- pmax(mc_results$lower_cov -
                      qnorm(1 - alpha) * mc_results$se_lower, 0)
ci_upper_ub <- pmin(mc_results$upper_cov +
                      qnorm(1 - alpha) * mc_results$se_upper, 1)
coverage <- mean(ci_lower_lb <= theta0 & ci_upper_ub >= theta0)
cat(sprintf("  Coverage (95%% CI): %.3f\n", coverage))

cat("\nWithout Covariates (Benchmark):\n")
cat(sprintf("  Avg lower bound: %.4f\n", mean(mc_results$lower_nocov)))
cat(sprintf("  Avg upper bound: %.4f\n", mean(mc_results$upper_nocov)))
cat(sprintf("  Avg bound width: %.4f\n",
            mean(mc_results$upper_nocov - mc_results$lower_nocov)))

cat(sprintf("\nImprovement in avg width: %.1f%%\n",
            100 * (1 - mean(mc_results$upper_cov - mc_results$lower_cov) /
                     mean(mc_results$upper_nocov - mc_results$lower_nocov))))
