# =============================================================================
# Example: Running CAIDE on Experimental Data
# =============================================================================
#
# This script demonstrates how to use the CAIDE estimator to bound the
# fraction of units harmed (or helped) by a treatment using data from a
# randomized experiment with pre-treatment covariates.
#
# The example uses a simulated dataset that mimics a typical RCT structure.
# Replace the data loading section with your own data.
#
# Required packages: tidyverse, rsample, ranger, doMC
# =============================================================================

library(tidyverse)
library(rsample)

# Load CAIDE functions
source("R/caide.R")

# =============================================================================
# Step 1: Prepare Your Data
# =============================================================================
#
# You need:
#   Y  - Numeric vector of outcomes (length n)
#   X  - Data frame of pre-treatment covariates (n x p)
#   D  - Binary treatment indicator (0 = control, 1 = treated)
#   pX - Propensity scores (probability of treatment given covariates)
#
# For a completely randomized experiment: pX = rep(mean(D), length(D))
# For a stratified experiment: pX should reflect strata-level probabilities

# --- Example: Simulated Data ---
set.seed(42)
n <- 1000
p <- 5

# Covariates
X <- data.frame(matrix(rnorm(n * p), nrow = n))
colnames(X) <- paste0("x", 1:p)

# Treatment assignment (completely randomized)
D <- sample(0:1, n, replace = TRUE)

# Potential outcomes with heterogeneous treatment effects
Y0 <- 2 * X$x1 + X$x2^2 + rnorm(n)
Y1 <- Y0 - 1 + 0.5 * X$x1 + 0.3 * X$x3  # some units harmed, some helped

# Observed outcome
Y <- D * Y1 + (1 - D) * Y0

# Propensity score (known in a completely randomized experiment)
pX <- rep(mean(D), n)

# True fraction harmed (for validation; not available in practice)
true_fraction_harmed <- mean(Y1 - Y0 <= 0)
cat("True fraction with Y(1) - Y(0) <= 0:", true_fraction_harmed, "\n\n")


# =============================================================================
# Step 2: Run CAIDE with Cross-Fitting (Recommended)
# =============================================================================

# Using a single model (faster, no model selection needed)
result_cf <- caide_cf(
  Y = Y,
  X = X,
  D = D,
  pX = pX,
  delta = 0,           # Test: fraction with treatment effect <= 0
  K = 5,               # Number of cross-fitting folds
  models = 'quant_rf', # Quantile random forest
  quants_seq = seq(0, 1, length.out = 101),
  ncores = 1L,         # Set higher for parallel computation
  mode = 'cdf'         # 'cdf' for CDF-based conditioning (sharper)
)

# Print results
print_caide_results(result_cf, alpha = 0.05, method = 'cf')


# =============================================================================
# Step 3 (Optional): Run CAIDE with Multiple Models + Model Selection
# =============================================================================

# CAIDE automatically selects the best model per fold when given multiple options
# result_multi <- caide_cf(
#   Y = Y,
#   X = X,
#   D = D,
#   pX = pX,
#   delta = 0,
#   K = 5,
#   K_sub = 3,         # Inner CV folds for model selection
#   models = c('quant_rf', 'regr.ranger', 'none'),
#   quants_seq = seq(0, 1, length.out = 101),
#   ncores = 4L,
#   mode = 'cdf'
# )
# print_caide_results(result_multi, alpha = 0.05, method = 'cf')


# =============================================================================
# Step 4 (Optional): Run CAIDE with Sample Splitting
# =============================================================================
#
# The sample-splitting estimator provides finite-sample valid confidence
# intervals using DKW-type critical values. It uses less data than
# cross-fitting (since only the test half is used for inference) but has
# stronger theoretical guarantees in finite samples.

# result_ss <- caide_ss(
#   Y = Y,
#   X = X,
#   D = D,
#   pX = pX,
#   delta = 0,
#   prop = 0.5,          # Fraction used for training
#   alpha = 0.1,         # Significance level
#   models = 'quant_rf',
#   quants_seq = seq(0, 1, length.out = 101),
#   ncores = 1L
# )
# print_caide_results(result_ss, alpha = 0.1, method = 'ss')


# =============================================================================
# Step 5: Interpreting Results
# =============================================================================
#
# result$makarov_lower: Lower bound on P(Y(1) - Y(0) <= delta)
#   - If delta = 0, this bounds the fraction of units harmed by treatment
#   - A significantly positive lower bound means some units are genuinely harmed
#
# result$makarov_upper: Upper bound on P(Y(1) - Y(0) <= delta)
#
# For cross-fitting (caide_cf):
#   - result$sigma2_L, result$sigma2_U: Variance estimates
#   - Use qnorm(1 - alpha) * sqrt(sigma2) for confidence interval half-widths
#   - p-value for H0: "lower bound = 0" is 1 - pnorm(lower / sqrt(sigma2_L))
#
# For sample splitting (caide_ss):
#   - result$c_alpha: DKW critical value
#   - CI for lower bound: [lower - c_alpha, lower + c_alpha]
#   - CI for upper bound: [upper - c_alpha, upper + c_alpha]
