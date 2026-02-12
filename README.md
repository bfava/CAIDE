# CAIDE: Covariate-Adjusted Inference on the Distribution of Treatment Effects

R implementation of the **CAIDE** estimator from:

> Fava, B. (2024). *Predicting the Distribution of Treatment Effects via Covariate-Adjustment, with an Application to Microcredit.* [arXiv:2407.14635](https://arxiv.org/abs/2407.14635).

CAIDE bounds the fraction of units with treatment effects below (or above) a given threshold in randomized experiments, using pre-treatment covariates and machine learning to sharpen classical [Makarov (1982)](https://doi.org/10.1137/1126065) bounds.

---

## Overview

In a randomized experiment with outcome $Y$, binary treatment $D$, and pre-treatment covariates $X$, the parameter of interest is:

$$\theta(\delta) = P\bigl(Y(1) - Y(0) \le \delta\bigr)$$

which represents the fraction of units with an individual treatment effect at or below $\delta$. When $\delta = 0$, this is the **fraction of units harmed** by the treatment.

Without further assumptions, $\theta(\delta)$ is only partially identified. Classical Makarov bounds provide sharp bounds using only the marginal distributions of treated and control outcomes. **CAIDE improves these bounds** by leveraging pre-treatment covariates $X$ through machine learning, while maintaining valid inference.

### Key Features

- **Tighter bounds** than standard Makarov bounds when covariates are predictive of outcomes
- **Two estimation strategies**: cross-fitting (recommended for most applications) and sample splitting
- **Valid asymptotic inference** with standard errors and confidence intervals using cross-fitting
- **Finite-sample valid** confidence intervals via the sample-splitting estimator

---

## Installation

<!-- Clone this repository:

```bash
git clone https://github.com/bfava/CAIDE.git
cd CAIDE
``` -->

<!-- ### R Dependencies -->

Install the required packages:

```r
install.packages(c("tidyverse", "rsample", "ranger", "doMC", "foreach"))
```

For additional ML models (optional):

```r
install.packages(c("mlr3", "mlr3learners", "mlr3tuning",
                    "mlr3tuningspaces", "mlr3extralearners", "qrnn"))
```

---

## Quick Start

```r
source("R/caide.R")

# Your data: Y (outcome), X (covariates), D (treatment), pX (propensity scores)
# For a completely randomized experiment: pX <- rep(mean(D), length(D))

result <- caide_cf(
  Y = Y, X = X, D = D, pX = pX,
  delta = 0,
  K = 5,
  models = "quant_rf",
  quants_seq = seq(0, 1, length.out = 101),
  mode = "cdf"
)

print_caide_results(result)
```

---

## Functions

### Main Estimators

| Function | Description |
|---|---|
| `caide_cf()` | **Cross-fitting estimator** (recommended). Uses K-fold cross-fitting for valid inference with efficient data use. |
| `caide_ss()` | **Sample-splitting estimator**. Finite-sample valid confidence intervals using DKW-type critical values. |

### Utility

| Function | Description |
|---|---|
| `print_caide_results()` | Prints a formatted summary of bounds, standard errors, confidence intervals, and p-values. |

### Internal (called automatically)

| Function | Description |
|---|---|
| `calc_cdf()` | Estimates conditional CDFs via ML (quantile RF, quantile NN, or mlr3 learners). |
| `calc_expectation()` | Estimates conditional expectations via ML (mlr3 learners). |
| `theta_t()` | Evaluates the Makarov-type statistic at a threshold. |
| `sigma2_hat_pX_known()` | Variance estimator (known propensity scores). |
| `sigma2_hat_pX_hat()` | Variance estimator (estimated/stratified propensity scores). |
| `makarov_cf()` | Computes Makarov bounds with cross-fitting inference. |
| `makarov_ss()` | Computes Makarov bounds with sample-splitting inference. |
| `sharp_makarov()` | Classical Makarov bounds without covariates. |
| `.calc_s_star()` | Finds optimal conditioning points for lower and upper bounds. |
| `.calc_s_star_oneside()` | Finds optimal conditioning point for one side only. |

---

## Detailed Usage

### `caide_cf()` — Cross-Fitting Estimator

```r
result <- caide_cf(
  Y,                    # Numeric vector of outcomes
  X,                    # Data frame of covariates
  D,                    # Binary treatment (0/1)
  pX,                   # Propensity scores
  delta = 0,            # Treatment effect threshold
  K = 5,                # Cross-fitting folds
  K_sub = 3,            # Inner CV folds (model selection)
  alpha = 0.05,         # Significance level
  models = "quant_rf",  # ML model(s) to use
  quants_seq = seq(0, 1, length.out = 101),
  ncores = 1L,          # Parallel cores
  mode = "cdf",         # "cdf" or "exp"
  tune = FALSE          # Hyperparameter tuning
)
```

**Arguments:**

- **`delta`**: Threshold for the treatment effect. Set `delta = 0` to bound the fraction harmed. Set `delta = c` for any constant `c` to bound $P(Y(1) - Y(0) \le c)$.

- **`models`**: One or more ML models. Options:
  - `"quant_rf"` — Quantile random forest via `ranger` (recommended)
  - `"quant_nn"` — Quantile neural network via `qrnn`
  - `"none"` — No covariate adjustment (classical Makarov bounds)
  - Any `mlr3` regression learner: `"regr.ranger"`, `"regr.xgboost"`, `"regr.svm"`, `"regr.nnet"`, `"regr.glmnet"`, etc.

  When multiple models are provided, CAIDE automatically selects the best model per cross-fitting fold using an inner cross-validation loop.

- **`mode`**:
  - `"cdf"` — Uses estimated conditional CDFs to find optimal conditioning (sharper bounds, slower). Works with all model types.
  - `"exp"` — Uses estimated conditional expectations as the conditioning variable (faster, requires mlr3 learners).

**Output:** A list containing:
- `makarov_lower`, `makarov_upper`: Estimated bounds on $P(Y(1) - Y(0) \le \delta)$
- `sigma2_L`, `sigma2_U`: Variance estimates
- `vira`: SJLS estimator results (for comparison)

**Inference:**
```r
# 95% confidence interval for the lower bound
ci_low <- max(result$makarov_lower - qnorm(0.95) * sqrt(result$sigma2_L), 0)
ci_high <- result$makarov_lower + qnorm(0.95) * sqrt(result$sigma2_L)

# p-value for H0: theta(delta) >= 0 vs H1: theta(delta) > 0
pval <- 1 - pnorm(result$makarov_lower / sqrt(result$sigma2_L))
```

### `caide_ss()` — Sample-Splitting Estimator

```r
result <- caide_ss(
  Y, X, D, pX,
  delta = 0,
  K_sub = 2,
  prop = 0.5,           # Training proportion
  alpha = 0.1,          # Significance level
  models = "quant_rf",
  quants_seq = seq(0, 1, length.out = 101),
  ncores = 1L,
  tune = FALSE
)
```

**Inference (finite-sample valid):**
```r
# Confidence interval
ci_lower <- max(result$makarov_lower - result$c_alpha, 0)
ci_upper <- min(result$makarov_upper + result$c_alpha, 1)
```

---

## Supported ML Models

| Model ID | Package | Type | Description |
|---|---|---|---|
| `quant_rf` | `ranger` | CDF | Quantile random forest (recommended) |
| `quant_nn` | `qrnn` | CDF | Quantile neural network |
| `regr.ranger` | `mlr3` | CDF/Exp | Random forest regression |
| `regr.xgboost` | `mlr3` | CDF/Exp | Gradient boosted trees |
| `regr.svm` | `mlr3` | CDF/Exp | Support vector machine |
| `regr.nnet` | `mlr3` | CDF/Exp | Neural network |
| `regr.glmnet` | `mlr3` | CDF/Exp | Elastic net regression |
| `none` | — | — | No covariate adjustment |

---

## Examples

See the `examples/` directory:

- **[`example_application.R`](examples/example_application.R)**: Step-by-step walkthrough using simulated experimental data
- **[`example_simulation.R`](examples/example_simulation.R)**: Monte Carlo study evaluating CAIDE performance

---

## Repository Structure

```
CAIDE/
├── R/
│   └── caide.R                    # All CAIDE functions
├── examples/
│   ├── example_application.R      # Application walkthrough
│   └── example_simulation.R       # Monte Carlo simulation
├── README.md
├── LICENSE
└── .gitignore
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{fava2024distribution,
  title={The Distribution of Treatment Effects: Covariate-Adjusted Inference},
  author={Fava, Bruno},
  year={2024},
  note={Working Paper}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
