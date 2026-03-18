# PIC — Pivotal Information Criterion

**PIC** is a Python library for high-dimensional sparse regression and classification. It implements the _Pivotal Detection Boundary_ (PDB) method for automatic, data-driven regularisation parameter selection — no cross-validation required.

> Sardy, S., van Cutsem, M., and van de Geer, S. (2026).
> _The Pivotal Information Criterion._ arXiv:2603.04172

---

## Features

- **Automatic λ selection** via the PDB method — controls the family-wise type-I error rate at level α
- **Three computation modes** for λ: exact Monte Carlo, Gaussian CLT approximation, or closed-form Bonferroni bound
- **Full GLM support** — Gaussian, Binomial, Poisson, Exponential, Gumbel, Cox proportional hazards
- **Three penalties** — L1 (LASSO), SCAD, MCP, with warm-started regularisation path
- **FISTA optimiser** with Nesterov momentum and backtracking line search

---

## Installation

```bash
pip install numpy pandas scipy matplotlib scienceplots
```

Clone and run from the repository root (formal packaging in progress):

```bash
git clone <repo-url>
cd PIC
python examples.ipynb   # or: jupyter notebook examples.ipynb
```

---

## Quick Start

```python
import numpy as np
from picpy import LinearRegression, LogisticRegression, CoxRegression

rng = np.random.default_rng(42)
n, p = 300, 30
X = rng.standard_normal((n, p))
beta = np.zeros(p); beta[:5] = [2.0, -1.5, 1.2, -0.8, 0.6]
y = X @ beta + rng.standard_normal(n)

model = LinearRegression()
model.fit(X, y)
model.summary()
```

```
+---------------------------------------------------+
|  Model         : LinearRegression                 |
|  Family        : Gaussian                         |
|  Penalty       : L1Penalty                        |
|  n             :        300                       |
|  p             :         30                       |
|  fit_intercept :       True                       |
+---------------------------------------------------+
|  lambda (PDB)  :     0.2031                       |
|  alpha         :     0.0500                       |
+---------------------------------------------------+
|  Selected variables (5 / 30)                      |
|  feature       :       coef                       |
|  0             :     +1.9841                      |
|  1             :     -1.4763                      |
|  2             :     +1.1892                      |
|  3             :     -0.7954                      |
|  4             :     +0.5801                      |
+---------------------------------------------------+
```

---

## Models

### `LinearRegression`

Penalised GLM for continuous and count responses. Defaults to the Gaussian family; any family can be passed via the `family` argument.

```python
from picpy import LinearRegression
from picpy.families import Poisson
from picpy.penalties import SCADPenalty

# Gaussian + L1 (default)
model = LinearRegression()

# Poisson count data + SCAD penalty
model = LinearRegression(family=Poisson(), penalty=SCADPenalty())

model.fit(X, y)
print(model.selected_features_)   # indices of non-zero coefficients
print(model.coef_table)           # tidy DataFrame sorted by |coef|
print(model.lambda_)              # selected regularisation parameter
```

### `LogisticRegression`

Binary classification with logistic link.

```python
from picpy import LogisticRegression
from picpy.penalties import MCPPenalty

clf = LogisticRegression(penalty=MCPPenalty())
clf.fit(X, y_bin)

clf.predict(X)           # class labels {0, 1}
clf.predict_proba(X)     # class-1 probabilities
clf.score(X, y_bin)      # accuracy
```

### `CoxRegression`

Sparse proportional hazards model via Breslow partial likelihood.
Response `y` must be a 2-column array `[time, event]`.

```python
from picpy import CoxRegression

cox = CoxRegression()
cox.fit(X, y_surv)        # y_surv: (n, 2) array [time, event]

cox.score(X, y_surv)                   # Harrell's C-index
cox.predict_survival_function(X_new)   # individual survival curves
```

---

## Families and Penalties

| Family        | Response type          | Link     | Variance stabiliser |
| ------------- | ---------------------- | -------- | ------------------- |
| `Gaussian`    | Continuous             | Identity | Square-root         |
| `Binomial`    | Binary {0,1}           | Logistic | Identity            |
| `Poisson`     | Count ≥ 0              | Log      | Identity            |
| `Exponential` | Positive continuous    | Log      | Identity            |
| `Gumbel`      | Continuous             | Identity | Exponential         |
| `Cox`         | Survival [time, event] | Identity | Square-root         |

| Penalty                 | Parameter           | Notes                             |
| ----------------------- | ------------------- | --------------------------------- |
| `L1Penalty()`           | —                   | LASSO; convex                     |
| `SCADPenalty(a=3.7)`    | `a` — concavity     | Nearly unbiased for large signals |
| `MCPPenalty(gamma=3.0)` | `gamma` — concavity | Minimax concave; near-unbiased    |

---

## λ Computation Methods

The regularisation parameter is set to the `(1 − α)`-quantile of the null gradient norm distribution. Three estimation strategies are available:

```python
# Exact Monte Carlo (default) — most accurate
model = LinearRegression(lambda_method="mc_exact", lambda_n_simu=5000)

# Gaussian CLT approximation — no null MLE fitting required, faster for large n
model = LinearRegression(lambda_method="mc_gaussian", lambda_n_simu=5000)

# Bonferroni closed-form — instant, valid for large n
model = LinearRegression(lambda_method="analytical")
```

The approximations are asymptotically valid for variance-stabilised families (Gaussian, Binomial, Poisson, Exponential) when _n_ is large relative to _p_.

---

## Key Parameters

| Parameter       | Default       | Description                                             |
| --------------- | ------------- | ------------------------------------------------------- |
| `penalty`       | `L1Penalty()` | Regularisation penalty                                  |
| `family`        | `Gaussian()`  | Regression type                                         |
| `lambda_n_simu` | `5000`        | MC draws for PDB λ selection                            |
| `lambda_alpha`  | `0.05`        | Type-I error level                                      |
| `lambda_method` | `"mc_exact"`  | `"mc_exact"` / `"mc_gaussian"` / `"analytical"`         |
| `fit_intercept` | `True`        | Estimate unpenalised intercept                          |
| `normalize`     | `True`        | Standardise X before fitting                            |
| `tol`           | `1e-8`        | FISTA convergence tolerance                             |
| `refit`         | `False`       | Unpenalised refit on selected support (passed to `fit`) |

---

## Stability Selection

```python
from picpy.utils import StabilitySelection

ss = StabilitySelection(
    model              = LinearRegression(lambda_n_simu=500),
    m                  = 100,    # bootstrap subsamples
    subsample_fraction = 0.5,    # half-sample (Meinshausen & Bühlmann, 2010)
    random_state       = 42,
)
ss.fit(X, y)

print(ss.selection_table_)   # selection_frequency, n_selected per feature
ss.plot(threshold=0.6)       # bar chart with threshold reference line
```

---

## Visualisations

All plots use the PIC theme built on [SciencePlots](https://github.com/garrettj403/SciencePlots).

```python
from picpy.utils.visuals import plot_cox_summary, plot_stability, plot_pdb_distribution

# PDB null distribution with λ threshold
plot_pdb_distribution(
    statistics   = model.pdb_selector.statistics,
    lambda_value = model.pdb_selector.value,
    alpha        = model.pdb_selector.alpha,
)

# Stability selection bar chart
ss.plot(threshold=0.6)

# Cox big-picture: baseline curves + per-feature survival effects
fig = plot_cox_summary(cox, X)
```

---

## Reference

```bibtex
@article{sardy2026pic,
  title   = {The Pivotal Information Criterion},
  author  = {Sardy, Sylvain and van Cutsem, Maxime and van de Geer, Sara},
  journal = {arXiv preprint arXiv:2603.04172},
  year    = {2026}
}
```
