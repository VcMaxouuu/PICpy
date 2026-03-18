"""Base class for all models presented in the PIC framework."""

from abc import ABC
from typing import Optional

import numpy as np
from pandas import DataFrame

from ..families import Family
from ..penalties import Penalty, L1Penalty
from ..lambda_pdb import LambdaPDB
from ..utils.validate_input import check_X, check_Xy
from ..utils.training_logger import TrainingLogger
from ..utils.fista import fista


class _BaseRegression(ABC):
    """Abstract base class for PIC regression models.

    Encapsulates the shared fitting pipeline common to all models in the
    PIC (Pivotal Information Criterion) framework [1]:

    1. Input validation and optional standardization of *X*.
    2. Automatic regularization parameter selection via the PDB method.
    3. A warm-started regularization path (4 steps).
    4. Optional unpenalized refit on the selected support.

    Subclasses fix the family and optionally override :meth:`_after_fit` and
    :meth:`score`.

    Parameters
    ----------
    family : Family
        Distributional family (loss function, link, variance stabilizer).
    penalty : Penalty, default=L1Penalty()
        Sparsity-inducing penalty applied at the final regularization step.
        L1 is always used for the warm-up steps regardless of this choice.
    lambda_ : float or None, default=None
        Fixed regularization parameter.  When None the PDB method selects
        :math:`\\lambda` automatically.
    fit_intercept : bool, default=True
        Whether to estimate an unpenalized intercept term.
    normalize : bool, default=True
        Whether to standardize *X* to zero mean and unit variance before
        fitting.  Strongly recommended for PDB to be calibrated.
    lambda_n_simu : int, default=5000
        Number of Monte Carlo null draws used by the PDB selector.
    lambda_alpha : float, default=0.05
        Nominal level of the PDB selector (type-I error rate).
    lambda_method : {"mc_exact", "mc_gaussian", "analytical"}, default="mc_exact"
        Method for computing the PDB regularization parameter.

        - ``"mc_exact"`` — Monte Carlo simulation from the family's null
          distribution (default, most accurate).
        - ``"mc_gaussian"`` — Gaussian CLT approximation; draws
          :math:`W \\sim N(0, I_n)` and estimates
          :math:`\\lambda = n^{-1}\\,Q_{1-\\alpha}(\\|X^\\top W\\|_\\infty)`.
          Slightly faster than ``"mc_exact"`` since no MLE fitting; requires large *n*.
        - ``"analytical"`` — Bonferroni bound
          :math:`\\lambda = n^{-1/2}\\,\\Phi^{-1}(1 - \\alpha / (2p))`.
          No simulation; requires large *n*.
    tol : float, default=1e-8
        Relative convergence tolerance for FISTA at the final path step.

    References
    ----------
    .. [1] Sardy, S., van Cutsem, M., and van de Geer, S. (2026).
       "The Pivotal Information Criterion."
       arXiv:2603.04172. Available at https://arxiv.org/abs/2603.04172
    """

    def __init__(
        self,
        family: Family,
        penalty: Penalty = L1Penalty(),
        lambda_: float | None = None,
        fit_intercept: bool = True,
        normalize: bool = True,
        lambda_n_simu: int = 5000,
        lambda_alpha: float = 0.05,
        lambda_method: str = "mc_exact",
        tol: float = 1e-8,
    ):
        self.family = family
        self.penalty = penalty
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.pdb_selector = LambdaPDB(
            lambda_=lambda_,
            n_simu=lambda_n_simu,
            alpha=lambda_alpha,
            method=lambda_method,
        )
        self.tol = tol

        # Set during fit
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.n_samples_in_: Optional[int] = None
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

        # Internal preprocessing statistics
        self._X_mean: Optional[np.ndarray] = None
        self._X_std: Optional[np.ndarray] = None
        self._fit_info: Optional[dict] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lambda_(self) -> float | None:
        """Selected (or user-supplied) regularization parameter.

        Returns None before :meth:`fit` has been called.
        """
        return self.pdb_selector.value

    @property
    def selected_features_(self) -> np.ndarray:
        """Indices of features with non-zero fitted coefficients.

        Returns
        -------
        ndarray of int
            Zero-based column indices of the selected features.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        self._check_is_fitted()
        return np.where(self.coef_ != 0)[0]

    @property
    def n_selected_(self) -> int:
        """Number of features with non-zero fitted coefficients.

        Returns
        -------
        int

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        self._check_is_fitted()
        return int(np.sum(self.coef_ != 0))

    @property
    def selected_feature_names_(self) -> np.ndarray:
        """Names of features with non-zero fitted coefficients.

        Returns
        -------
        ndarray of str

        Raises
        ------
        ValueError
            If the model has not been fitted yet, or if no feature names
            are available (input was a plain array, not a DataFrame).
        """
        self._check_is_fitted()
        if self.feature_names_in_ is None:
            raise ValueError(
                "Feature names are not available. Fit the model with a "
                "pandas DataFrame to enable this property."
            )
        return self.feature_names_in_[self.selected_features_]

    @property
    def coef_table(self) -> DataFrame:
        """Return a tidy summary of estimated coefficients.

        One row per feature.  Selected features (non-zero coefficients) appear
        first, sorted by descending absolute coefficient value, followed by the
        unselected features.

        Returns
        -------
        pd.DataFrame
            Columns:

            - ``feature`` — feature name (column index when no names are
              available).
            - ``coef`` — estimated coefficient.
            - ``abs_coef`` — absolute value of the coefficient.
            - ``selected`` — True when the coefficient is non-zero.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model.fit(X, y)
        >>> model.coef_table()
           feature     coef  abs_coef  selected
        0        0   2.1032    2.1032      True
        1        2   0.9841    0.9841      True
        2        1  -1.4720    1.4720      True
        ...
        """
        self._check_is_fitted()

        if self.feature_names_in_ is not None:
            names = list(self.feature_names_in_)
        else:
            names = list(range(self.n_features_in_))

        abs_coef = np.abs(self.coef_)
        selected = self.coef_ != 0

        df = DataFrame(
            {
                "coef": self.coef_,
                "abs_coef": abs_coef,
                "selected": selected,
            },
            index=names,
        )

        df.index.name = "feature"

        # Selected features first, sorted by |coef| descending; then unselected
        df = df.sort_values(
            ["selected", "abs_coef"], ascending=[False, False]
        )

        return df

    def summary(self) -> str:
        """Print and return a formatted summary of the fitted model.

        Displays three sections:

        1. **Model** — class name, family, penalty, and fit settings.
        2. **Regularization (PDB)** — selected lambda and, when computed via
           Monte Carlo, the key quantiles of the null distribution.
        3. **Selected variables** — coefficient table for non-zero features,
           sorted by descending absolute value; intercept appended when fitted.

        Returns
        -------
        str
            Multi-line summary string (also printed).

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model.fit(X, y)
        >>> model.summary()
        +---------------------------------------------------+
        |  Model         : LinearRegression                 |
        |  Family        : Gaussian                         |
        |  Penalty       : L1Penalty                        |
        |  n             :        200                       |
        |  p             :         20                       |
        |  fit_intercept :       True                       |
        +---------------------------------------------------+
        |  lambda (PDB)  :     0.2254                       |
        |  alpha         :     0.0500                       |
        |  95th pct      :     0.2254 <- lambda             |
        +---------------------------------------------------+
        |  Selected variables (3 / 20)                      |
        |  feature       :       coef                       |
        |  x0            :     2.0821                       |
        |  x1            :    -1.4763                       |
        |  x2            :     0.9912                       |
        |  intercept     :     0.0312                       |
        +---------------------------------------------------+
        """
        self._check_is_fitted()

        inner_width = 49

        def sep() -> str:
            return "+" + "-" * (inner_width + 2) + "+"

        def row(label: str, value: str, marker: str = "") -> str:
            content = f"  {label:<16}: {value:>10}{marker}"
            return f"| {content:<{inner_width}} |"

        def section_header(title: str) -> str:
            content = f"  {title}"
            return f"| {content:<{inner_width}} |"

        lines = [sep()]

        # --- Section 1: Model ---
        lines.append(row("Model",         type(self).__name__))
        lines.append(row("Family",        type(self.family).__name__))
        lines.append(row("Penalty",       type(self.penalty).__name__))
        lines.append(row("n",             f"{self.n_samples_in_:,}"))
        lines.append(row("p",             f"{self.n_features_in_:,}"))
        lines.append(row("fit_intercept", str(self.fit_intercept)))

        # --- Section 2: PDB ---
        lines.append(sep())
        pdb = self.pdb_selector
        lines.append(row("lambda (PDB)",  f"{pdb.value:.4f}"))
        lines.append(row("alpha",         f"{pdb.alpha:.4f}"))

        # --- Section 3: Selected variables ---
        lines.append(sep())
        n_sel = self.n_selected_
        lines.append(section_header(
            f"Selected variables ({n_sel} / {self.n_features_in_})"
        ))
        lines.append(row("feature", "coef"))

        if self.feature_names_in_ is not None:
            names = list(self.feature_names_in_)
        else:
            names = [str(i) for i in range(self.n_features_in_)]

        # selected features sorted by |coef| descending
        sel_idx = self.selected_features_
        order   = sel_idx[np.argsort(-np.abs(self.coef_[sel_idx]))]
        for i in order:
            lines.append(row(str(names[i]), f"{self.coef_[i]:+.4f}"))

        if self.fit_intercept and self.intercept_ is not None:
            lines.append(row("intercept", f"{self.intercept_:+.4f}"))

        if n_sel == 0:
            lines.append(section_header("  (no features selected)"))

        lines.append(sep())

        result = "\n".join(lines)
        print(result)
        return result

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbosity: int = 0,
        log_interval: int = 10,
        refit: bool = False,
    ) -> "_BaseRegression":
        """Fit the model using PDB-regularized FISTA on a warm-started path.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix.
        y : array-like of shape (n_samples,) or (n_samples, 2)
            Response vector.  For survival models *y* must be a 2-column
            matrix ``[time, event]``.
        verbosity : int, default=0
            Amount of output printed to stdout:

            - ``0`` — silent.
            - ``1`` — high-level progress (PDB result, per-step convergence,
              final summary).
            - ``2`` — adds per-iteration objective values.

        log_interval : int, default=10
            At ``verbosity=2``, print one iteration line every *log_interval*
            iterations.
        refit : bool, default=False
            Whether to run an unpenalized refit on the selected support after
            the regularization path.  Reduces estimation bias.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # --- Reset family state so every fit starts clean ---
        self.family._reset_fit_state()

        # --- Store meta ---
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else len(X[0])
        self.n_samples_in_ = X.shape[0] if hasattr(X, "shape") else len(X)
        if isinstance(X, DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()

        # --- Validate and standardize ---
        X, X_mean, X_std, y = check_Xy(
            X, y, y_kind=self.family.y_kind(), normalize_X=self.normalize
        )

        logger = TrainingLogger(verbosity=verbosity, log_interval=log_interval)

        # --- PDB regularization parameter ---
        logger.start_pdb_computation(
            n_simu=self.pdb_selector.n_simu,
            alpha=self.pdb_selector.alpha,
        )
        lambda_val = self.pdb_selector.compute(X=X, family=self.family)
        logger.log_pdb_result(
            lambda_value=lambda_val,
            pdb_statistics=self.pdb_selector.statistics,
        )

        # --- Warm-started regularization path ---
        path_fractions = [(i / 4) ** 0.8 for i in range(1, 5)]
        lambda_path = [lambda_val * f for f in path_fractions]
        tol_path = [1e-5] * (len(lambda_path) - 1) + [self.tol]

        logger.start_fitting(
            n_samples=X.shape[0],
            n_features=X.shape[1],
            lambda_reg=lambda_val,
            penalty_name=str(self.penalty),
        )

        is_l1 = isinstance(self.penalty, L1Penalty)
        warmup_penalty = self.penalty if is_l1 else L1Penalty()

        coef_prev: Optional[np.ndarray] = None
        intercept_prev: Optional[float] = None
        training_info: dict = {}

        for idx, (lam, tol) in enumerate(zip(lambda_path, tol_path)):
            logger.log_path_start(
                path_idx=idx,
                total_paths=len(lambda_path),
                lambda_current=lam,
                rel_tol=tol,
            )
            # Use L1 for warm-up steps; switch to specified penalty at last step
            step_penalty = (
                self.penalty if (is_l1 or idx == len(lambda_path) - 1)
                else warmup_penalty
            )

            coef_, intercept_, info = fista(
                X=X,
                y=y,
                family=self.family,
                penalty=step_penalty,
                lambda_reg=lam,
                fit_intercept=self.fit_intercept,
                rel_tol=tol,
                logger=logger,
                coef_init=coef_prev,
                intercept_init=intercept_prev,
            )
            coef_prev = coef_
            intercept_prev = intercept_
            training_info[f"step_{idx}"] = info

        # --- Store penalized solution ---
        self.coef_ = coef_
        self.intercept_ = intercept_
        self.is_fitted_ = True
        self._X_mean = X_mean
        self._X_std = X_std
        self._fit_info = training_info

        # --- Optional unpenalized refit on selected support ---
        if refit:
            support = self.selected_features_
            logger.start_refit(n_selected=len(support))

            if len(support) > 0:
                coef_refit, intercept_refit, info = fista(
                    X=X[:, support],
                    y=y,
                    family=self.family,
                    penalty=self.penalty,
                    lambda_reg=0.0,
                    fit_intercept=self.fit_intercept,
                    rel_tol=1e-6,
                    logger=logger,
                    coef_init=self.coef_[support],
                    intercept_init=self.intercept_,
                )
                coef_full = np.zeros_like(self.coef_)
                coef_full[support] = coef_refit
                self.coef_ = coef_full
                self.intercept_ = intercept_refit
                # save training info for refit step
                self._fit_info[f"step_{idx + 1}"] = info

        logger.log_final_results(
            n_selected=len(self.selected_features_),
            objective_final=self._fit_info[f"step_{len(self._fit_info) - 1}"][
                "objective_path"
            ][-1],
        )

        self._after_fit(X, y)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _linear_predictor(self, X: np.ndarray) -> np.ndarray:
        """Compute the linear predictor :math:`X\\hat\\beta + \\hat b`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix (raw, unstandardized).

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        self._check_is_fitted()
        X, _, _ = check_X(
            X,
            normalize_X=self.normalize,
            X_mean=self._X_mean,
            X_std=self._X_std,
        )
        eta = X @ self.coef_
        if self.fit_intercept:
            eta = eta + self.intercept_
        return eta

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict fitted values on the response scale.

        Applies the mean-function link *g* to the linear predictor:
        :math:`\\hat y = g(X\\hat\\beta + \\hat b)`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values on the response scale.
        """
        return self.family.g(self._linear_predictor(X))

    # ------------------------------------------------------------------
    # Hook for subclasses
    # ------------------------------------------------------------------

    def _after_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Post-fit hook for subclass-specific computations.

        Called at the end of :meth:`fit` after the final coefficients have
        been stored.  Override in subclasses that need to compute additional
        quantities (e.g. baseline survival functions for Cox regression).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Standardized design matrix used during fitting.
        y : ndarray
            Response vector/matrix used during fitting.
        """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise ValueError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' before using this estimator."
            )
