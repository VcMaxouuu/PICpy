"""Penalized Cox regression for survival analysis via the PIC method."""

import numpy as np
import pandas as pd

from .base import _BaseRegression
from ..families import Cox
from ..penalties import Penalty, L1Penalty
from ..utils.survival import baseline_functions, concordance_index


class CoxRegression(_BaseRegression):
    """Penalized regression for survival analysis as presented in the PIC framework [1].

    Fits a sparse Cox model by minimizing the Breslow partial log-likelihood
    with a PDB-calibrated regularization penalty:

    .. math::

        \\hat\\beta = \\arg\\min_{\\beta}\\;
        -\\frac{1}{n}\\ell_{\\text{Cox}}(\\beta) + \\lambda\\, P(\\beta)

    where :math:`\\lambda` is the Pivotal Detection Boundary (PDB), selected
    automatically by Monte Carlo simulation, unless a fixed value is
    supplied.

    The response *y* must be a 2-column array ``[time, event]``:

    - ``time`` — observed time (follow-up or event time, non-negative).
    - ``event`` — event indicator (1 = event occurred, 0 = censored).

    No intercept is fitted (proportional hazards parameterization).

    Parameters
    ----------
    penalty : Penalty, default=L1Penalty()
        Regularization penalty.
    lambda_ : float or None, default=None
        Fixed regularization parameter.  When None the PDB method selects
        :math:`\\lambda` automatically.
    normalize : bool, default=True
        Whether to standardize *X* to zero mean and unit variance.
    lambda_n_simu : int, default=5000
        Number of Monte Carlo draws for the PDB null distribution.
    lambda_alpha : float, default=0.05
        Nominal type-I error level for PDB regularization selection.
    lambda_method : {"mc_exact", "mc_gaussian", "analytical"}, default="mc_exact"
        Method for computing the PDB regularization parameter.  Note that
        Cox uses a custom pivotal estimator; ``"mc_gaussian"`` and
        ``"analytical"`` fall back to ``"mc_exact"`` with a warning.
    tol : float, default=1e-8
        Relative convergence tolerance for FISTA at the final path step.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Fitted log-hazard-ratio coefficients.
    lambda_ : float
        Selected (or user-supplied) regularization parameter.
    selected_features_ : ndarray of int
        Indices of features with non-zero fitted coefficients.
    baseline_cumulative_hazard_ : pd.DataFrame
        Breslow estimator of the baseline cumulative hazard :math:`H_0(t)`,
        indexed by unique observed times.  Available after :meth:`fit`.
    baseline_survival_ : pd.DataFrame
        Baseline survival function :math:`S_0(t) = \\exp(-H_0(t))`,
        indexed by unique observed times.  Available after :meth:`fit`.
    unique_times_ : ndarray
        Unique observed event/censoring times from the training data.
    is_fitted_ : bool
        True after :meth:`fit` has been called.

    Examples
    --------
    >>> import numpy as np
    >>> from picpy import CoxRegression
    >>> rng = np.random.default_rng(0)
    >>> n, p = 200, 20
    >>> X = rng.standard_normal((n, p))
    >>> beta_true = np.zeros(p); beta_true[:3] = [1.0, -0.8, 0.6]
    >>> lp = X @ beta_true
    >>> T = - np.log(rng.uniform(size=n)) / np.exp(lp)
    >>> C = np.quantile(T, 0.8)  # administrative censoring at 80th percentile
    >>> times = np.minimum(T, C)
    >>> events = (T <= C).astype(int)
    >>> y = np.column_stack([times, events])
    >>> model = CoxRegression()
    >>> model.fit(X, y)
    CoxRegression()
    >>> model.selected_features_
    array([0, 1, 2])
    >>> model.score(X, y)    # Harrell's C-index

    References
    ----------
    .. [1] Sardy, S., van Cutsem, M., and van de Geer, S. (2026).
       "The Pivotal Information Criterion."
       arXiv:2603.04172. Available at https://arxiv.org/abs/2603.04172
    """

    def __init__(
        self,
        penalty: Penalty = L1Penalty(),
        lambda_: float | None = None,
        normalize: bool = True,
        lambda_n_simu: int = 5000,
        lambda_alpha: float = 0.05,
        lambda_method: str = "mc_exact",
        tol: float = 1e-8,
    ):
        super().__init__(
            family=Cox(),
            penalty=penalty,
            lambda_=lambda_,
            fit_intercept=False,
            normalize=normalize,
            lambda_n_simu=lambda_n_simu,
            lambda_alpha=lambda_alpha,
            lambda_method=lambda_method,
            tol=tol,
        )
        self.baseline_cumulative_hazard_: pd.DataFrame | None = None
        self.baseline_survival_: pd.DataFrame | None = None
        self.unique_times_: np.ndarray | None = None

    def __repr__(self) -> str:
        return (
            f"CoxRegression(penalty={self.penalty!r}, "
            f"lambda_={self.pdb_selector.value!r})"
        )

    def _after_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Compute Breslow baseline hazard and survival after fitting."""
        baselines = baseline_functions(y[:, 0], y[:, 1], self.predict(X))
        self.baseline_cumulative_hazard_ = baselines[["baseline_cumulative_hazard"]]
        self.baseline_survival_ = baselines[["baseline_survival"]]
        self.unique_times_ = baselines.index.values

    def predict_survival_function(self, X: np.ndarray) -> pd.DataFrame:
        """Predict individual survival functions for new observations.

        Computes :math:`S(t \\mid x_i) = S_0(t)^{\\exp(x_i^\\top \\hat\\beta)}`
        using the Breslow baseline survival estimated from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix for new observations.

        Returns
        -------
        pd.DataFrame
            Survival probabilities with unique training times as the index
            and sample indices (0, 1, …, n_samples−1) as columns.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        self._check_is_fitted()
        linear_pred = self._linear_predictor(X)
        H0 = self.baseline_cumulative_hazard_["baseline_cumulative_hazard"].to_numpy()
        surv = {i: np.exp(-H0 * np.exp(lp)) for i, lp in enumerate(linear_pred)}
        return pd.DataFrame(surv, index=self.unique_times_)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute Harrell's concordance index (C-index) on test data.

        A value of 1.0 indicates perfect discrimination; 0.5 is the
        expected value under random predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test design matrix.
        y : array-like of shape (n_samples, 2)
            Survival response matrix with columns ``[time, event]``.

        Returns
        -------
        float
            Concordance index in [0, 1].
        """
        y = np.asarray(y)
        linear_pred = self._linear_predictor(X)
        return concordance_index(y[:, 0], y[:, 1], linear_pred)
