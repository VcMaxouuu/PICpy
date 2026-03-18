"""Penalized logistic regression for binary classification via the PIC method."""

import numpy as np

from .base import _BaseRegression
from ..families import Binomial
from ..penalties import Penalty, L1Penalty


class LogisticRegression(_BaseRegression):
    """Penalized regression for binary classification as presented in the PIC framework [1].

    Fits a binary classification model with Binomial loss and logistic link:

    .. math::

        \\hat\\beta = \\arg\\min_{\\beta}\\;
        \\ell_{\\text{Bin}}(y,\\, \\sigma(X\\beta + b)) + \\lambda\\, P(\\beta)

    where :math:`\\sigma` denotes the logistic (sigmoid) function, :math:`\\ell_{\\text{Bin}}` is
    the custom Binomial loss defined in PIC:

    .. math::
        \\ell_{\\text{Bin}}(y, p) = 2y\\sqrt{\\frac{1-p}{p}} + 2(1-y) \\sqrt{\\frac{p}{1-p}}


    and :math:`\\lambda` is the Pivotal Detection Boundary (PDB), selected
    automatically by Monte Carlo simulation, unless a fixed value is
    supplied.

    Parameters
    ----------
    penalty : Penalty, default=L1Penalty()
        Regularization penalty.  One of :class:`~picpy.penalties.L1Penalty`,
        :class:`~picpy.penalties.SCADPenalty`, or
        :class:`~picpy.penalties.MCPPenalty`.
    lambda_ : float or None, default=None
        Fixed regularization parameter.  When None the PDB method selects
        :math:`\\lambda` automatically.
    fit_intercept : bool, default=True
        Whether to estimate an unpenalized intercept.
    normalize : bool, default=True
        Whether to standardize *X* to zero mean and unit variance before
        fitting.
    lambda_n_simu : int, default=5000
        Number of Monte Carlo draws for the PDB null distribution.
    lambda_alpha : float, default=0.05
        Nominal type-I error level for PDB regularization selection.
    lambda_method : {"mc_exact", "mc_gaussian", "analytical"}, default="mc_exact"
        Method for computing the PDB regularization parameter.  See
        :class:`~picpy.lambda_pdb.LambdaPDB` for details.
    tol : float, default=1e-8
        Relative convergence tolerance for FISTA at the final path step.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Fitted log-odds coefficients.
    intercept_ : float or None
        Fitted intercept (log-odds).  None when *fit_intercept=False*.
    lambda_ : float
        Selected (or user-supplied) regularization parameter.
    selected_features_ : ndarray of int
        Indices of features with non-zero fitted coefficients.
    is_fitted_ : bool
        True after :meth:`fit` has been called.

    Examples
    --------
    >>> import numpy as np
    >>> from picpy import LogisticRegression
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((300, 20))
    >>> beta_true = np.zeros(20); beta_true[:3] = [2.0, -1.5, 1.0]
    >>> proba = 1 / (1 + np.exp(-X @ beta_true))
    >>> y = rng.binomial(1, proba)
    >>> model = LogisticRegression()
    >>> model.fit(X, y)
    LogisticRegression()
    >>> model.selected_features_
    array([0, 1, 2])
    >>> model.score(X, y)   # classification accuracy

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
        fit_intercept: bool = True,
        normalize: bool = True,
        lambda_n_simu: int = 5000,
        lambda_alpha: float = 0.05,
        lambda_method: str = "mc_exact",
        tol: float = 1e-8,
    ):
        super().__init__(
            family=Binomial(),
            penalty=penalty,
            lambda_=lambda_,
            fit_intercept=fit_intercept,
            normalize=normalize,
            lambda_n_simu=lambda_n_simu,
            lambda_alpha=lambda_alpha,
            lambda_method=lambda_method,
            tol=tol,
        )

    def __repr__(self) -> str:
        return (
            f"LogisticRegression(penalty={self.penalty!r}, "
            f"lambda_={self.pdb_selector.value!r})"
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix.

        Returns
        -------
        ndarray of shape (n_samples,)
            Estimated probability of class 1.
        """
        return self.family.g(self._linear_predictor(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (0 or 1).

        Thresholds :meth:`predict_proba` at 0.5.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix.

        Returns
        -------
        ndarray of shape (n_samples,) with dtype int
        """
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy on test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test design matrix.
        y : array-like of shape (n_samples,)
            True binary labels (0 or 1).

        Returns
        -------
        float
            Proportion of correctly classified observations.
        """
        return float(np.mean(self.predict(X) == np.asarray(y, dtype=int)))
