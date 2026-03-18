"""Penalized linear and GLM regression via the PIC method."""

import numpy as np

from .base import _BaseRegression
from ..families import Family, Gaussian
from ..penalties import Penalty, L1Penalty


class LinearRegression(_BaseRegression):
    """Penalized regression for continuous and count responses as presented in the PIC framework [1].

    Fits a generalized linear model of the form

    .. math::

        \\hat\\beta = \\arg\\min_{\\beta}\\;
        \\phi\\!\\bigl(\\ell(y,\\, g(X\\beta + b))\\bigr) + \\lambda\\, P(\\beta)

    where :math:`\\ell` is the appropriate loss function for the specified family,
    and :math:`\\lambda` is the Pivotal Detection Boundary (PDB), selected
    automatically by Monte Carlo simulation, unless a fixed value is
    supplied.

    The default family is :class:`~picpy.families.Gaussian` (ordinary linear
    regression).  Other families — :class:`~picpy.families.Poisson`,
    :class:`~picpy.families.Exponential`, :class:`~picpy.families.Gumbel` — can
    be passed via the *family* argument to fit the corresponding GLM.

    Parameters
    ----------
    family : Family, default=Gaussian()
        Distributional family specifying the loss function and link.
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
        Fitted regression coefficients.
    intercept_ : float or None
        Fitted intercept.  None when *fit_intercept=False*.
    lambda_ : float
        Selected (or user-supplied) regularization parameter.
    selected_features_ : ndarray of int
        Indices of features with non-zero fitted coefficients.
    is_fitted_ : bool
        True after :meth:`fit` has been called.

    Examples
    --------
    Ordinary linear regression with PDB-selected L1 regularization:

    >>> import numpy as np
    >>> from picpy import LinearRegression
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 20))
    >>> beta_true = np.zeros(20); beta_true[:3] = [2.0, -1.5, 1.0]
    >>> y = X @ beta_true + rng.standard_normal(200)
    >>> model = LinearRegression()
    >>> model.fit(X, y)
    LinearRegression()
    >>> model.selected_features_
    array([0, 1, 2])

    Using the Poisson family for count data:

    >>> from picpy.families import Poisson
    >>> model_pois = LinearRegression(family=Poisson())

    References
    ----------
    .. [1] Sardy, S., van Cutsem, M., and van de Geer, S. (2026).
       "The Pivotal Information Criterion."
       arXiv:2603.04172. Available at https://arxiv.org/abs/2603.04172
    """

    def __init__(
        self,
        family: Family = Gaussian(),
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
            family=family,
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
            f"LinearRegression(family={self.family!r}, penalty={self.penalty!r}, "
            f"lambda_={self.pdb_selector.value!r})"
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the coefficient of determination :math:`R^2` on test data.

        .. math::

            R^2 = 1 - \\frac{\\sum_i (y_i - \\hat y_i)^2}
                            {\\sum_i (y_i - \\bar y)^2}

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test design matrix.
        y : array-like of shape (n_samples,)
            True response values.

        Returns
        -------
        float
            :math:`R^2` score.  1.0 indicates a perfect fit; 0.0 is the
            score of a constant (mean) predictor.
        """
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
