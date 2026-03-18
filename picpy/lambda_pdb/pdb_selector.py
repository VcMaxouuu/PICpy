"""Automatic regularization parameter selection via the PDB method presented in the PIC framework."""


import numpy as np
from warnings import warn

from ..families import Family

_METHODS = ("mc_exact", "mc_gaussian", "analytical")


class LambdaPDB:
    """Regularization parameter selector based on the Pivotal Detection Boundary (PDB).

    In the PIC (Pivotal Information Criterion) [1] framework, the regularization
    parameter :math:`\\lambda` is selected as the Pivotal Detection Boundary:
    the smallest value above which no predictor would be selected under the
    null model.  Three estimation methods are supported:

    ``"mc_exact"`` (default)
        Monte Carlo simulation: ``n_simu`` response vectors are drawn from the
        family's null distribution, the :math:`\\ell_\\infty`-norm of the
        gradient at the null intercept is computed for each draw, and
        :math:`\\lambda` is the ``(1 - alpha)``-quantile.  Families with a
        dedicated pivotal estimator (:class:`~picpy.families.Gumbel`,
        :class:`~picpy.families.Cox`) use their custom
        :meth:`~Family._null_grad_norms`; all others use the standard
        gradient.

    ``"mc_gaussian"``
        Gaussian approximation: exploits the fact that, for large *n* and
        variance-stabilized families, :math:`\\sqrt{n}\\,\\nabla\\ell \\to
        N(0, \\Sigma_X)` under the null, where
        :math:`\\Sigma_X = X^\\top X / n`.  Instead of simulating responses,
        ``n_simu`` draws :math:`W \\sim N(0, I_n)` are generated and
        :math:`\\lambda` is estimated as

        .. math::

            \\hat\\lambda = \\frac{1}{n}\\,
            Q_{1-\\alpha}\\!\\left(\\|X^\\top W\\|_\\infty\\right)

        This avoids fitting null intercepts.

    ``"analytical"``
        Bonferroni approximation: uses the closed-form bound

        .. math::

            \\hat\\lambda = \\frac{1}{\\sqrt{n}}\\,
            \\Phi^{-1}\\!\\left(1 - \\frac{\\alpha}{2p}\\right)

        No simulation is required.  Requires *n* to be large relative to *p*.

    Parameters
    ----------
    lambda_ : float or None, default=None
        If provided, this fixed value is used instead of running any
        computation.
    n_simu : int, default=5000
        Number of Monte Carlo draws for ``"mc_exact"`` and ``"mc_gaussian"``.
        Ignored for ``"analytical"``.
    alpha : float, default=0.05
        Nominal level.  The regularization parameter is set to the
        ``(1 - alpha)``-quantile of the null gradient norm distribution,
        controlling the family-wise type-I error rate.
    method : {"mc_exact", "mc_gaussian", "analytical"}, default="mc_exact"
        Estimation method.  See class docstring for details.

    Attributes
    ----------
    value : float or None
        The selected (or user-supplied) regularization parameter.  Set after
        calling :meth:`compute`.
    statistics : ndarray of shape (n_simu,) or None
        Simulated statistics from the most recent call to :meth:`compute`.
        For ``"mc_exact"`` these are null gradient norms; for
        ``"mc_gaussian"`` they are the Gaussian inf-norm draws.
        None when a fixed lambda is used or ``method="analytical"``.

    References
    ----------
    .. [1] Sardy, S., van Cutsem, M., and van de Geer, S. (2026).
       "The Pivotal Information Criterion."
       arXiv:2603.04172. Available at https://arxiv.org/abs/2603.04172
    """

    def __init__(
        self,
        lambda_: float | None = None,
        n_simu: int = 5000,
        alpha: float = 0.05,
        method: str = "mc_exact",
    ):
        if method not in _METHODS:
            raise ValueError(
                f"method must be one of {_METHODS!r}; got {method!r}."
            )
        self._fixed = lambda_
        self.n_simu = n_simu
        self.alpha = alpha
        self.method = method
        self.value: float | None = None
        self.statistics: np.ndarray | None = None

    def __str__(self) -> str:
        return (
            f"LambdaPDB(lambda_={self._fixed!r}, n_simu={self.n_simu}, "
            f"alpha={self.alpha}, method={self.method!r})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, X: np.ndarray, family: Family) -> float:
        """Compute (or return the fixed) regularization parameter.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Standardized design matrix.  A warning is issued when the columns
            do not appear to be zero-mean, unit-variance standardized.
        family : Family
            The distributional family defining the loss.

        Returns
        -------
        lambda_ : float
            Selected regularization parameter (Pivotal Detection Boundary).
        """
        if self._fixed is not None:
            self.value = self._fixed
            return self._fixed

        # Warn if X is not properly standardized
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=0)
        if np.max(np.abs(mu)) >= 1e-4 or np.max(np.abs(sd - 1.0)) >= 1e-4:
            warn(
                "LambdaPDB: design matrix X does not appear to be standardized. "
                f"max|mean| = {np.max(np.abs(mu)):.3e}, "
                f"max|std - 1| = {np.max(np.abs(sd - 1.0)):.3e}. "
                "Results may be unreliable."
            )

        c = family.variance_scaling_factor(X.shape[0])

        if self.method == "mc_exact":
            self.value = self._compute_mc_exact(X, family)
        elif self.method == "mc_gaussian":
            self.value = self._compute_mc_gaussian(X, c)
        else:
            self.value = self._compute_analytical(X, c)

        return self.value

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_mc_exact(self, X: np.ndarray, family: Family) -> float:
        """Estimate the PDB threshold by exact Monte Carlo simulation."""
        n = X.shape[0]
        y_null = family.generate_y(size=(n, self.n_simu))

        if hasattr(family, "_null_grad_norms"):
            grad_norms = family._null_grad_norms(X, y_null)
        else:
            intercept_null = family.starting_intercept(y_null)
            coef_grad, _ = family.grad(X, y_null, intercept_null)
            grad_norms = np.max(np.abs(coef_grad), axis=0)

        self.statistics = np.asarray(grad_norms)
        return float(np.quantile(self.statistics, 1.0 - self.alpha))

    def _compute_mc_gaussian(self, X: np.ndarray, c: float) -> float:
        """Estimate the PDB threshold using the Gaussian approximation.

        For families satisfying the corresponding asymptotic null-gradient
        approximation,

        .. math::

            \\nabla \\ell \\approx \\frac{\\sqrt{c}}{n} X^\\top W,
            \\qquad W \\sim N(0, I_n),

        so the PDB threshold is approximated by the empirical
        ``(1 - alpha)``-quantile of the simulated sup-norms.
        """
        n = X.shape[0]
        rng = np.random.default_rng()
        W = rng.standard_normal((n, self.n_simu))
        self.statistics = (np.sqrt(c) / n) * np.max(np.abs(X.T @ W), axis=0)
        return float(np.quantile(self.statistics, 1.0 - self.alpha))

    def _compute_analytical(self, X: np.ndarray, c: float) -> float:
        """Bonferroni closed-form approximation.

        .. math::

            \\hat\\lambda =
            \\sqrt{\\frac{c}{n}}\\,
            \\Phi^{-1}\\!\\left(1 - \\frac{\\alpha}{2p}\\right)
        """
        from scipy.stats import norm

        n, p = X.shape
        self.statistics = None
        return float(norm.ppf(1.0 - self.alpha / (2.0 * p)) * np.sqrt(c / n))


    def summary(self) -> str:
        """Return a formatted summary card of the PDB computation.

        Prints key statistics of the Monte Carlo null distribution and the
        selected regularization parameter.  When a fixed lambda was supplied
        by the user, only that value is shown.

        Returns
        -------
        str
            Multi-line summary string (also printed).

        Raises
        ------
        RuntimeError
            If :meth:`compute` has not been called yet.

        Examples
        --------
        >>> selector = LambdaPDB(n_simu=5000, alpha=0.05)
        >>> selector.compute(X, family)
        >>> selector.summary()
        +---PDB Selector-----------------+
        |  n_simu          :      5,000  |
        |  alpha           :      0.050  |
        |  lambda (PDB)    :      0.198  |
        +---Null distribution------------+
        |  mean            :     0.1584  |
        |  std             :     0.0224  |
        |  25th pct        :     0.1424  |
        |  median          :     0.1563  |
        |  75th pct        :     0.1718  |
        +--------------------------------+
        """
        if self.value is None:
            raise RuntimeError(
                "LambdaPDB.summary() called before compute(). "
                "Call compute(X, family) first."
            )

        inner_width = 30

        def top(title: str) -> str:
            return "+" + "---" + f"{title}" + "-" * (inner_width - 1 - len(title)) + "+"

        def row(label: str, value: str, marker: str = "") -> str:
            content = f" {label:<16}: {value:>10}{marker}"
            return f"| {content:<{inner_width}} |"

        lines = [top("PDB Selector")]
        if self._fixed is not None:
            lines.append(row("method", "user"))
            lines.append(row("lambda (PDB)", f"{self.value:.3f}"))
        elif self.statistics is not None:
            lines.append(row("method", self.method))
            lines.append(row("n_simu", f"{self.n_simu:,}"))
            lines.append(row("alpha", f"{self.alpha:.3f}"))
            lines.append(row("lambda (PDB)", f"{self.value:.3f}"))
            lines.append(top("Null distribution"))
            lines.append(row("mean", f"{np.mean(self.statistics):.4f}"))
            lines.append(row("std", f"{np.std(self.statistics):.4f}"))
            lines.append(row(f"25th pct", f"{np.quantile(self.statistics, 0.25):.4f}"))
            lines.append(row("median", f"{np.median(self.statistics):.4f}"))
            lines.append(row(f"75th pct", f"{np.quantile(self.statistics, 0.75):.4f}"))
        else:
            lines.append(row("method", "analytical"))
            lines.append(row("lambda (PDB)", f"{self.value:.3f}"))

        lines.append(top(""))
        result = "\n".join(lines)
        print(result)
        return result

    def plot_pdb_distribution(self, ax=None, bins: int = 60):
        """Plot the Monte Carlo null distribution with the PDB threshold.

        Examples
        --------
        >>> selector.compute(X, family)
        >>> selector.plot_pdb_distribution()
        """
        from ..utils.visuals import plot_pdb_distribution

        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plot_pdb_distribution(). "
                "Install it with: pip install matplotlib"
            ) from e

        if self.value is None:
            raise RuntimeError(
                "plot_pdb_distribution() called before compute(). "
                "Call compute(X, family) first."
            )
        if self.statistics is None:
            raise RuntimeError(
                "No null distribution to plot: a fixed lambda was supplied."
            )

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))

        ax = plot_pdb_distribution(
            statistics=self.statistics,
            lambda_value=self.value,
            alpha=self.alpha,
            ax=ax
        )

        return ax
