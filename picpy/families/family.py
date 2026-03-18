"""GLM family distributions for the PIC framework.

Each family encapsulates a loss function, its gradient, a mean-function link
*g*, a variance-stabilizing transform *phi*, and utilities needed by the
PDB regularization selector.
"""

import numpy as np
from scipy.stats import gumbel_r

from .links import Link, IdentityLink, SquareRootLink, ExponentialLink, LogisticLink


class Family:
    """Abstract base class for a GLM family.

    A family bundles three things required by the PIC fitting algorithm:

    1. A **loss** function :math:`\\ell(y, \\hat\\mu)` and its gradient.
    2. A **mean-function link** *g* mapping the linear predictor
       :math:`\\eta = X\\beta + b` to the response scale: :math:`\\hat\\mu = g(\\eta)`.
    3. A **variance-stabilizing transform** *phi* applied to the raw loss
       before differentiation.

    The composite objective minimized by FISTA is therefore

    .. math::

        \\phi\\!\\left(\\ell(y,\\, g(X\\beta + b))\\right) + \\lambda\\, P(\\beta)

    Subclasses must implement :meth:`raw_loss`, :meth:`raw_loss_derivative`,
    and :meth:`generate_y`.

    Attributes
    ----------
    g : Link
        Mean-function link: maps linear predictor → fitted values.
    phi : Link
        Variance-stabilizing transform applied to the raw loss scalar.
    """

    g: Link = IdentityLink()
    phi: Link = IdentityLink()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(phi={self.phi}, g={self.g})"

    def __repr__(self) -> str:
        return self.__str__()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def raw_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Unstabilized loss before applying *phi*.

        Parameters
        ----------
        y_true : ndarray of shape (n_samples,) or (n_samples, n_simu)
            Observed response(s).
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_simu)
            Fitted values on the response scale (i.e. after applying *g*).

        Returns
        -------
        float or ndarray
            Scalar loss (or column-wise loss for batched null simulations).
        """
        raise NotImplementedError

    def raw_loss_derivative(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """Element-wise derivative of the raw loss w.r.t. *y_pred*.

        Parameters
        ----------
        y_true : ndarray
            Observed response.
        y_pred : ndarray
            Fitted values on the response scale.

        Returns
        -------
        ndarray of the same shape as *y_pred*
        """
        raise NotImplementedError

    def generate_y(self, size: tuple) -> np.ndarray:
        """Draw null responses for the PDB Monte Carlo simulation.

        Parameters
        ----------
        size : tuple
            Shape of the output array, typically ``(n_samples, n_simu)``.

        Returns
        -------
        ndarray of shape *size*
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Derived interface (used by FISTA and LambdaPDB)
    # ------------------------------------------------------------------

    def y_kind(self) -> str:
        """String identifier for the response type.

        Returns
        -------
        str
            One of ``'continuous'``, ``'count'``, ``'binary'``,
            ``'survival'``.
        """
        return "continuous"

    def _reset_fit_state(self) -> None:
        """Reset any cached state before a new fit.

        Called by :meth:`~picpy.models.base._BaseRegression.fit` at the start
        of every fitting procedure so that families with cached per-dataset
        quantities (e.g. tie-group metadata in :class:`Cox`) are always
        consistent with the data being fitted.  The default implementation is
        a no-op; stateful families must override this method.
        """

    def variance_scaling_factor(self, n: int) -> float:
        """Return the variance scaling factor c(n).

        Parameters
        ----------
        n : int
            Sample size.

        Returns
        -------
        float
            Scaling factor used in the Gaussian approximation of the null gradient.
        """
        return 1.0

    def starting_intercept(self, y: np.ndarray) -> float:
        """Compute a reasonable initial intercept estimate.

        The default implementation inverts the mean response through *g*:
        :math:`b_0 = g^{-1}(\\bar y)`.

        Parameters
        ----------
        y : ndarray of shape (n_samples,) or (n_samples, n_simu)
            Response vector(s).

        Returns
        -------
        float or ndarray
        """
        return self.g.inverse(np.mean(y, axis=0))

    def evaluate(self, y: np.ndarray, lin_pred: np.ndarray) -> float:
        """Compute the (stabilized) loss at linear predictor *lin_pred*.

        Applies the mean-function link *g* to obtain fitted values, evaluates
        :meth:`raw_loss`, and then applies *phi*:

        .. math::

            \\phi\\!\\bigl(\\ell(y,\\; g(\\eta))\\bigr)

        Parameters
        ----------
        y : ndarray
            Observed response.
        lin_pred : ndarray
            Linear predictor :math:`\\eta = X\\beta + b`.

        Returns
        -------
        float
            Scalar stabilized loss.
        """
        y_pred = self.g(lin_pred)
        raw = self.raw_loss(y_true=y, y_pred=y_pred)
        return self.phi(raw)

    def grad(
        self, X: np.ndarray, y: np.ndarray, lin_pred: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Gradient of the stabilized loss w.r.t. coefficients and intercept.

        Uses the chain rule through *phi*, *raw_loss*, and *g*:

        .. math::

            \\nabla_{\\beta}\\, \\phi(\\ell) =
            \\phi'(\\ell)\\cdot X^{\\top}
            \\left[g'(\\eta) \\odot \\ell'(y, g(\\eta))\\right]

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Design matrix.
        y : ndarray
            Observed response.
        lin_pred : ndarray of shape (n_samples,) or (n_samples, n_simu)
            Linear predictor.

        Returns
        -------
        coef_grad : ndarray of shape (n_features,) or (n_features, n_simu)
            Gradient w.r.t. regression coefficients.
        intercept_grad : float or ndarray of shape (n_simu,)
            Gradient w.r.t. the intercept.
        """
        y_pred = self.g(lin_pred)
        raw = self.raw_loss(y_true=y, y_pred=y_pred)
        phi_prime = self.phi.derivative(raw)
        g_prime = self.g.derivative(lin_pred)
        residual = phi_prime * g_prime * self.raw_loss_derivative(y_true=y, y_pred=y_pred)
        coef_grad = X.T @ residual
        intercept_grad = np.sum(residual, axis=0)
        return coef_grad, intercept_grad


# ---------------------------------------------------------------------------
# Concrete families
# ---------------------------------------------------------------------------

class Gaussian(Family):
    """Gaussian (normal) family with identity link and square-root stabilization.

    Loss:

    .. math::

        \\ell(y, \\hat\\mu) = \\frac{1}{n}\\sum_i (y_i - \\hat\\mu_i)^2

    with :math:`\\phi = \\sqrt{\\cdot}`.

    Examples
    --------
    >>> import numpy as np
    >>> from picpy.families import Gaussian
    >>> fam = Gaussian()
    >>> fam.y_kind()
    'continuous'
    """

    g: Link = IdentityLink()
    phi: Link = SquareRootLink()

    def raw_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        r = y_true - y_pred
        return np.mean(r * r, axis=0)

    def raw_loss_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n = y_true.shape[0]
        return (2.0 / n) * (y_pred - y_true)

    def generate_y(self, size: tuple) -> np.ndarray:
        return np.random.normal(0.0, 1.0, size=size)


class Binomial(Family):
    """Bernoulli family with logistic link for binary classification.

    Loss (variance-stabilized form of the binomial deviance):

    .. math::

        \\ell(y, p) = 2\\,\\mathbb{E}_n\\!\\left[
            y\\sqrt{\\tfrac{1-p}{p}} + (1-y)\\sqrt{\\tfrac{p}{1-p}}
        \\right]

    with :math:`\\phi = \\mathrm{id}` and :math:`g = \\mathrm{logistic}`.

    Examples
    --------
    >>> import numpy as np
    >>> from picpy.families import Binomial
    >>> fam = Binomial()
    >>> fam.y_kind()
    'binary'
    """

    g: Link = LogisticLink()
    phi: Link = IdentityLink()

    def raw_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        sqrt_val = (1.0 - y_pred) / y_pred
        return 2.0 * np.mean(
            y_true * np.sqrt(sqrt_val) + (1.0 - y_true) * np.sqrt(1.0 / sqrt_val),
            axis=0,
        )

    def raw_loss_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n = y_true.shape[0]
        return (1.0 / n) * (y_pred - y_true) / (y_pred * (1.0 - y_pred)) ** 1.5

    def generate_y(self, size: tuple) -> np.ndarray:
        return np.random.binomial(1, 0.5, size=size)

    def y_kind(self) -> str:
        return "binary"


class Poisson(Family):
    """Poisson family with log link for count data.

    Loss (variance-stabilized Poisson deviance):

    .. math::

        \\ell(y, \\hat\\mu) = 2\\,\\mathbb{E}_n\\!\\left[
            \\frac{y}{\\sqrt{\\hat\\mu}} + \\sqrt{\\hat\\mu}
        \\right]

    with :math:`g = \\exp` and :math:`\\phi = \\mathrm{id}`.

    Examples
    --------
    >>> import numpy as np
    >>> from picpy.families import Poisson
    >>> fam = Poisson()
    >>> fam.y_kind()
    'count'
    """

    g: Link = ExponentialLink()
    phi: Link = IdentityLink()

    def raw_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return 2.0 * np.mean(y_true / np.sqrt(y_pred) + np.sqrt(y_pred), axis=0)

    def raw_loss_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n = y_true.shape[0]
        return (1.0 / n) * (y_pred - y_true) / y_pred ** 1.5

    def generate_y(self, size: tuple) -> np.ndarray:
        return np.random.poisson(5.0, size=size)

    def y_kind(self) -> str:
        return "count"


class Exponential(Family):
    """Exponential family with log link for positive continuous outcomes.

    Loss (negative log-likelihood of the exponential distribution):

    .. math::

        \\ell(y, \\hat\\mu) = \\mathbb{E}_n\\!\\left[\\log\\hat\\mu + \\frac{y}{\\hat\\mu}\\right]

    with :math:`g = \\exp` and :math:`\\phi = \\mathrm{id}`.

    Examples
    --------
    >>> import numpy as np
    >>> from picpy.families import Exponential
    >>> fam = Exponential()
    >>> fam.y_kind()
    'continuous'
    """

    g: Link = ExponentialLink()
    phi: Link = IdentityLink()

    def raw_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.log(y_pred) + y_true / y_pred, axis=0)

    def raw_loss_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n = y_true.shape[0]
        return (1.0 / n) * (1.0 / y_pred - y_true / y_pred ** 2)

    def generate_y(self, size: tuple) -> np.ndarray:
        return np.random.exponential(5.0, size=size)


class Gumbel(Family):
    """Gumbel (extreme-value type I) family with identity link.

    The scale parameter :math:`\\sigma` is either fixed by the user or
    estimated from the residuals :math:`y - \\hat\\mu` via MLE at each
    gradient evaluation.

    Loss:

    .. math::

        \\ell(y, \\hat\\mu) = \\log\\sigma +
            \\mathbb{E}\\!\\left[z + e^{-z}\\right], \\quad
            z = \\frac{y - \\hat\\mu}{\\sigma}

    with :math:`g = \\mathrm{id}` and :math:`\\phi = \\exp`.

    Parameters
    ----------
    sigma : float or None, default=None
        Fixed scale parameter.  When None, :math:`\\sigma` is estimated from
        residuals using MLE at each gradient step.

    Examples
    --------
    >>> import numpy as np
    >>> from picpy.families import Gumbel
    >>> fam = Gumbel()
    >>> fam.y_kind()
    'continuous'
    """

    g: Link = IdentityLink()
    phi: Link = ExponentialLink()

    def __init__(self, sigma: float | None = None):
        super().__init__()
        self._fixed_sigma = sigma
        self.sigma: float | None = sigma

    def starting_intercept(self, y: np.ndarray) -> float:
        """Initialize intercept using the Gumbel mean formula."""
        if self._fixed_sigma is not None:
            sigma = self._fixed_sigma
        else:
            sigma = np.sqrt(6.0) / np.pi * np.std(y)
            self.sigma = sigma
        return self.g.inverse(np.mean(y, axis=0) - sigma * np.euler_gamma)

    def variance_scaling_factor(self, n: int) -> float:
        """Return the variance scaling factor c(n) for the Gumbel family."""
        return np.exp(2 * (np.euler_gamma + 1))

    def _get_sigma(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return sigma, re-estimating from residuals if not fixed."""
        if self._fixed_sigma is not None:
            return self._fixed_sigma
        self.sigma = gumbel_r.fit(y_true - y_pred, floc=0.0)[1]
        return self.sigma

    def raw_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        sigma = self._get_sigma(y_true, y_pred)
        z = (y_true - y_pred) / sigma
        return float(np.log(sigma) + np.mean(z + np.exp(-z)))

    def raw_loss_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        sigma = self._get_sigma(y_true, y_pred)
        z = (y_true - y_pred) / sigma
        return (np.exp(-z) - 1.0) / (sigma * y_true.shape[0])

    def generate_y(self, size: tuple) -> np.ndarray:
        return np.random.gumbel(loc=0.0, scale=1.0, size=size)

    def _null_grad_norms(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Pivotal gradient norms for the PDB null distribution (Gumbel).

        Uses the profile-likelihood parameterization to produce
        variance-stabilized gradient norms under the Gumbel null.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Standardized design matrix.
        y : ndarray of shape (n_samples, n_simu)
            Null response draws from :meth:`generate_y`.

        Returns
        -------
        ndarray of shape (n_simu,)
            :math:`\\ell_\\infty`-norm of the gradient for each simulation.
        """
        n = X.shape[0]
        mle = np.apply_along_axis(gumbel_r.fit, 0, y)
        loc_hats = mle[0, :]
        scale_hats = mle[1, :]

        z = (y - loc_hats) / scale_hats
        loss = np.log(scale_hats) + np.mean(z + np.exp(-z), axis=0)
        L = np.exp(loss)

        s = np.exp(-z)
        v = s - 1.0
        grads = ((X.T @ v) / (n * scale_hats)) * L
        return np.linalg.norm(grads, ord=np.inf, axis=0)


class Cox(Family):
    """Cox proportional hazards family for survival outcomes.

    Implements the Breslow partial log-likelihood and its gradient.  The
    response matrix must have two columns: ``(time, event)``, where event is
    1 for observed failures and 0 for censored observations.  Observations
    must be sorted in ascending order of time before use (handled automatically
    by :func:`~picpy.utils.validate_input.check_Xy`).

    Loss (average negative partial log-likelihood, Breslow approximation):

    .. math::

        \\ell = -\\frac{1}{n}\\left[
            \\eta^{\\top} e
            - \\sum_{t: e_t=1} \\log \\sum_{j \\in R_t} e^{\\eta_j}
        \\right]

    where :math:`R_t` is the risk set at time *t* and *e* is the event
    indicator vector.

    Notes
    -----
    No intercept is fitted for the Cox model (``fit_intercept=False`` in
    :class:`~picpy.models.CoxRegression`).

    Examples
    --------
    >>> import numpy as np
    >>> from picpy.families import Cox
    >>> fam = Cox()
    >>> fam.y_kind()
    'survival'
    """

    g: Link = IdentityLink()
    phi: Link = SquareRootLink()

    def __init__(self) -> None:
        super().__init__()
        self._counts: np.ndarray | None = None
        self._starts: np.ndarray | None = None
        self._sum_uncensored: np.ndarray | None = None

    def _reset_fit_state(self) -> None:
        """Clear cached tie-group metadata so the next fit reinitializes from scratch."""
        self._counts = None
        self._starts = None
        self._sum_uncensored = None

    def variance_scaling_factor(self, n: int) -> float:
        """Return the variance scaling factor c(n) for the Cox family."""
        return 1 / (4.0 * np.log(n))

    def _initialize_counts(self, y: np.ndarray) -> None:
        """Pre-compute tie-group metadata from the sorted survival matrix."""
        self._times = y[:, 0]
        self._events = y[:, 1]
        _, self._counts = np.unique(self._times, return_counts=True)
        self._has_ties = np.any(self._counts > 1)
        self._starts = np.r_[0, np.cumsum(self._counts[:-1])]
        self._sum_uncensored = np.add.reduceat(self._events, self._starts)
        self._n = len(self._times)

    def raw_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self._counts is None:
            self._initialize_counts(y_true)

        m = np.max(y_pred)
        exp_shifted = np.exp(y_pred - m)
        risk_sum = np.cumsum(exp_shifted[::-1])[::-1]

        log_risk = np.log(risk_sum) + m
        if not self._has_ties:
            loss = -np.dot(y_pred, self._events) + np.dot(self._events, log_risk)
        else:
            loss = (
                -np.dot(y_pred, self._events)
                + np.dot(self._sum_uncensored, log_risk[self._starts])
            )
        return loss / self._n

    def raw_loss_derivative(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        if self._counts is None:
            self._initialize_counts(y_true)

        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        m = np.max(y_pred)
        exp_shifted = np.exp(y_pred - m)
        risk_sum = np.cumsum(exp_shifted[::-1])[::-1]

        if self._has_ties:
            risk_sum_per_time = risk_sum[self._starts]
            coeff = self._sum_uncensored / risk_sum_per_time
            cum_coeff = np.cumsum(coeff)
            cum_coeff_per_index = np.repeat(cum_coeff, self._counts)
            grad_eta = (-self._events + exp_shifted * cum_coeff_per_index) / self._n
        else:
            coeff = self._events / risk_sum
            cum_coeff = np.cumsum(coeff)
            grad_eta = (-self._events + exp_shifted * cum_coeff) / self._n

        return grad_eta

    def generate_y(self, size: tuple) -> np.ndarray:
        """Draw null event indicators (Bernoulli 1/2) for PDB simulation."""
        return np.random.binomial(1, 0.5, size=size)

    def _null_grad_norms(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Pivotal gradient norms for the PDB null distribution (Cox).

        Uses a permutation-based estimator adapted to the partial likelihood
        structure of the Cox model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Standardized design matrix.
        y : ndarray of shape (n_samples, n_simu)
            Null event indicator draws from :meth:`generate_y`.

        Returns
        -------
        ndarray of shape (n_simu,)
        """
        n = X.shape[0]
        counts = np.arange(1, n + 1, dtype=X.dtype)
        log_counts = np.log(counts)
        events = y

        den = (
            2.0 * np.sqrt(np.sum(events * log_counts[:, None], axis=0)) * np.sqrt(n)
            + 1e-10
        )
        grad_norms = np.empty(events.shape[1], dtype=X.dtype)

        for it in range(events.shape[1]):
            idx = np.random.permutation(n)
            X_perm = X[idx]
            running_mean = np.cumsum(X_perm, axis=0) / counts[:, None]
            X_perm_centred = X_perm - running_mean
            v = X_perm_centred.T @ events[:, it]
            grad_norms[it] = np.linalg.norm(v, ord=np.inf)

        return grad_norms / den

    def y_kind(self) -> str:
        return "survival"
