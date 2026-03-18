"""Link functions for the PIC GLM family system.

Each :class:`Link` maps a linear predictor :math:`\\eta` to the response
scale and provides its derivative and inverse.  They are used as the
mean-function link *g* and the variance-stabilizing transform *phi* inside
:class:`~picpy.families.Family` subclasses.
"""

import numpy as np


class Link:
    """Abstract base class for a link function.

    A link :math:`g` operates on the linear predictor :math:`\\eta`:

    - ``g(eta)`` — forward map: linear predictor → response scale.
    - ``g.derivative(eta)`` — derivative :math:`g'(\\eta)`.
    - ``g.inverse(mu)`` — inverse map: response scale → linear predictor.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the link function."""
        raise NotImplementedError

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the derivative of the link function."""
        raise NotImplementedError

    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Apply the inverse of the link function."""
        raise NotImplementedError

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__str__()


class IdentityLink(Link):
    """Identity link: :math:`g(\\eta) = \\eta`.

    Used by :class:`~picpy.families.Gaussian`, :class:`~picpy.families.Gumbel`,
    and :class:`~picpy.families.Cox`.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> float:
        return 1.0

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x


class SquareRootLink(Link):
    """Square-root link: :math:`g(\\eta) = \\sqrt{\\eta}`.

    Used as the variance-stabilizing *phi* for the Gaussian and Cox families.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 0.5 / np.sqrt(x)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x ** 2


class SquareDiv4Link(Link):
    """Square-divided-by-4 link: :math:`g(\\eta) = \\eta^2 / 4`.

    Satisfies :math:`g^{-1}(\\mu) = 2\\sqrt{\\mu}`.  Used in an alternative
    Poisson parameterization (not active by default).
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.any(x < 0):
            return np.ones_like(x) * np.inf
        return x ** 2 / 4.0

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return x / 2.0

    def inverse(self, x: np.ndarray) -> np.ndarray:
        if np.any(x < 0):
            raise ValueError("SquareDiv4Link: inverse is undefined for negative inputs.")
        return 2.0 * np.sqrt(x)


class ExponentialLink(Link):
    """Exponential (log) link: :math:`g(\\eta) = e^{\\eta}`.

    Standard log link for the Poisson and Exponential families.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        if np.any(x <= 0):
            raise ValueError(
                "ExponentialLink: inverse (log) is undefined for non-positive inputs."
            )
        return np.log(x)


class LogisticLink(Link):
    """Logistic (sigmoid) link: :math:`g(\\eta) = (1 + e^{-\\eta})^{-1}`.

    Standard link for the Binomial family.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        p = self(x)
        return p * (1.0 - p)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        if np.any((x <= 0) | (x >= 1)):
            raise ValueError(
                "LogisticLink: inverse (logit) is undefined outside the open interval (0, 1)."
            )
        return np.log(x / (1.0 - x))
