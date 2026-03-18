"""Sparsity-inducing penalty functions for the PIC framework.

All penalties are approximately equal to :math:`\\lambda\\|\\beta\\|_1` near
the origin and are used both for evaluation and for defining proximal operators
inside the FISTA solver.
"""

import numpy as np
from .proximal import soft_thresholding, mcp_thresholding, scad_thresholding


class Penalty:
    """Abstract base class for a regularization penalty.

    Subclasses must implement :meth:`evaluate` and :meth:`prox`.
    """

    def __str__(self) -> str:
        params = self._param_str()
        return f"{self.__class__.__name__}({params})" if params else self.__class__.__name__

    def __repr__(self) -> str:
        return self.__str__()

    def _param_str(self) -> str:
        """Return a string representation of penalty-specific parameters."""
        return ""

    def evaluate(self, parameter: np.ndarray, lambda_: float) -> float:
        """Evaluate the penalty at *parameter*.

        Parameters
        ----------
        parameter : ndarray
            Coefficient vector :math:`\\beta`.
        lambda_ : float
            Regularization parameter.

        Returns
        -------
        float
            Penalty value.
        """
        raise NotImplementedError

    def prox(
        self, parameter: np.ndarray, lambda_: float, step_size: float
    ) -> np.ndarray:
        """Apply the proximal operator of the penalty.

        Solves :math:`\\arg\\min_u \\{t\\lambda P(u) + \\tfrac{1}{2}\\|u - v\\|^2\\}`
        where *step_size* is :math:`t`.

        Parameters
        ----------
        parameter : ndarray
            Input vector :math:`v`.
        lambda_ : float
            Regularization parameter.
        step_size : float
            Step size :math:`t`.

        Returns
        -------
        ndarray
            Result of the proximal map.
        """
        raise NotImplementedError


class L1Penalty(Penalty):
    """Lasso (L1) penalty: :math:`P(\\beta) = \\lambda\\|\\beta\\|_1`.

    The proximal operator is soft-thresholding.

    Examples
    --------
    >>> import numpy as np
    >>> from picpy.penalties import L1Penalty
    >>> pen = L1Penalty()
    >>> pen.evaluate(np.array([1.0, -2.0, 0.5]), lambda_=0.1)
    0.35
    """

    def evaluate(self, parameter: np.ndarray, lambda_: float) -> float:
        return float(lambda_ * np.sum(np.abs(parameter)))

    def prox(
        self, parameter: np.ndarray, lambda_: float, step_size: float
    ) -> np.ndarray:
        return soft_thresholding(parameter, lambda_ * step_size)


class SCADPenalty(Penalty):
    """Smoothly Clipped Absolute Deviation (SCAD) penalty.

    Equals the L1 penalty near the origin and is constant beyond
    :math:`a\\lambda`, producing approximate unbiasedness for large
    coefficients.

    Parameters
    ----------
    a : float, default=3.7
        Concavity parameter (:math:`a > 2`).  The default value of 3.7
        is recommended by Fan and Li (2001).

    References
    ----------
    .. [1] Fan, J. and Li, R. (2001). Variable selection via nonconcave
       penalized likelihood and its oracle properties. *JASA*, 96(456),
       1348–1360.

    Examples
    --------
    >>> import numpy as np
    >>> from picpy.penalties import SCADPenalty
    >>> pen = SCADPenalty(a=3.7)
    >>> pen.evaluate(np.array([0.5, 0.0, -2.0]), lambda_=0.5)
    """

    def __init__(self, a: float = 3.7):
        self.a = a

    def _param_str(self) -> str:
        return f"a={self.a}"

    def evaluate(self, parameter: np.ndarray, lambda_: float) -> float:
        a = self.a
        abs_param = np.abs(parameter)
        p = np.zeros_like(abs_param, dtype=float)

        m1 = abs_param <= lambda_
        m2 = (abs_param > lambda_) & (abs_param <= a * lambda_)
        m3 = abs_param > a * lambda_

        p[m1] = lambda_ * abs_param[m1]
        p[m2] = (
            2.0 * a * lambda_ * abs_param[m2]
            - abs_param[m2] ** 2
            - lambda_ ** 2
        ) / (2.0 * (a - 1.0))
        p[m3] = (a + 1.0) * lambda_ ** 2 / 2.0

        return float(np.sum(p))

    def prox(
        self, parameter: np.ndarray, lambda_: float, step_size: float
    ) -> np.ndarray:
        return scad_thresholding(parameter, lambda_, self.a, step_size)


class MCPPenalty(Penalty):
    """Minimax Concave Penalty (MCP).

    Reduces the L1 bias by linearly relaxing the penalty beyond
    :math:`\\gamma\\lambda`, yielding exact selection consistency under
    milder conditions than SCAD.

    Parameters
    ----------
    gamma : float, default=3.0
        Concavity parameter (:math:`\\gamma > 1`).

    References
    ----------
    .. [1] Zhang, C.-H. (2010). Nearly unbiased variable selection under
       minimax concave penalty. *Annals of Statistics*, 38(2), 894–942.

    Examples
    --------
    >>> import numpy as np
    >>> from picpy.penalties import MCPPenalty
    >>> pen = MCPPenalty(gamma=3.0)
    >>> pen.evaluate(np.array([0.5, 0.0, -2.0]), lambda_=0.5)
    """

    def __init__(self, gamma: float = 3.0):
        self.gamma = gamma

    def _param_str(self) -> str:
        return f"gamma={self.gamma}"

    def evaluate(self, parameter: np.ndarray, lambda_: float) -> float:
        gamma = self.gamma
        abs_param = np.abs(parameter)
        p = np.zeros_like(abs_param, dtype=float)

        m1 = abs_param <= lambda_ * gamma
        m2 = abs_param > lambda_ * gamma

        p[m1] = lambda_ * abs_param[m1] - abs_param[m1] ** 2 / (2.0 * gamma)
        p[m2] = gamma * lambda_ ** 2 / 2.0

        return float(p.sum())

    def prox(
        self, parameter: np.ndarray, lambda_: float, step_size: float
    ) -> np.ndarray:
        return mcp_thresholding(parameter, lambda_, self.gamma, step_size)
