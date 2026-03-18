"""Proximal operators for sparsity-inducing penalties.

These functions implement the closed-form proximal maps used inside the FISTA
solver for the L1, SCAD, and MCP penalties.
"""

import numpy as np


def soft_thresholding(parameter: np.ndarray, threshold: float) -> np.ndarray:
    """Soft-thresholding operator (proximal map of the L1 penalty).

    .. math::

        \\operatorname{prox}_{t\\lambda\\|\\cdot\\|_1}(u) =
        \\operatorname{sign}(u)\\,\\max(|u| - t\\lambda,\\; 0)

    Parameters
    ----------
    parameter : ndarray
        Input vector :math:`u`.
    threshold : float
        Combined regularization-step product :math:`t\\lambda`.

    Returns
    -------
    ndarray
        Thresholded vector.
    """
    return np.sign(parameter) * np.maximum(np.abs(parameter) - threshold, 0.0)


def mcp_thresholding(
    parameter: np.ndarray,
    lambda_: float,
    gamma: float,
    step_size: float,
) -> np.ndarray:
    """Proximal map of the Minimax Concave Penalty (MCP).

    Applies soft-thresholding on the region where the MCP is active and
    leaves the parameter unchanged in the unpenalized flat region.

    Parameters
    ----------
    parameter : ndarray
        Input vector.
    lambda_ : float
        Regularization parameter.
    gamma : float
        MCP concavity parameter (:math:`\\gamma > 1`).
    step_size : float
        FISTA step size (backtracking parameter).

    Returns
    -------
    ndarray
    """
    out = parameter.copy()
    mid = np.abs(parameter) <= lambda_ * gamma
    out[mid] = (
        (gamma / (gamma - step_size))
        * soft_thresholding(parameter[mid], lambda_ * step_size)
    )
    return out


def scad_thresholding(
    parameter: np.ndarray,
    lambda_: float,
    a: float,
    step_size: float,
) -> np.ndarray:
    """Proximal map of the Smoothly Clipped Absolute Deviation (SCAD) penalty.

    Applies a piecewise proximal rule with three regions corresponding to
    the three pieces of the SCAD penalty.

    Parameters
    ----------
    parameter : ndarray
        Input vector.
    lambda_ : float
        Regularization parameter.
    a : float
        SCAD concavity parameter (:math:`a > 2`, typically :math:`a = 3.7`).
    step_size : float
        FISTA step size (backtracking parameter).

    Returns
    -------
    ndarray
    """
    out = parameter.copy()
    abs_param = np.abs(parameter)

    m1 = abs_param <= lambda_ * (1.0 + step_size)
    out[m1] = soft_thresholding(parameter[m1], lambda_ * step_size)

    m2 = (abs_param > lambda_ * (1.0 + step_size)) & (abs_param <= a * lambda_)
    out[m2] = (
        ((a - 1.0) / (a - 1.0 - step_size))
        * soft_thresholding(parameter[m2], (a * lambda_ * step_size) / (a - 1.0))
    )

    return out
