import numpy as np
from typing import Tuple, Optional

from ..families import Family
from ..penalties import Penalty
from .training_logger import TrainingLogger

def fista(
    X: np.ndarray,
    y: np.ndarray,
    family: Family,
    penalty: Penalty,
    lambda_reg: float,
    fit_intercept: bool = True,
    rel_tol: float = 1e-4,
    step_size_init: float = 1e-2,
    logger: Optional[TrainingLogger] = None,
    coef_init: Optional[np.ndarray] = None,
    intercept_init: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[float], dict]:
    """
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) with Nesterov
    momentum and backtracking line search.

    Minimizes the composite objective :math:`F(\\beta) = f(\\beta) + g(\\beta)`
    where *f* is the (smooth) stabilized family loss and *g* is the
    (non-smooth) penalty.

    At each iteration two proximal candidates are evaluated — one from the
    extrapolated point and one from the current point — and the better of
    the two is accepted.  This safeguard prevents divergence when the
    objective is non-convex.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Design matrix (assumed standardized).
    y : ndarray of shape (n_samples,) or (n_samples, 2)
        Response vector or survival matrix.
    family : Family
        Loss function, link, and variance-stabilizing transform.
    penalty : Penalty
        Regularization penalty providing the proximal operator.
    lambda_reg : float
        Regularization parameter :math:`\\lambda`.
    fit_intercept : bool, default=True
        Whether to update an unpenalized intercept term.
    rel_tol : float, default=1e-4
        Relative change in the objective used as the convergence criterion.
    step_size_init : float, default=1e-2
        Initial step size for gradient updates.
    logger : TrainingLogger or None
        Optional logger for iteration-level diagnostics.
    coef_init : ndarray or None
        Warm-start coefficient vector.  Zeros when None.
    intercept_init : float or None
        Warm-start intercept.  Family-specific initialization when None.

    Returns
    -------
    coef : ndarray of shape (n_features,)
        Converged coefficient vector.
    intercept : float or None
        Converged intercept (None when *fit_intercept=False*).
    info : dict
        Optimization diagnostics:

        - ``'n_iter'`` — number of iterations performed.
        - ``'lambda'`` — regularization parameter used.
        - ``'objective_path'`` — objective value at each iteration.
        - ``'features_path'`` — active feature indices at each iteration.
    """
    # Initialize parameters if not provided
    n_samples, n_features = X.shape
    coef = np.zeros(n_features) if coef_init is None else coef_init.copy()

    if fit_intercept:
        if intercept_init is None:
            intercept = family.starting_intercept(y)
        else:
            intercept = intercept_init
    else:
        intercept = 0.0

    # z_1 = x_1 = x_0
    coef_z = coef.copy()
    intercept_z = intercept

    coef_prev = coef.copy()
    intercept_prev = intercept

    # t_1 = 1, t_0 = 0
    t_prev = 0.0
    t_curr = 1.0

    step_size_y = step_size_init
    step_size_x = step_size_init
    objective_path = []
    features_path = []
    prev_obj = family.evaluate(y, X @ coef + intercept) + penalty.evaluate(coef, lambda_reg)
    iteration = 0

    while True:
        # === 1. Extrapolation step (compute y_k) ===
        coef_y = (
            coef
            + (t_prev / t_curr) * (coef_z - coef)
            + ((t_prev - 1) / t_curr) * (coef - coef_prev)
        )
        intercept_y = (
            intercept
            + (t_prev / t_curr) * (intercept_z - intercept)
            + ((t_prev - 1) / t_curr) * (intercept - intercept_prev)
        )

        # === 2. Compute gradients ===
        linear_pred_y = X @ coef_y + intercept_y
        grad_coef_y, grad_intercept_y = family.grad(X, y, linear_pred_y)

        linear_pred_x = X @ coef + intercept
        grad_coef_x, grad_intercept_x = family.grad(X, y, linear_pred_x)

        # === 3. Two proximal updates ===
        step_size_y, coef_z_next, _ = backtracking_line_search(
            X, y, family, penalty, lambda_reg, coef_y, intercept_y, grad_coef_y, step_size_y
        )
        step_size_x, coef_v_next, _ = backtracking_line_search(
            X, y, family, penalty, lambda_reg, coef, intercept, grad_coef_x, step_size_x
        )

        if fit_intercept:
            intercept_z_next = intercept_line_search(
                X, y, family, coef_z_next, intercept_y, grad_intercept_y, step_size_y
            )
            intercept_v_next = intercept_line_search(
                X, y, family, coef_v_next, intercept, grad_intercept_x, step_size_x
            )
        else:
            intercept_z_next = intercept_v_next = 0.0

        # === 4. Evaluate both objectives ===
        smooth_loss_z = family.evaluate(y, X @ coef_z_next + intercept_z_next)
        smooth_loss_v = family.evaluate(y, X @ coef_v_next + intercept_v_next)

        Fz = smooth_loss_z + penalty.evaluate(coef_z_next, lambda_reg)
        Fv = smooth_loss_v + penalty.evaluate(coef_v_next, lambda_reg)

        # === 5. Choose better candidate ===
        if Fz <= Fv:
            coef_next, intercept_next, F_next, smooth_loss = coef_z_next, intercept_z_next, Fz, smooth_loss_z
        else:
            coef_next, intercept_next, F_next, smooth_loss = coef_v_next, intercept_v_next, Fv, smooth_loss_v

        # === 6. Update momentum term ===
        t_next = (np.sqrt(4 * t_curr**2 + 1) + 1) / 2

        # === 7. Convergence check ===
        if prev_obj < float("inf"):
            rel_change = abs(F_next - prev_obj) / abs(prev_obj)
            if rel_change < rel_tol:
                objective_path.append(F_next)
                features_path.append(np.where(coef_next != 0)[0])
                coef, intercept = coef_next.copy(), intercept_next
                if logger is not None:
                    logger.log_convergence(n_iter=iteration + 1)
                break

        if logger is not None:
            logger.log_iteration(
                iteration=iteration,
                objective=F_next,
                loss=smooth_loss,
                n_active=int(np.sum(coef_next != 0)),
            )

        # === 8. Prepare next iteration ===
        coef_prev, intercept_prev = coef.copy(), intercept
        coef_z, intercept_z = coef_z_next.copy(), intercept_z_next
        coef, intercept = coef_next.copy(), intercept_next

        t_prev, t_curr = t_curr, t_next
        prev_obj = F_next
        objective_path.append(F_next)
        features_path.append(np.where(coef_next != 0)[0])
        iteration += 1


    info = {"n_iter": iteration + 1, "lambda": lambda_reg, "objective_path": objective_path, "features_path": features_path}
    intercept = float(intercept) if fit_intercept else None
    return coef, intercept, info



def intercept_line_search(
    X: np.ndarray,
    y: np.ndarray,
    family: Family,
    coef: np.ndarray,
    intercept: float,
    grad_intercept: float,
    step_size: float = 1.0,
    beta: float = 0.5,
    max_iter: int = 10,
) -> float:
    """
    Simple line search for updating the intercept.

    Tries decreasing steps in the gradient direction and accepts the first
    one that reduces the loss. If no improvement is found, returns the last value.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target values.
    family : Family
        The family distribution object that defines the loss function and its gradient.
    coef : np.ndarray
        Current coefficients.
    intercept : float
        Current intercept value.
    grad_intercept : float
        Gradient of the loss w.r.t. the intercept.
    step_size : float, default=1.0
        Initial step size for the intercept update.
    beta : float, default=0.5
        Step size reduction factor.
    max_iter : int, default=10
        Maximum number of reductions.

    Returns
    -------
    new_intercept : float
        Updated intercept (or last tried value if no improvement found).
    """
    f_init = family.evaluate(y, X @ coef + intercept)

    for _ in range(max_iter):
        intercept_trial = intercept - step_size * grad_intercept
        f_trial = family.evaluate(y, X @ coef + intercept_trial)

        if f_trial < f_init:
            return intercept_trial

        step_size *= beta
    return intercept


def backtracking_line_search(
    X: np.ndarray,
    y: np.ndarray,
    family: Family,
    penalty: Penalty,
    lambda_reg: float,
    coef: np.ndarray,
    intercept: float,
    gradient: np.ndarray,
    step_size: float = 1.0,
    beta: float = 0.5,
    max_iter: int = 10
) -> Tuple[float, np.ndarray, float]:
    """
    Backtracking line search for finding appropriate step size.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target values.
    family : Family
        The family distribution object that defines the loss function and its gradient.
    penalty : Penalty
        Penalty function object.
    lambda_reg : float
        L1 regularization parameter.
    coef : np.ndarray
        Current coefficients.
    intercept : float
        Current intercept.
    gradient : np.ndarray
        Current gradient.
    step_size : float, default=1.0
        Initial step size.
    beta : float, default=0.5
        Step size reduction factor.
    max_iter : int, default=10
        Maximum number of backtracking iterations.

    Returns
    -------
    step_size : float
        Found step size.
    coef_new : np.ndarray
        Updated coefficients after applying proximal operator.
    intercept_new : float
        Updated intercept.
    """
    linear_pred = X @ coef + intercept
    f_init = family.evaluate(y, linear_pred)
    coef_init = coef.copy()

    for _ in range(max_iter):
        coef_trial = penalty.prox(coef - step_size * gradient, lambda_reg, step_size)
        linear_pred_trial = X @ coef_trial + intercept
        lhs = family.evaluate(y, linear_pred_trial)

        rhs = f_init + np.dot(coef_trial - coef_init, gradient) + (1 / (2 * step_size)) * np.linalg.norm(coef_trial - coef_init)**2

        if lhs < rhs:
            return step_size, coef_trial, intercept

        step_size *= beta

        if step_size < 1e-10:
            step_size = 1e-4
            break
    return step_size, coef_init, intercept
