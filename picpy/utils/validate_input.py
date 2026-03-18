"""Input validation and preprocessing utilities."""

import numpy as np


def _as_2d_float_array(X: np.ndarray, name: str = "X") -> np.ndarray:
    """Convert *X* to a 2-D float array, reshaping a 1-D input if necessary."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D (n_samples, n_features). Got shape {X.shape}."
        )
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{name} must have positive shape. Got {X.shape}.")
    return X.astype(float)


def _as_1d_array(y: np.ndarray, name: str = "y") -> np.ndarray:
    """Convert *y* to a 1-D float array, squeezing a degenerate 2-D input."""
    y = np.asarray(y)
    if y.ndim == 2 and (y.shape[1] == 1 or y.shape[0] == 1):
        y = y.reshape(-1)
    if y.ndim != 1:
        raise ValueError(f"{name} must be 1-D (n_samples,). Got shape {y.shape}.")
    if y.shape[0] == 0:
        raise ValueError(f"{name} must have positive length. Got {y.shape}.")
    return y.astype(float)


def _check_finite(arr: np.ndarray, name: str) -> None:
    """Raise ValueError if *arr* contains NaN or Inf."""
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf values.")


def check_X(
    X: np.ndarray,
    normalize_X: bool = True,
    X_mean: np.ndarray | None = None,
    X_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Validate and optionally standardize the design matrix.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Design matrix.
    normalize_X : bool, default=True
        Whether to standardize each column to zero mean and unit variance.
        When *X_mean* and *X_std* are supplied they are applied directly
        without recomputing statistics from *X* (used at prediction time).
    X_mean : ndarray of shape (n_features,) or None
        Column means used for standardization. Computed from *X* when None.
    X_std : ndarray of shape (n_features,) or None
        Column standard deviations used for standardization. Computed from
        the centred *X* when None.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Validated (and optionally standardized) design matrix.
    X_mean : ndarray of shape (n_features,) or None
        Column means applied. None when *normalize_X=False*.
    X_std : ndarray of shape (n_features,) or None
        Column standard deviations applied. None when *normalize_X=False*.
    """
    X = _as_2d_float_array(X, "X")
    _check_finite(X, "X")

    if normalize_X:
        if X_mean is None:
            X_mean = X.mean(axis=0)
        X = X - X_mean
        if X_std is None:
            X_std = X.std(axis=0, ddof=0)
        X = X / X_std

    return X, X_mean, X_std


def check_y(
    y: np.ndarray,
    n_samples: int | None = None,
    y_kind: str = "continuous",
) -> np.ndarray:
    """Validate the response vector according to its distributional type.

    Parameters
    ----------
    y : array-like
        Response vector or matrix (survival case).
    n_samples : int or None
        Expected number of observations. When provided, the length of *y*
        is checked against this value.
    y_kind : {'continuous', 'count', 'binary', 'survival'}
        Type of response variable:

        - ``'continuous'`` — real-valued outcomes (Gaussian, Gumbel, …).
        - ``'count'`` — non-negative integers (Poisson).
        - ``'binary'`` — 0/1 labels (Binomial / logistic regression).
        - ``'survival'`` — 2-column matrix ``[time, event]`` (Cox model).

    Returns
    -------
    y : ndarray
        Validated response array.

    Raises
    ------
    ValueError
        If *y* does not satisfy the constraints implied by *y_kind*.
    """
    _check_finite(y, "y")

    if n_samples is not None and y.shape[0] != n_samples:
        raise ValueError(
            f"Length of y ({y.shape[0]}) does not match the number of rows "
            f"in X ({n_samples})."
        )

    if y_kind == "count":
        y = _as_1d_array(y, "y")
        if np.any(y < 0):
            raise ValueError("Count response y must be non-negative.")
        if not np.all(np.isclose(y, np.round(y))):
            raise ValueError("Count response y must contain integer-valued entries.")
        y = y.astype(int)

    elif y_kind == "binary":
        y = _as_1d_array(y, "y")
        uniq = np.unique(y)
        valid = (
            np.array_equal(uniq, [0])
            or np.array_equal(uniq, [1])
            or np.array_equal(uniq, [0, 1])
        )
        if not valid:
            raise ValueError(
                f"Binary response y must contain only values 0 and 1. Found: {uniq}."
            )
        y = y.astype(int)

    elif y_kind == "survival":
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "Survival response y must be 2-D with shape (n_samples, 2) "
                f"containing (time, event) columns. Got shape {y.shape}."
            )
        times, events = y[:, 0], y[:, 1]
        if np.any(times < 0):
            raise ValueError("Survival times must be non-negative.")
        uniq = np.unique(events)
        valid = (
            np.array_equal(uniq, [0])
            or np.array_equal(uniq, [1])
            or np.array_equal(uniq, [0, 1])
        )
        if not valid:
            raise ValueError(
                f"Survival event indicators must contain only values 0 and 1. "
                f"Found: {uniq}."
            )

    elif y_kind != "continuous":
        raise ValueError(
            f"Unknown y_kind='{y_kind}'. "
            "Expected one of: 'continuous', 'count', 'binary', 'survival'."
        )

    return y


def check_Xy(
    X: np.ndarray,
    y: np.ndarray,
    y_kind: str,
    normalize_X: bool = True,
    X_mean: np.ndarray | None = None,
    X_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    """Jointly validate and preprocess the design matrix and response.

    Wraps :func:`check_X` and :func:`check_y`. For survival data both arrays
    are additionally sorted in ascending order of observed time.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Design matrix.
    y : array-like
        Response vector or matrix.
    y_kind : {'continuous', 'count', 'binary', 'survival'}
        Type of response; passed to :func:`check_y`.
    normalize_X : bool, default=True
        Whether to standardize columns of *X*.
    X_mean : ndarray of shape (n_features,) or None
        Pre-computed column means; see :func:`check_X`.
    X_std : ndarray of shape (n_features,) or None
        Pre-computed column standard deviations; see :func:`check_X`.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    X_mean : ndarray or None
    X_std : ndarray or None
    y : ndarray
    """
    X, X_mean, X_std = check_X(X, normalize_X, X_mean=X_mean, X_std=X_std)
    y = check_y(y, n_samples=X.shape[0], y_kind=y_kind)

    if y_kind == "survival":
        order = np.argsort(y[:, 0], kind="mergesort")
        X = X[order]
        y = y[order]

    return X, X_mean, X_std, y
