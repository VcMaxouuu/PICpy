"""Stability selection for PIC regression models."""

from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd


class StabilitySelection:
    """Estimate variable selection stability via bootstrap subsampling.

    Fits a PIC model on *m* random subsamples of size
    ``floor(n * subsample_fraction)`` (drawn **without** replacement) and
    records which features are selected in each subsample.  The selection
    frequency for each feature — the proportion of subsamples in which its
    coefficient is non-zero — is stored in :attr:`selection_table_` after
    calling :meth:`fit`.

    Features with high selection frequency are considered stably selected.
    A common threshold is 0.5 (Meinshausen & Bühlmann, 2010); more
    conservative analyses use 0.6–0.8.

    Parameters
    ----------
    model : _BaseRegression
        A PIC model instance used as a template.  A deep copy is created
        before each subsample fit so the original is never modified.
    m : int, default=100
        Number of bootstrap subsamples.
    subsample_fraction : float, default=0.5
        Fraction of *n* observations to draw per subsample (without
        replacement).  Values in ``(0, 1)`` are supported; 0.5 is the
        standard choice from the stability selection literature.
    random_state : int, numpy.random.Generator, or None, default=None
        Seed or generator for reproducible subsampling.

    Attributes
    ----------
    selection_table_ : pd.DataFrame
        One row per feature, sorted by descending selection frequency.
        Set after calling :meth:`fit`.  Columns:

        - ``selection_frequency`` — fraction of subsamples (in [0, 1]) in
          which the feature was selected.
        - ``n_selected`` — absolute count (out of *m*).

        The index is named ``"feature"`` and contains column names when *X*
        is a DataFrame, or integer indices otherwise.
    n_failed_ : int
        Number of subsample fits that raised an exception and were skipped.
        Set after calling :meth:`fit`.
    is_fitted_ : bool
        True after :meth:`fit` has been called.

    References
    ----------
    .. [1] Meinshausen, N. and Bühlmann, P. (2010).
       "Stability selection."
       *Journal of the Royal Statistical Society: Series B*, 72(4), 417–473.

    Examples
    --------
    >>> import numpy as np
    >>> from picpy import LinearRegression
    >>> from picpy.utils import StabilitySelection
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 20))
    >>> beta = np.zeros(20); beta[:3] = [2.0, -1.5, 1.0]
    >>> y = X @ beta + rng.standard_normal(200)
    >>> ss = StabilitySelection(LinearRegression(), m=100, random_state=42)
    >>> ss.fit(X, y)
    StabilitySelection(m=100, subsample_fraction=0.5)
    >>> ss.selection_table_.head()
    >>> ss.plot(threshold=0.6)
    """

    def __init__(
        self,
        model,
        m: int = 100,
        subsample_fraction: float = 0.5,
        random_state: int | np.random.Generator | None = None,
    ):
        if not (0.0 < subsample_fraction < 1.0):
            raise ValueError(
                f"subsample_fraction must be in (0, 1); got {subsample_fraction}."
            )
        self.model = model
        self.m = m
        self.subsample_fraction = subsample_fraction
        self.random_state = random_state

        self.selection_table_: Optional[pd.DataFrame] = None
        self.n_failed_: int = 0
        self.is_fitted_: bool = False

    def __repr__(self) -> str:
        return (
            f"StabilitySelection(m={self.m}, "
            f"subsample_fraction={self.subsample_fraction})"
        )

    # ------------------------------------------------------------------
    # Main method
    # ------------------------------------------------------------------

    def fit(self, X, y) -> "StabilitySelection":
        """Run stability selection on *X* and *y*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Full design matrix.  If a :class:`pandas.DataFrame`, column names
            are preserved in :attr:`selection_table_`.
        y : array-like of shape (n_samples,) or (n_samples, 2)
            Response vector (or survival matrix for Cox models).

        Returns
        -------
        self : StabilitySelection
            Fitted instance with :attr:`selection_table_` populated.
        """
        rng = np.random.default_rng(self.random_state)

        if hasattr(X, "columns"):
            feature_names = list(X.columns)
            X_arr = np.asarray(X, dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
            feature_names = list(range(X_arr.shape[1]))

        y_arr = np.asarray(y, dtype=float)
        n, p = X_arr.shape
        subsample_size = max(1, int(np.floor(n * self.subsample_fraction)))

        selection_counts = np.zeros(p, dtype=int)
        n_failed = 0

        for _ in range(self.m):
            idx = rng.choice(n, size=subsample_size, replace=False)
            X_sub = X_arr[idx]
            y_sub = y_arr[idx] if y_arr.ndim == 1 else y_arr[idx, :]

            model_copy = deepcopy(self.model)
            try:
                model_copy.fit(X_sub, y_sub)
                selection_counts += (model_copy.coef_ != 0).astype(int)
            except Exception:
                n_failed += 1

        self.selection_table_ = pd.DataFrame(
            {
                "selection_frequency": selection_counts / self.m,
                "n_selected": selection_counts,
            },
            index=pd.Index(feature_names, name="feature"),
        ).sort_values("selection_frequency", ascending=False)

        self.n_failed_ = n_failed
        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(
        self,
        threshold: float = 0.5,
        ax=None,
        title: str = "Stability Selection",
        max_features: int | None = None,
    ):
        """Plot selection frequencies as a bar chart.

        Delegates to :func:`~picpy.utils.visuals.plot_stability`.

        Parameters
        ----------
        threshold : float, default=0.5
            Selection frequency threshold drawn as a reference line.
            Bars at or above it are highlighted.
        ax : matplotlib.axes.Axes or None, default=None
            Axes to draw on.  A new figure is created when None.
        title : str, default="Stability Selection"
            Plot title.
        max_features : int or None, default=None
            If provided, only the top *max_features* features are shown.

        Returns
        -------
        ax : matplotlib.axes.Axes

        Raises
        ------
        ValueError
            If :meth:`fit` has not been called yet.
        """
        self._check_is_fitted()
        from .visuals import plot_stability
        return plot_stability(
            self.selection_table_,
            threshold=threshold,
            ax=ax,
            title=title,
            max_features=max_features,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise ValueError(
                "This StabilitySelection instance has not been fitted yet. "
                "Call fit(X, y) first."
            )
