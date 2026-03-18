"""Cox regression summary visualisation for PIC models."""

import numpy as np

from ._theme import pic_theme, PALETTE, COLOR_CYCLE


def plot_cox_summary(model, X):
    """Big-picture summary figure for a fitted :class:`~picpy.models.CoxRegression`.

    Produces a multi-panel figure organised in two sections:

    **Top row** — baseline curves estimated from the training data:

    - *Left*: Breslow baseline survival function :math:`S_0(t)`.
    - *Right*: Breslow baseline cumulative hazard function :math:`H_0(t)`.

    **Feature-effect rows** — one panel per selected variable (two per row),
    each showing how the predicted survival curve shifts when that variable is
    varied across five representative values (its empirical quintiles) while
    all other variables are held at their training-set means.

    Parameters
    ----------
    model : CoxRegression
        A fitted :class:`~picpy.models.CoxRegression` instance.  Must have
        :attr:`baseline_survival_` and :attr:`baseline_cumulative_hazard_`
        populated (i.e. :meth:`fit` must have been called).
    X : array-like of shape (n_samples, n_features)
        Training design matrix.  Used to compute the per-feature mean row and
        the representative values for each selected variable.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The summary figure.

    Raises
    ------
    ImportError
        If ``matplotlib`` is not installed.
    ValueError
        If *model* has not been fitted or has no baseline curves.

    Examples
    --------
    >>> from picpy.utils.visuals import plot_cox_summary
    >>> model.fit(X, y)
    >>> fig = plot_cox_summary(model, X)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_cox_summary(). "
            "Install it with: pip install matplotlib"
        ) from exc

    if not model.is_fitted_:
        raise ValueError("model must be fitted before calling plot_cox_summary().")
    if model.baseline_survival_ is None:
        raise ValueError(
            "model.baseline_survival_ is None. "
            "Ensure the model was fitted with fit()."
        )

    from ..survival import feature_effects_on_survival

    X_arr = np.asarray(X, dtype=float)
    sel_idx = model.selected_features_
    n_sel = len(sel_idx)

    # Resolve feature names
    if model.feature_names_in_ is not None:
        feature_names = list(model.feature_names_in_)
    else:
        feature_names = [str(i) for i in range(X_arr.shape[1])]

    # Layout ---------------------------------------------------------------
    n_effect_rows = int(np.ceil(n_sel / 2)) if n_sel > 0 else 0
    n_rows = 1 + n_effect_rows
    n_cols = 2

    with pic_theme():
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(12, 4 * n_rows),
            squeeze=False,
        )
        fig.suptitle("Cox Regression - Summary", fontsize=14, fontweight="bold", y=1.01)

        # ── Row 0: baseline curves ─────────────────────────────────────
        t = model.baseline_survival_.index.to_numpy()

        # Left: baseline survival
        ax = axes[0, 0]
        S0 = model.baseline_survival_["baseline_survival"].to_numpy()
        ax.step(t, S0, where="post", color=PALETTE["primary"], linewidth=1.8)
        ax.set_xlabel("Time")
        ax.set_ylabel(r"$S_0(t)$")
        ax.set_title("Baseline survival function")
        ax.set_ylim(-0.05, 1.05)

        # Right: baseline cumulative hazard
        ax = axes[0, 1]
        H0 = model.baseline_cumulative_hazard_["baseline_cumulative_hazard"].to_numpy()
        ax.step(t, H0, where="post", color=PALETTE["secondary"], linewidth=1.8)
        ax.set_xlabel("Time")
        ax.set_ylabel(r"$H_0(t)$")
        ax.set_title("Baseline cumulative hazard")
        ax.set_ylim(-0.05, H0.max() * 1.05)
        # ── Rows 1+: feature effects ───────────────────────────────────
        for i, feat_idx in enumerate(sel_idx):
            row = 1 + i // 2
            col = i % 2
            ax = axes[row, col]

            feat_name = feature_names[feat_idx]

            curves_df = feature_effects_on_survival(
                model, X_arr, idx=feat_idx
            )
            times = curves_df.index.to_numpy()

            for j, col_name in enumerate(curves_df.columns):
                color = COLOR_CYCLE[j % len(COLOR_CYCLE)]
                ax.step(
                    times,
                    curves_df[col_name].to_numpy(),
                    where="post",
                    color=color,
                    linewidth=1.6,
                    label=col_name,
                )

            ax.set_xlabel("Time")
            ax.set_ylabel("Survival probability")
            ax.set_title(f"Effect of {feat_name}")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=8, loc="upper right")

        # Hide any unused axes in the last effect row
        if n_sel > 0 and n_sel % 2 == 1:
            axes[-1, -1].set_visible(False)

        fig.tight_layout()

    return fig
