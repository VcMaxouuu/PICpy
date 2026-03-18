"""Stability selection visualization for PIC."""

import numpy as np
import pandas as pd

from ._theme import pic_theme, PALETTE


def plot_stability(
    stability_df: pd.DataFrame,
    threshold: float = 0.5,
    ax=None,
    title: str = "Stability Selection",
    max_features: int | None = None,
):
    """Plot selection frequencies from a stability selection run.

    Draws a vertical bar chart where each bar represents one feature.
    Bars are sorted by descending selection frequency. Features with
    zero selection frequency are excluded.

    Parameters
    ----------
    stability_df : pd.DataFrame
        Output of stability_selection. Must contain a
        "selection_frequency" column and a "feature" index or column.
    threshold : float, default=0.5
        Selection frequency threshold for highlighting stable features.
    ax : matplotlib.axes.Axes or None, default=None
        Axes to draw on. A new figure is created when None.
    title : str, default="Stability Selection"
        Plot title.
    max_features : int or None, default=None
        If provided, only the top max_features features are shown.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_stability(). "
            "Install it with: pip install matplotlib"
        ) from exc

    df = stability_df.copy()

    if "feature" in df.columns:
        df = df.set_index("feature")

    df = df[df["selection_frequency"] > 0]

    df = df.sort_values("selection_frequency", ascending=False)

    if max_features is not None:
        df = df.head(max_features)

    freqs = df["selection_frequency"].to_numpy()
    labels = df.index.tolist()

    colors = [
        PALETTE["primary"] if f >= threshold else PALETTE["neutral"]
        for f in freqs
    ]

    with pic_theme():
        if ax is None:
            width = max(6.0, 0.5 * len(labels))
            _, ax = plt.subplots(figsize=(width, 4))

        x = np.arange(len(labels))

        ax.bar(
            x,
            freqs,
            color=colors,
            edgecolor="white",
            linewidth=0.4,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Selection frequency")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title)

    return ax
