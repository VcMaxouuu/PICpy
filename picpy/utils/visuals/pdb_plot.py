"""Visualization utilities for PDB regularization selection."""

import numpy as np

from ._theme import pic_theme, PALETTE


def plot_pdb_distribution(
    statistics: np.ndarray,
    lambda_value: float,
    alpha: float,
    ax=None,
    bins: int = 60,
    title: str = "PDB null distribution",
):
    """Plot the Monte Carlo null distribution with the PDB threshold.

    Draws a histogram of the simulated null statistics together with a
    vertical line marking the selected regularization parameter.

    Parameters
    ----------
    statistics : ndarray of shape (n_simu,)
        Simulated null distribution used to determine the PDB threshold.
    lambda_value : float
        Selected PDB regularization parameter.
    alpha : float
        Nominal upper-tail probability used to define the PDB threshold.
    ax : matplotlib.axes.Axes or None, default=None
        Axes on which to draw the plot. If None, a new figure and axes are
        created.
    bins : int, default=60
        Number of histogram bins.
    title : str, default="PDB null distribution"
        Title of the plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing the plot.

    Raises
    ------
    ImportError
        If ``matplotlib`` is not installed.
    ValueError
        If ``statistics`` is empty.

    Examples
    --------
    >>> ax = plot_pdb_distribution(
    ...     statistics=selector.statistics,
    ...     lambda_value=selector.value,
    ...     alpha=selector.alpha,
    ... )
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_pdb_distribution(). "
            "Install it with: pip install matplotlib"
        ) from exc

    statistics = np.asarray(statistics)
    if statistics.size == 0:
        raise ValueError("statistics must contain at least one simulated value.")

    with pic_theme():
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))

        ax.hist(
            statistics,
            bins=bins,
            color=PALETTE["primary"],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.90,
            density=True,
            label="Null distribution",
        )

        ax.axvline(
            lambda_value,
            color=PALETTE["secondary"],
            linewidth=1.8,
            linestyle="--",
            label=rf"$\lambda_{{\mathrm{{PDB}}}} = {lambda_value:.4f}$"
                  rf" ($\alpha = {alpha:.3f}$)",
            zorder=3,
        )

        ax.set_xlabel(r"Null statistic")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()

    return ax
