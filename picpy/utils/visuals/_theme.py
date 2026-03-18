"""Matplotlib theme for PIC visualizations using SciencePlots."""

from contextlib import contextmanager

import matplotlib.pyplot as plt

PALETTE = {
    "primary":   "#2E6BA8",
    "secondary": "#C0392B",
    "accent":    "#27AE60",
    "neutral":   "#7F8C8D",
    "highlight": "#F39C12",
}

COLOR_CYCLE = [
    PALETTE["primary"],
    PALETTE["secondary"],
    PALETTE["accent"],
    PALETTE["highlight"],
    PALETTE["neutral"],
    "#8E44AD",
    "#16A085",
    "#D35400",
]


def _use_scienceplots():
    try:
        import scienceplots
    except ImportError as exc:
        raise ImportError(
            "SciencePlots is required for PIC visualizations.\n"
            "Install it with: pip install SciencePlots"
        ) from exc

def apply_theme() -> None:
    """Apply the PIC matplotlib theme globally using SciencePlots.

    This applies the 'science' style with optional enhancements and
    overrides the color cycle with the PIC palette.

    Examples
    --------
    >>> from picpy.utils.visuals import apply_theme
    >>> apply_theme()
    """
    _use_scienceplots()

    # Use science style + no LaTeX dependency by default
    plt.style.use(["science", "no-latex"])
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=COLOR_CYCLE)

    # Add margins so curves don't stick to axes
    plt.rcParams["axes.ymargin"] = 0.05
    plt.rcParams["axes.xmargin"] = 0.02


@contextmanager
def pic_theme():
    """Context manager that temporarily applies the PIC theme.

    Uses SciencePlots ('science' style) with PIC color overrides.

    Examples
    --------
    >>> from picpy.utils.visuals import pic_theme
    >>> with pic_theme():
    ...     plot_stability(df)
    """
    _use_scienceplots()

    with plt.rc_context():
        plt.style.use(["science", "no-latex"])
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=COLOR_CYCLE)
        plt.rcParams["axes.ymargin"] = 0.05
        plt.rcParams["axes.xmargin"] = 0.02
        yield
