"""Visualization utilities for PIC.

All plot functions use the fixed PIC matplotlib theme, which can also be
applied globally or used as a context manager.

Examples
--------
Apply the theme once for a whole session:

>>> from picpy.utils.visuals import apply_theme
>>> apply_theme()

Or wrap individual plots in a context manager:

>>> from picpy.utils.visuals import pic_theme
>>> with pic_theme():
...     ax = model.pdb_selector.plot_pdb_distribution()
"""

from ._theme import apply_theme, pic_theme, PALETTE, COLOR_CYCLE
from .stability_plot import plot_stability
from .pdb_plot import plot_pdb_distribution
from .cox_plot import plot_cox_summary

__all__ = [
    "apply_theme",
    "pic_theme",
    "PALETTE",
    "COLOR_CYCLE",
    "plot_stability",
    "plot_pdb_distribution",
    "plot_cox_summary",
]
