"""PIC — Pivotal Information Criterion.

High-dimensional sparse regression and classification with automatic
regularisation parameter selection via the Pivotal Detection Boundary (PDB).
"""

__version__ = "1.0.0"

from .models import LinearRegression, LogisticRegression, CoxRegression

__all__ = ["LinearRegression", "LogisticRegression", "CoxRegression"]
