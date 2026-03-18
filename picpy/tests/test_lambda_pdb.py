"""Tests for the LambdaPDB regularization parameter selector."""

import warnings

import numpy as np
import pytest

from picpy.lambda_pdb import LambdaPDB
from picpy.families import Gaussian, Gumbel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _standardized_X(n=200, p=15, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
    return X


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestLambdaPDBInit:
    def test_valid_methods(self):
        for m in ("mc_exact", "mc_gaussian", "analytical"):
            sel = LambdaPDB(method=m)
            assert sel.method == m

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method must be one of"):
            LambdaPDB(method="invalid")

    def test_fixed_lambda_bypasses_computation(self):
        sel = LambdaPDB(lambda_=0.5)
        X = _standardized_X()
        val = sel.compute(X, Gaussian())
        assert val == 0.5
        assert sel.value == 0.5

    def test_default_attributes(self):
        sel = LambdaPDB()
        assert sel.value is None
        assert sel.statistics is None
        assert sel.alpha == 0.05
        assert sel.n_simu == 5000


# ---------------------------------------------------------------------------
# mc_exact
# ---------------------------------------------------------------------------

class TestMCExact:
    def test_returns_positive_float(self):
        X = _standardized_X()
        sel = LambdaPDB(n_simu=200, method="mc_exact")
        val = sel.compute(X, Gaussian())
        assert isinstance(val, float)
        assert val > 0.0

    def test_statistics_shape(self):
        X = _standardized_X()
        sel = LambdaPDB(n_simu=300, method="mc_exact")
        sel.compute(X, Gaussian())
        assert sel.statistics is not None
        assert sel.statistics.shape == (300,)

    def test_lambda_is_quantile(self):
        X = _standardized_X()
        sel = LambdaPDB(n_simu=500, alpha=0.05, method="mc_exact")
        val = sel.compute(X, Gaussian())
        expected = np.quantile(sel.statistics, 0.95)
        assert np.isclose(val, expected)


# ---------------------------------------------------------------------------
# mc_gaussian
# ---------------------------------------------------------------------------

class TestMCGaussian:
    def test_returns_positive_float(self):
        X = _standardized_X()
        sel = LambdaPDB(n_simu=200, method="mc_gaussian")
        val = sel.compute(X, Gaussian())
        assert isinstance(val, float)
        assert val > 0.0

    def test_statistics_shape(self):
        X = _standardized_X()
        sel = LambdaPDB(n_simu=300, method="mc_gaussian")
        sel.compute(X, Gaussian())
        assert sel.statistics is not None
        assert sel.statistics.shape == (300,)

    def test_lambda_is_quantile(self):
        X = _standardized_X()
        sel = LambdaPDB(n_simu=500, alpha=0.05, method="mc_gaussian")
        val = sel.compute(X, Gaussian())
        expected = np.quantile(sel.statistics, 0.95)
        assert np.isclose(val, expected)


# ---------------------------------------------------------------------------
# analytical
# ---------------------------------------------------------------------------

class TestAnalytical:
    def test_returns_positive_float(self):
        X = _standardized_X()
        sel = LambdaPDB(method="analytical")
        val = sel.compute(X, Gaussian())
        assert isinstance(val, float)
        assert val > 0.0

    def test_no_statistics(self):
        X = _standardized_X()
        sel = LambdaPDB(method="analytical")
        sel.compute(X, Gaussian())
        assert sel.statistics is None

    def test_formula(self):
        from scipy.stats import norm

        n, p = 200, 15
        X = _standardized_X(n=n, p=p)
        alpha = 0.05
        sel = LambdaPDB(alpha=alpha, method="analytical")
        val = sel.compute(X, Gaussian())
        # c=1 for Gaussian; formula: Phi^-1(1 - alpha/(2p)) / sqrt(n)
        expected = norm.ppf(1.0 - alpha / (2.0 * p)) / np.sqrt(n)
        assert np.isclose(val, expected)

    def test_larger_p_gives_larger_lambda(self):
        """Bonferroni bound should increase with more predictors."""
        n = 200
        X_small = _standardized_X(n=n, p=5)
        X_large = _standardized_X(n=n, p=50)

        sel_small = LambdaPDB(method="analytical")
        sel_large = LambdaPDB(method="analytical")

        val_small = sel_small.compute(X_small, Gaussian())
        val_large = sel_large.compute(X_large, Gaussian())

        assert val_large > val_small


# ---------------------------------------------------------------------------
# Gumbel / Cox fallback warning
# ---------------------------------------------------------------------------

class TestGumbelMCGaussian:
    def test_gumbel_mc_gaussian_runs(self):
        """mc_gaussian should complete for Gumbel (no special pivot for it)."""
        X = _standardized_X()
        sel = LambdaPDB(n_simu=100, method="mc_gaussian")
        val = sel.compute(X, Gumbel())
        assert isinstance(val, float)
        assert val > 0.0


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_before_compute_raises(self):
        sel = LambdaPDB()
        with pytest.raises(RuntimeError):
            sel.summary()

    def test_summary_returns_string(self):
        X = _standardized_X()
        sel = LambdaPDB(n_simu=200, method="mc_exact")
        sel.compute(X, Gaussian())
        result = sel.summary()
        assert isinstance(result, str)
        assert "lambda" in result.lower()
