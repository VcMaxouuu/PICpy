"""Tests for penalty functions and proximal operators."""

import numpy as np
import pytest

from picpy.penalties import L1Penalty, SCADPenalty, MCPPenalty
from picpy.penalties.proximal import soft_thresholding, mcp_thresholding, scad_thresholding


# ---------------------------------------------------------------------------
# soft_thresholding
# ---------------------------------------------------------------------------

class TestSoftThresholding:
    def test_positive_above_threshold(self):
        result = soft_thresholding(np.array([2.0]), threshold=0.5)
        assert np.isclose(result[0], 1.5)

    def test_negative_above_threshold(self):
        result = soft_thresholding(np.array([-2.0]), threshold=0.5)
        assert np.isclose(result[0], -1.5)

    def test_exactly_at_threshold(self):
        result = soft_thresholding(np.array([0.5]), threshold=0.5)
        assert np.isclose(result[0], 0.0)

    def test_below_threshold(self):
        result = soft_thresholding(np.array([0.3]), threshold=0.5)
        assert np.isclose(result[0], 0.0)

    def test_zero_threshold(self):
        v = np.array([1.0, -2.0, 0.0])
        result = soft_thresholding(v, threshold=0.0)
        np.testing.assert_array_almost_equal(result, v)

    def test_vector(self):
        v = np.array([3.0, -1.0, 0.2, 0.0, -0.8])
        result = soft_thresholding(v, threshold=0.5)
        expected = np.array([2.5, -0.5, 0.0, 0.0, -0.3])
        np.testing.assert_array_almost_equal(result, expected)


# ---------------------------------------------------------------------------
# L1Penalty
# ---------------------------------------------------------------------------

class TestL1Penalty:
    def setup_method(self):
        self.pen = L1Penalty()

    def test_evaluate_basic(self):
        beta = np.array([1.0, -2.0, 0.5])
        val = self.pen.evaluate(beta, lambda_=0.1)
        assert np.isclose(val, 0.1 * 3.5)

    def test_evaluate_zero(self):
        assert self.pen.evaluate(np.zeros(5), lambda_=1.0) == 0.0

    def test_prox_is_soft_threshold(self):
        beta = np.array([2.0, -1.0, 0.3])
        result = self.pen.prox(beta, lambda_=0.5, step_size=1.0)
        expected = soft_thresholding(beta, threshold=0.5)
        np.testing.assert_array_almost_equal(result, expected)

    def test_prox_step_size_scaling(self):
        beta = np.array([2.0])
        result = self.pen.prox(beta, lambda_=1.0, step_size=0.5)
        # effective threshold = lambda * step_size = 0.5
        assert np.isclose(result[0], 1.5)

    def test_str(self):
        assert "L1Penalty" in str(self.pen)


# ---------------------------------------------------------------------------
# SCADPenalty
# ---------------------------------------------------------------------------

class TestSCADPenalty:
    def setup_method(self):
        self.pen = SCADPenalty(a=3.7)

    def test_default_a(self):
        assert SCADPenalty().a == 3.7

    def test_evaluate_small_coef(self):
        # |beta| <= lambda → same as L1
        lam = 0.5
        beta = np.array([0.3])
        val = self.pen.evaluate(beta, lambda_=lam)
        assert np.isclose(val, lam * 0.3)

    def test_evaluate_large_coef(self):
        # |beta| > a*lambda → constant (a+1)*lambda^2/2
        a, lam = 3.7, 0.5
        beta = np.array([10.0])
        val = self.pen.evaluate(beta, lambda_=lam)
        assert np.isclose(val, (a + 1) * lam ** 2 / 2.0)

    def test_evaluate_nonnegative(self):
        beta = np.random.default_rng(0).standard_normal(50)
        val = self.pen.evaluate(beta, lambda_=0.3)
        assert val >= 0.0

    def test_prox_returns_finite_array(self):
        beta = np.array([0.5, -1.0, 2.0])
        result = self.pen.prox(beta, lambda_=0.3, step_size=0.5)
        assert result.shape == beta.shape
        assert np.all(np.isfinite(result))

    def test_prox_shrinks_toward_zero(self):
        # SCAD prox should not increase |coef| (for positive lambda)
        beta = np.array([0.5])
        result = self.pen.prox(beta, lambda_=0.3, step_size=1.0)
        assert abs(result[0]) <= abs(beta[0]) + 1e-10

    def test_str(self):
        assert "a=3.7" in str(self.pen)

    def test_param_a(self):
        pen = SCADPenalty(a=2.5)
        assert pen.a == 2.5


# ---------------------------------------------------------------------------
# MCPPenalty
# ---------------------------------------------------------------------------

class TestMCPPenalty:
    def setup_method(self):
        self.pen = MCPPenalty(gamma=3.0)

    def test_default_gamma(self):
        assert MCPPenalty().gamma == 3.0

    def test_evaluate_small_coef(self):
        # |beta| <= gamma*lambda → lambda*|beta| - beta^2/(2*gamma)
        gamma, lam = 3.0, 0.5
        beta_val = 0.4
        beta = np.array([beta_val])
        val = self.pen.evaluate(beta, lambda_=lam)
        expected = lam * beta_val - beta_val ** 2 / (2.0 * gamma)
        assert np.isclose(val, expected)

    def test_evaluate_large_coef(self):
        # |beta| > gamma*lambda → gamma*lambda^2/2
        gamma, lam = 3.0, 0.5
        beta = np.array([10.0])
        val = self.pen.evaluate(beta, lambda_=lam)
        assert np.isclose(val, gamma * lam ** 2 / 2.0)

    def test_evaluate_nonnegative(self):
        beta = np.random.default_rng(1).standard_normal(50)
        val = self.pen.evaluate(beta, lambda_=0.3)
        assert val >= 0.0

    def test_prox_returns_finite_array(self):
        beta = np.array([0.5, -1.0, 2.0])
        result = self.pen.prox(beta, lambda_=0.3, step_size=0.5)
        assert result.shape == beta.shape
        assert np.all(np.isfinite(result))

    def test_prox_shrinks_toward_zero(self):
        # MCP prox should not increase |coef| (for positive lambda)
        beta = np.array([0.5])
        result = self.pen.prox(beta, lambda_=0.3, step_size=1.0)
        assert abs(result[0]) <= abs(beta[0]) + 1e-10

    def test_str(self):
        assert "gamma=3.0" in str(self.pen)
