"""Tests for GLM family implementations."""

import numpy as np
import pytest

from picpy.families import Gaussian, Binomial, Poisson, Exponential, Gumbel, Cox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_Xy(n=100, p=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    return X


# ---------------------------------------------------------------------------
# y_kind
# ---------------------------------------------------------------------------

class TestYKind:
    def test_gaussian(self):
        assert Gaussian().y_kind() == "continuous"

    def test_binomial(self):
        assert Binomial().y_kind() == "binary"

    def test_poisson(self):
        assert Poisson().y_kind() == "count"

    def test_exponential(self):
        assert Exponential().y_kind() == "continuous"

    def test_gumbel(self):
        assert Gumbel().y_kind() == "continuous"

    def test_cox(self):
        assert Cox().y_kind() == "survival"


# ---------------------------------------------------------------------------
# generate_y shapes
# ---------------------------------------------------------------------------

class TestGenerateY:
    def test_gaussian_shape(self):
        fam = Gaussian()
        y = fam.generate_y(size=(50, 200))
        assert y.shape == (50, 200)

    def test_binomial_values(self):
        fam = Binomial()
        y = fam.generate_y(size=(100, 10))
        assert set(np.unique(y)).issubset({0, 1})

    def test_poisson_nonneg(self):
        fam = Poisson()
        y = fam.generate_y(size=(100, 10))
        assert (y >= 0).all()

    def test_exponential_positive(self):
        fam = Exponential()
        y = fam.generate_y(size=(50, 10))
        assert (y > 0).all()


# ---------------------------------------------------------------------------
# grad shape
# ---------------------------------------------------------------------------

class TestGradShape:
    def _check_grad(self, family, y):
        X = _make_Xy(n=y.shape[0] if y.ndim > 1 else len(y), p=10)
        intercept = family.starting_intercept(y)
        coef_grad, intercept_grad = family.grad(X, y, intercept)
        assert coef_grad.shape == (X.shape[1], 1) or coef_grad.shape == X.shape

    def test_gaussian(self):
        fam = Gaussian()
        rng = np.random.default_rng(0)
        y = rng.standard_normal((100, 5))
        X = _make_Xy(n=100, p=10)
        intercept = fam.starting_intercept(y)
        coef_grad, _ = fam.grad(X, y, intercept)
        assert coef_grad.shape[0] == 10

    def test_binomial(self):
        fam = Binomial()
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.5, (100, 5)).astype(float)
        X = _make_Xy(n=100, p=10)
        intercept = fam.starting_intercept(y)
        coef_grad, _ = fam.grad(X, y, intercept)
        assert coef_grad.shape[0] == 10


# ---------------------------------------------------------------------------
# variance_scaling_factor
# ---------------------------------------------------------------------------

class TestVarianceScalingFactor:
    def test_gaussian_returns_1(self):
        assert Gaussian().variance_scaling_factor(100) == 1.0

    def test_binomial_returns_1(self):
        assert Binomial().variance_scaling_factor(100) == 1.0

    def test_poisson_returns_1(self):
        assert Poisson().variance_scaling_factor(100) == 1.0

    def test_exponential_returns_1(self):
        assert Exponential().variance_scaling_factor(100) == 1.0

    def test_cox_smaller_than_1(self):
        # Cox variance scaling = 1/(4*log(n))
        n = 100
        expected = 1.0 / (4.0 * np.log(n))
        assert np.isclose(Cox().variance_scaling_factor(n), expected)


# ---------------------------------------------------------------------------
# Cox _reset_fit_state
# ---------------------------------------------------------------------------

class TestCoxResetFitState:
    def test_reset_clears_cache(self):
        cox = Cox()
        # Manually set cached attributes as if a fit had happened
        cox._counts = np.array([1, 2, 3])
        cox._starts = np.array([0, 1])
        cox._sum_uncensored = 10.0

        cox._reset_fit_state()

        assert cox._counts is None
        assert cox._starts is None
        assert cox._sum_uncensored is None

    def test_reset_on_base_family_is_noop(self):
        fam = Gaussian()
        fam._reset_fit_state()  # Should not raise

    def test_refit_on_different_n(self, survival_data):
        """Cox model should be refittable on datasets of different sizes."""
        from picpy import CoxRegression

        X, y, _ = survival_data
        model = CoxRegression(lambda_method="analytical")

        model.fit(X[:80], y[:80])
        n1 = model.n_samples_in_

        # Refit on different n — must not raise a shape error
        model.fit(X[80:130], y[80:130])
        n2 = model.n_samples_in_

        assert n1 != n2
