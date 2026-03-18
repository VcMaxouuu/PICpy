"""Tests for LinearRegression, LogisticRegression, and CoxRegression."""

import numpy as np
import pytest

from picpy import LinearRegression, LogisticRegression, CoxRegression
from picpy.families import Poisson
from picpy.penalties import SCADPenalty, MCPPenalty


# ---------------------------------------------------------------------------
# LinearRegression
# ---------------------------------------------------------------------------

class TestLinearRegressionFit:
    def test_fit_returns_self(self, gaussian_data):
        X, y, _ = gaussian_data
        model = LinearRegression(lambda_method="analytical")
        result = model.fit(X, y)
        assert result is model

    def test_is_fitted_after_fit(self, gaussian_data):
        X, y, _ = gaussian_data
        model = LinearRegression(lambda_method="analytical")
        model.fit(X, y)
        assert model.is_fitted_

    def test_coef_shape(self, gaussian_data):
        X, y, _ = gaussian_data
        model = LinearRegression(lambda_method="analytical")
        model.fit(X, y)
        assert model.coef_.shape == (X.shape[1],)

    def test_lambda_positive(self, gaussian_data):
        X, y, _ = gaussian_data
        model = LinearRegression(lambda_method="analytical")
        model.fit(X, y)
        assert model.lambda_ > 0.0

    def test_selects_true_signals(self, gaussian_data):
        """At high SNR, all 5 true signals should be selected."""
        X, y, beta = gaussian_data
        true_signals = set(np.where(beta != 0)[0])
        model = LinearRegression(lambda_method="analytical")
        model.fit(X, y)
        selected = set(model.selected_features_)
        # At minimum, the largest signal should be found
        assert 0 in selected

    def test_n_selected_consistent(self, gaussian_data):
        X, y, _ = gaussian_data
        model = LinearRegression(lambda_method="analytical")
        model.fit(X, y)
        assert model.n_selected_ == len(model.selected_features_)

    def test_score_reasonable(self, gaussian_data):
        X, y, _ = gaussian_data
        model = LinearRegression(lambda_method="analytical")
        model.fit(X, y)
        r2 = model.score(X, y)
        assert r2 > 0.5  # high-SNR data: expect good fit

    def test_predict_shape(self, gaussian_data):
        X, y, _ = gaussian_data
        model = LinearRegression(lambda_method="analytical")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_fixed_lambda(self, gaussian_data):
        X, y, _ = gaussian_data
        model = LinearRegression(lambda_=0.5)
        model.fit(X, y)
        assert model.lambda_ == 0.5

    def test_not_fitted_raises(self):
        model = LinearRegression()
        with pytest.raises(ValueError, match="not fitted"):
            _ = model.coef_table

    def test_coef_table_columns(self, gaussian_data):
        X, y, _ = gaussian_data
        model = LinearRegression(lambda_method="analytical")
        model.fit(X, y)
        df = model.coef_table
        assert "coef" in df.columns
        assert "selected" in df.columns

    def test_refit_option(self, gaussian_data):
        X, y, _ = gaussian_data
        model = LinearRegression(lambda_method="analytical")
        model.fit(X, y, refit=True)
        assert model.is_fitted_

    def test_with_scad_penalty(self, gaussian_data):
        X, y, _ = gaussian_data
        model = LinearRegression(penalty=SCADPenalty(), lambda_method="analytical")
        model.fit(X, y)
        assert model.is_fitted_

    def test_with_mcp_penalty(self, gaussian_data):
        X, y, _ = gaussian_data
        model = LinearRegression(penalty=MCPPenalty(), lambda_method="analytical")
        model.fit(X, y)
        assert model.is_fitted_

    def test_with_poisson_family(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((150, 10))
        beta = np.zeros(10)
        beta[:2] = [1.0, -0.5]
        lam = np.exp(X @ beta)
        y = rng.poisson(lam).astype(float)
        model = LinearRegression(family=Poisson(), lambda_method="analytical")
        model.fit(X, y)
        assert model.is_fitted_

    def test_no_intercept(self, gaussian_data):
        X, y, _ = gaussian_data
        model = LinearRegression(fit_intercept=False, lambda_method="analytical")
        model.fit(X, y)
        assert model.intercept_ is None or model.intercept_ == 0.0

    def test_all_lambda_methods(self, gaussian_data):
        X, y, _ = gaussian_data
        for method in ("mc_exact", "mc_gaussian", "analytical"):
            model = LinearRegression(lambda_n_simu=200, lambda_method=method)
            model.fit(X, y)
            assert model.lambda_ > 0.0

    def test_refit_on_different_n(self, gaussian_data):
        """Model should be refittable on datasets of different sizes."""
        X, y, _ = gaussian_data
        model = LinearRegression(lambda_method="analytical")
        model.fit(X[:100], y[:100])
        model.fit(X[100:200], y[100:200])
        assert model.n_samples_in_ == 100

    def test_summary_runs(self, gaussian_data):
        X, y, _ = gaussian_data
        model = LinearRegression(lambda_method="analytical")
        model.fit(X, y)
        result = model.summary()
        assert isinstance(result, str)
        assert "LinearRegression" in result


# ---------------------------------------------------------------------------
# LogisticRegression
# ---------------------------------------------------------------------------

class TestLogisticRegressionFit:
    def test_fit_returns_self(self, binary_data):
        X, y, _ = binary_data
        model = LogisticRegression(lambda_method="analytical")
        result = model.fit(X, y)
        assert result is model

    def test_predict_binary_values(self, binary_data):
        X, y, _ = binary_data
        model = LogisticRegression(lambda_method="analytical")
        model.fit(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_range(self, binary_data):
        X, y, _ = binary_data
        model = LogisticRegression(lambda_method="analytical")
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (X.shape[0],)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_score_above_chance(self, binary_data):
        X, y, _ = binary_data
        model = LogisticRegression(lambda_method="analytical")
        model.fit(X, y)
        acc = model.score(X, y)
        assert acc > 0.6  # high-SNR binary data

    def test_coef_shape(self, binary_data):
        X, y, _ = binary_data
        model = LogisticRegression(lambda_method="analytical")
        model.fit(X, y)
        assert model.coef_.shape == (X.shape[1],)


# ---------------------------------------------------------------------------
# CoxRegression
# ---------------------------------------------------------------------------

class TestCoxRegressionFit:
    def test_fit_returns_self(self, survival_data):
        X, y, _ = survival_data
        model = CoxRegression(lambda_method="analytical")
        result = model.fit(X, y)
        assert result is model

    def test_coef_shape(self, survival_data):
        X, y, _ = survival_data
        model = CoxRegression(lambda_method="analytical")
        model.fit(X, y)
        assert model.coef_.shape == (X.shape[1],)

    def test_c_index_above_chance(self, survival_data):
        X, y, _ = survival_data
        model = CoxRegression(lambda_method="analytical")
        model.fit(X, y)
        c = model.score(X, y)
        assert c > 0.5

    def test_baseline_survival_fitted(self, survival_data):
        X, y, _ = survival_data
        model = CoxRegression(lambda_method="analytical")
        model.fit(X, y)
        assert hasattr(model, "baseline_survival_")
        assert model.baseline_survival_ is not None

    def test_predict_survival_function_shape(self, survival_data):
        X, y, _ = survival_data
        model = CoxRegression(lambda_method="analytical")
        model.fit(X, y)
        n_test = 5
        surv = model.predict_survival_function(X[:n_test])
        # Returns a 2D array: (n_time_points, n_test)
        assert surv.ndim == 2
        assert surv.shape[1] == n_test

    def test_reset_fit_state_allows_refit(self, survival_data):
        """Cox model must be refittable on data with different n."""
        X, y, _ = survival_data
        model = CoxRegression(lambda_method="analytical")
        model.fit(X[:80], y[:80])
        model.fit(X[80:], y[80:])  # different n — must not raise
        assert model.n_samples_in_ == X.shape[0] - 80
