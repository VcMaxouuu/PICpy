"""Tests for StabilitySelection."""

import numpy as np
import pandas as pd
import pytest

from picpy import LinearRegression
from picpy.utils import StabilitySelection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_data():
    rng = np.random.default_rng(7)
    n, p = 120, 10
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:2] = [3.0, -2.0]
    y = X @ beta + rng.standard_normal(n)
    return X, y


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestStabilitySelectionInit:
    def test_invalid_subsample_fraction_zero(self):
        with pytest.raises(ValueError):
            StabilitySelection(LinearRegression(), subsample_fraction=0.0)

    def test_invalid_subsample_fraction_one(self):
        with pytest.raises(ValueError):
            StabilitySelection(LinearRegression(), subsample_fraction=1.0)

    def test_valid_construction(self):
        ss = StabilitySelection(LinearRegression(), m=50, subsample_fraction=0.5)
        assert ss.m == 50
        assert not ss.is_fitted_

    def test_repr(self):
        ss = StabilitySelection(LinearRegression(), m=30)
        assert "StabilitySelection" in repr(ss)
        assert "m=30" in repr(ss)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

class TestStabilitySelectionFit:
    def test_fit_returns_self(self, small_data):
        X, y = small_data
        ss = StabilitySelection(
            LinearRegression(lambda_method="analytical"),
            m=10,
            random_state=0,
        )
        result = ss.fit(X, y)
        assert result is ss

    def test_is_fitted_after_fit(self, small_data):
        X, y = small_data
        ss = StabilitySelection(
            LinearRegression(lambda_method="analytical"),
            m=10,
            random_state=0,
        )
        ss.fit(X, y)
        assert ss.is_fitted_

    def test_selection_table_shape(self, small_data):
        X, y = small_data
        ss = StabilitySelection(
            LinearRegression(lambda_method="analytical"),
            m=10,
            random_state=0,
        )
        ss.fit(X, y)
        assert ss.selection_table_ is not None
        assert ss.selection_table_.shape == (X.shape[1], 2)

    def test_selection_table_columns(self, small_data):
        X, y = small_data
        ss = StabilitySelection(
            LinearRegression(lambda_method="analytical"),
            m=10,
            random_state=0,
        )
        ss.fit(X, y)
        assert "selection_frequency" in ss.selection_table_.columns
        assert "n_selected" in ss.selection_table_.columns

    def test_selection_frequency_range(self, small_data):
        X, y = small_data
        ss = StabilitySelection(
            LinearRegression(lambda_method="analytical"),
            m=10,
            random_state=0,
        )
        ss.fit(X, y)
        freq = ss.selection_table_["selection_frequency"]
        assert (freq >= 0.0).all() and (freq <= 1.0).all()

    def test_n_selected_consistent_with_freq(self, small_data):
        X, y = small_data
        m = 10
        ss = StabilitySelection(
            LinearRegression(lambda_method="analytical"),
            m=m,
            random_state=0,
        )
        ss.fit(X, y)
        freq = ss.selection_table_["selection_frequency"]
        n_sel = ss.selection_table_["n_selected"]
        np.testing.assert_array_almost_equal(n_sel / m, freq)

    def test_sorted_by_frequency_descending(self, small_data):
        X, y = small_data
        ss = StabilitySelection(
            LinearRegression(lambda_method="analytical"),
            m=10,
            random_state=0,
        )
        ss.fit(X, y)
        freq = ss.selection_table_["selection_frequency"].values
        assert all(freq[i] >= freq[i + 1] for i in range(len(freq) - 1))

    def test_true_signals_have_high_frequency(self, small_data):
        """True signals (features 0, 1) should appear among top-selected."""
        X, y = small_data
        ss = StabilitySelection(
            LinearRegression(lambda_method="analytical"),
            m=30,
            random_state=42,
        )
        ss.fit(X, y)
        top2 = set(ss.selection_table_.index[:3])
        # At least one of the true signals should be in the top 3
        assert len(top2.intersection({0, 1})) >= 1

    def test_reproducibility(self, small_data):
        X, y = small_data
        model = LinearRegression(lambda_method="analytical")
        ss1 = StabilitySelection(model, m=10, random_state=99)
        ss2 = StabilitySelection(model, m=10, random_state=99)
        ss1.fit(X, y)
        ss2.fit(X, y)
        pd.testing.assert_frame_equal(ss1.selection_table_, ss2.selection_table_)

    def test_different_seeds_may_differ(self, small_data):
        X, y = small_data
        model = LinearRegression(lambda_method="analytical")
        ss1 = StabilitySelection(model, m=20, random_state=1)
        ss2 = StabilitySelection(model, m=20, random_state=2)
        ss1.fit(X, y)
        ss2.fit(X, y)
        # Not guaranteed to differ, but usually will for small m
        # Just check both ran without error
        assert ss1.is_fitted_ and ss2.is_fitted_

    def test_n_failed_attribute(self, small_data):
        X, y = small_data
        ss = StabilitySelection(
            LinearRegression(lambda_method="analytical"),
            m=10,
            random_state=0,
        )
        ss.fit(X, y)
        assert isinstance(ss.n_failed_, int)
        assert ss.n_failed_ >= 0

    def test_dataframe_input_preserves_names(self):
        import pandas as pd

        rng = np.random.default_rng(3)
        X_df = pd.DataFrame(
            rng.standard_normal((100, 5)),
            columns=["a", "b", "c", "d", "e"],
        )
        y = rng.standard_normal(100)
        ss = StabilitySelection(
            LinearRegression(lambda_method="analytical"),
            m=5,
            random_state=0,
        )
        ss.fit(X_df, y)
        assert set(ss.selection_table_.index).issubset({"a", "b", "c", "d", "e"})


# ---------------------------------------------------------------------------
# plot (check it does not raise; no display)
# ---------------------------------------------------------------------------

class TestStabilitySelectionPlot:
    def test_plot_before_fit_raises(self):
        ss = StabilitySelection(LinearRegression())
        with pytest.raises(ValueError):
            ss.plot()

    def test_plot_runs(self, small_data):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X, y = small_data
        ss = StabilitySelection(
            LinearRegression(lambda_method="analytical"),
            m=5,
            random_state=0,
        )
        ss.fit(X, y)
        ax = ss.plot(threshold=0.5)
        assert ax is not None
        plt.close("all")
