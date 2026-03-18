"""Survival analysis utilities for Cox proportional hazards regression.

Provides post-fit evaluation metrics and curve estimation functions:

- :func:`concordance_index` — Harrell's C-index.
- :func:`cox_partial_log_likelihood` — Breslow negative partial log-likelihood.
- :func:`baseline_functions` — Breslow baseline cumulative hazard and survival.
- :func:`feature_effects_on_survival` — Marginal survival curves varying one feature.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from numpy import ndarray

def concordance_index(
    times: ndarray,
    events: ndarray,
    predictions: ndarray
) -> float:
    """
    Compute Harrell's concordance index (C-index) for survival data.

    Parameters
    ----------
    times : ndarray
        Survival times of shape (n_samples,).
    events : ndarray
        Event indicators of shape (n_samples,) where 1 indicates
        event and 0 indicates censoring.
    predictions : ndarray
        Risk scores from the model of shape (n_samples,).
        Higher values indicate higher risk (shorter survival).

    Returns
    -------
    float
        Concordance index value between 0 and 1, where 1.0 indicates
        perfect concordance, 0.5 indicates random predictions, and
        0.0 indicates perfect anti-concordance.

    Notes
    -----
    Only pairs where the comparison is meaningful are considered.
    Ties in predictions are handled by assigning a score of 0.5.
    """
    event_idx = np.where(events == 1)[0]

    times_i = times[event_idx][:, None]  # Shape: (n_events, 1)
    times_j = times[None, :]  # Shape: (1, n_samples)

    pred_i = predictions[event_idx][:, None]  # Shape: (n_events, 1)
    pred_j = predictions[None, :]  # Shape: (1, n_samples)

    events_j = events[None, :]  # Shape: (1, n_samples)

    # Determine comparable pairs
    # Case 1: Event i occurs before observation j
    check1 = (times_i < times_j)
    # Case 2: Same observation time but j is censored
    check2 = ((times_i == times_j) & (events_j == 0))
    comparable = check1 | check2

    # Remove self-comparisons
    comparable[np.arange(len(event_idx)), event_idx] = False

    n_comparable = comparable.sum()
    if n_comparable == 0:
        return 0.0

    # Compute concordance
    concordant = np.where(pred_i > pred_j, 1.0, 0.0)
    ties = np.where(pred_i == pred_j, 0.5, 0.0)
    score = (concordant + ties)[comparable].sum()

    return score / n_comparable

def cox_partial_log_likelihood(
    times: ndarray,
    events: ndarray,
    predictions: ndarray
) -> float:
    """
    Compute average negative partial log-likelihood for Cox regression.

    Uses the Breslow approximation for handling tied event times.

    Parameters
    ----------
    times : ndarray
        Survival times of shape (n_samples,).
    events : ndarray
        Event indicators of shape (n_samples,) where 1 indicates
        event and 0 indicates censoring.
    predictions : ndarray
        Linear predictions from the model of shape (n_samples,).
        These are the log-hazard ratios.

    Returns
    -------
    float
        Negative partial log-likelihood value (higher is worse).
    """
    # Sort by descending survival times
    order = np.argsort(times)[::-1]
    t_s = times[order]
    p_s = predictions[order]
    e_s = events[order].astype(float)

    exp_p = np.exp(p_s)

    # Group by unique times (descending)
    unique_t = np.unique(t_s)[::-1]
    G = len(unique_t)

    group_uncens = np.zeros(G)        # sum of eta for events at time
    group_exp = np.zeros(G)           # sum of exp(eta) for all at time
    group_events = np.zeros(G, dtype=int)  # number of events at time

    for i, ut in enumerate(unique_t):
        mask = (t_s == ut)
        group_uncens[i] = np.sum(p_s[mask] * e_s[mask])
        group_exp[i] = np.sum(exp_p[mask])
        group_events[i] = int(e_s[mask].sum())

    # Risk set sum of exp(eta) at each time (since sorted descending)
    cum_exp = np.cumsum(group_exp)

    # Breslow log term: d_i * log(R_i)
    log_term = np.zeros(G)
    for i in range(G):
        d_i = group_events[i]
        if d_i == 0:
            continue
        R_i = cum_exp[i]  # sum_{k in R_i} exp(eta_k)
        log_term[i] = d_i * np.log(R_i)

    loss_contrib = group_uncens - log_term
    loss = -np.mean(loss_contrib)
    return loss

def baseline_functions(
    times: ndarray,
    events: ndarray,
    predictions: ndarray
) -> pd.DataFrame:
    """
    Compute baseline cumulative hazard and baseline survival for a Cox model
    using the Breslow estimator.

    Returns a DataFrame indexed by increasing time with:
      - baseline_cumulative_hazard
      - baseline_survival
    """
    times = np.asarray(times)
    events = np.asarray(events).astype(int)
    predictions = np.asarray(predictions)

    exp_pred = np.exp(predictions)

    df = pd.DataFrame({
        "durations": times,
        "exp_pred": exp_pred,
        "events": events
    })

    # Aggregate ties
    df = df.groupby("durations", as_index=False).agg(
        exp_pred=("exp_pred", "sum"),
        events=("events", "sum")
    )

    # Descending so risk set is cumulative sum
    df = df.sort_values("durations", ascending=False).reset_index(drop=True)

    risk_set_sum = df["exp_pred"].cumsum().to_numpy()
    d = df["events"].to_numpy(dtype=float)

    hazard_inc = np.zeros_like(d, dtype=float)
    mask = d > 0
    hazard_inc[mask] = d[mask] / risk_set_sum[mask]

    H0_desc = np.flip(np.cumsum(np.flip(hazard_inc)))              # cumulative in descending order
    H0 = H0_desc[::-1]                           # flip to increasing time
    t = df["durations"].to_numpy()[::-1]

    out = pd.DataFrame(
        {
            "baseline_cumulative_hazard": H0,
            "baseline_survival": np.exp(-H0),
        },
        index=pd.Index(t, name="durations")
    )

    return out

def feature_effects_on_survival(
    model,
    X: ndarray,
    idx: int,
    values: Optional[List] = None,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Build a DataFrame of survival curves when varying one feature and holding others at their mean.

    Returns
    -------
    DataFrame:
      index: time (same as model.baseline_survival_.index)
      columns: values of the feature (or formatted names)
    """
    if not hasattr(model, "predict_survival_function"):
        raise ValueError(
            "model must be a fitted CoxRegression instance with a "
            "'predict_survival_function' method."
        )

    if idx not in model.selected_features_:
        raise ValueError(f"Feature index {idx} is not in the selected features of the model. It's effect is null. Selected features: {model.selected_features_}.")

    if hasattr(X, "columns"):
        feature_name = X.columns[idx]
    else:
        feature_name = f"feature_{idx}"

    X = np.asarray(X)
    col = X[:, idx]
    unique_vals = np.unique(col)

    if values is None:
        if unique_vals.size <= 5:
            values = unique_vals.tolist()
        else:
            values = np.quantile(col, np.linspace(0, 1, 4)).tolist()

    mean_row = X.mean(axis=0, keepdims=True)

    curves = {}
    for v in values:
        Xv = mean_row.copy()
        Xv[0, idx] = v

        surv_df = model.predict_survival_function(Xv)

        if agg == "median":
            curve = surv_df.median(axis=1)
        else:
            curve = surv_df.mean(axis=1)

        col_name = f"{feature_name}={v:.6g}" if isinstance(v, (float, np.floating)) else f"{feature_name}={v}"
        curves[col_name] = curve.to_numpy()

    out = pd.DataFrame(curves, index=model.baseline_survival_.index)
    out.index.name = "time"
    return out
