import numpy as np
import pandas as pd
import pytest

from corr_agent.analysis import align, analyze, strength_label


def daily(values, start="2024-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"), dtype="float64")


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_strong_positive_correlation(rng):
    x = daily(rng.normal(20, 5, 200))
    y = 3 * x + rng.normal(0, 2, 200)
    res = analyze(x, y)
    assert res.pearson_r > 0.95
    assert res.significant
    assert res.strength == "very strong positive"
    assert res.slope == pytest.approx(3, rel=0.05)
    assert res.linear_r2_cv > 0.8


def test_uncorrelated_series(rng):
    res = analyze(daily(rng.normal(size=300)), daily(rng.normal(size=300)))
    assert abs(res.pearson_r) < 0.15
    assert res.strength == "negligible"


def test_negative_correlation(rng):
    x = daily(rng.normal(size=100))
    res = analyze(x, -x + rng.normal(0, 0.3, 100))
    assert res.pearson_r < -0.8
    assert res.strength.endswith("negative")


def test_detects_lag_where_x_leads_y(rng):
    raw = rng.normal(size=203)
    x = daily(raw[3:])
    y = daily(raw[:-3])  # y_t = x_{t-3}
    res = analyze(x, y, max_lag=7)
    assert res.best_lag == 3
    assert res.best_lag_r > 0.95


def test_daily_vs_monthly_alignment(rng):
    days = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    monthly_level = pd.Series(np.arange(48, dtype=float), index=pd.date_range("2020-01-01", periods=48, freq="MS"))
    x = pd.Series(np.repeat(monthly_level.to_numpy(), days.to_period("M").value_counts().sort_index().to_numpy()),
                  index=days) + rng.normal(0, 0.1, len(days))
    res = analyze(x, monthly_level, x_freq="D", y_freq="M")
    assert res.freq == "M"
    assert res.n == 48
    assert res.pearson_r > 0.99
    assert any("averaged per month" in n for n in res.notes)


def test_business_days_inner_join_with_daily(rng):
    x = daily(rng.normal(size=60))
    bdays = x[x.index.dayofweek < 5]
    df, freq = align(x, bdays * 2, "D", "B")
    assert freq in ("D", "B")
    assert len(df) == len(bdays)


def test_too_few_points_raises():
    with pytest.raises(ValueError, match="overlapping"):
        analyze(daily(range(5)), daily(range(5)))


def test_constant_series_raises(rng):
    with pytest.raises(ValueError, match="constant"):
        analyze(daily(np.zeros(30)), daily(rng.normal(size=30)), x_label="snowfall")


def test_forecast(rng):
    x = daily(rng.normal(20, 5, 120))
    y = 2 * x + 10 + rng.normal(0, 0.5, 120)
    future = daily([15, 20, 25], start="2024-06-01")
    res = analyze(x, y, forecast_x=future)
    assert list(res.forecast.index) == list(future.index)
    assert res.forecast["y_pred_linear"].tolist() == pytest.approx([40, 50, 60], abs=1.0)
    assert "forecast" in res.to_llm_dict()


def test_strength_labels():
    assert strength_label(0.05) == "negligible"
    assert strength_label(0.2) == "weak positive"
    assert strength_label(-0.4) == "moderate negative"
    assert strength_label(0.6) == "strong positive"
