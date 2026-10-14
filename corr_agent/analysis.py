"""Correlation analysis between two time series: alignment, correlation, lag scan, regression, forecast.

Pure pandas/scipy/sklearn — no LLM or HTTP — so it is deterministic and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

MIN_POINTS = 10

# Coarseness order and pandas resample rules for the target frequency.
FREQ_RANK = {"D": 0, "B": 0, "W": 1, "M": 2, "Q": 3, "A": 4}
RESAMPLE_RULE = {"W": "W-SUN", "M": "MS", "Q": "QS", "A": "YS"}
PERIOD_NAME = {"D": "day", "B": "day", "W": "week", "M": "month", "Q": "quarter", "A": "year"}


@dataclass
class AnalysisResult:
    x_label: str
    y_label: str
    x_units: str
    y_units: str
    freq: str
    data: pd.DataFrame  # columns: x, y (aligned)
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    lags: pd.Series  # index: lag in periods (positive = x leads y), values: Pearson r
    best_lag: int
    best_lag_r: float
    slope: float
    intercept: float
    linear_r2_cv: float | None
    rf_r2_cv: float | None
    strength: str
    significant: bool
    forecast: pd.DataFrame | None = None  # columns: x, y_pred_linear, y_pred_rf
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.data)

    @property
    def period(self) -> str:
        return PERIOD_NAME[self.freq]

    def to_llm_dict(self) -> dict[str, Any]:
        """Compact, JSON-safe summary for the LLM."""
        out: dict[str, Any] = {
            "x": self.x_label,
            "y": self.y_label,
            "aligned_frequency": self.period,
            "n_points": self.n,
            "period_covered": f"{self.data.index.min().date()} to {self.data.index.max().date()}",
            "pearson_r": round(self.pearson_r, 3),
            "pearson_p_value": _fmt_p(self.pearson_p),
            "spearman_rho": round(self.spearman_rho, 3),
            "spearman_p_value": _fmt_p(self.spearman_p),
            "strength": self.strength,
            "statistically_significant_at_0.05": self.significant,
            "best_lag": {"lag_periods": self.best_lag, "r": round(self.best_lag_r, 3),
                         "meaning": _lag_meaning(self.best_lag, self.period)},
            "linear_fit": f"y ≈ {self.slope:.4g} * x + {self.intercept:.4g}",
            "linear_regression_r2_out_of_sample": _r(self.linear_r2_cv),
            "random_forest_r2_out_of_sample": _r(self.rf_r2_cv),
        }
        if self.forecast is not None:
            out["forecast"] = [
                {"date": d.date().isoformat(), "x": round(row.x, 2),
                 "y_pred_linear": round(row.y_pred_linear, 1), "y_pred_random_forest": round(row.y_pred_rf, 1)}
                for d, row in self.forecast.iterrows()
            ]
        if self.notes:
            out["notes"] = self.notes
        return out


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 3)


def _fmt_p(p: float) -> float | str:
    return "< 0.0001" if p < 1e-4 else round(p, 4)


def _lag_meaning(lag: int, period: str) -> str:
    if lag == 0:
        return "strongest relation on the same " + period
    if lag > 0:
        return f"x leads y by {lag} {period}(s)"
    return f"y leads x by {-lag} {period}(s)"


def strength_label(r: float) -> str:
    a = abs(r)
    if a < 0.1:
        return "negligible"
    label = "weak" if a < 0.3 else "moderate" if a < 0.5 else "strong" if a < 0.7 else "very strong"
    return f"{label} {'positive' if r > 0 else 'negative'}"


def coarser_freq(a: str, b: str) -> str:
    return a if FREQ_RANK[a] >= FREQ_RANK[b] else b


def to_freq(series: pd.Series, freq: str) -> pd.Series:
    """Resample to a coarser frequency using the period mean (works for counts and levels alike)."""
    rule = RESAMPLE_RULE.get(freq)
    if rule is None:
        return series
    return series.resample(rule).mean().dropna()


def align(x: pd.Series, y: pd.Series, x_freq: str, y_freq: str) -> tuple[pd.DataFrame, str]:
    """Bring both series to the coarser frequency and inner-join them on date."""
    freq = coarser_freq(x_freq, y_freq)
    df = pd.concat({"x": to_freq(x, freq), "y": to_freq(y, freq)}, axis=1, join="inner").dropna()
    return df, freq


def lag_scan(df: pd.DataFrame, max_lag: int) -> pd.Series:
    """Pearson r of y_t against x_{t-k} for k in [-max_lag, max_lag] (positive k: x leads y)."""
    out = {}
    for k in range(-max_lag, max_lag + 1):
        pair = pd.concat([df["x"].shift(k), df["y"]], axis=1).dropna()
        if len(pair) >= MIN_POINTS and pair.iloc[:, 0].std() > 0:
            out[k] = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    return pd.Series(out, dtype="float64")


def _cv_r2(model: Any, X: np.ndarray, y: np.ndarray) -> float | None:
    n_splits = max(2, min(5, len(y) // 10))
    try:
        scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=n_splits), scoring="r2")
    except ValueError:
        return None
    return float(np.mean(scores))


def analyze(x: pd.Series, y: pd.Series, *, x_freq: str = "D", y_freq: str = "D",
            x_label: str = "x", y_label: str = "y", x_units: str = "", y_units: str = "",
            max_lag: int = 7, forecast_x: pd.Series | None = None) -> AnalysisResult:
    df, freq = align(x, y, x_freq, y_freq)
    notes: list[str] = []
    if len(df) < MIN_POINTS:
        raise ValueError(
            f"Only {len(df)} overlapping {PERIOD_NAME[freq]}s between the two series (need ≥ {MIN_POINTS}). "
            "Fetch both series over the same, longer time window."
        )
    for col, label in (("x", x_label), ("y", y_label)):
        if df[col].std() == 0:
            raise ValueError(f"{label} is constant over the period (e.g. no snowfall at all), "
                             "so a correlation cannot be computed. Try another variable or period.")
    if FREQ_RANK[x_freq] != FREQ_RANK[y_freq]:
        notes.append(f"Series had different frequencies; both were averaged per {PERIOD_NAME[freq]}.")

    pearson = stats.pearsonr(df["x"], df["y"])
    spearman = stats.spearmanr(df["x"], df["y"])

    lags = lag_scan(df, max_lag=min(max_lag, len(df) // 3))
    best_lag = int(lags.abs().idxmax())
    if best_lag != 0 and abs(lags[best_lag]) - abs(lags.get(0, 0.0)) < 0.05:
        best_lag = 0  # prefer the simple same-period explanation unless a lag is clearly better

    X, Y = df[["x"]].to_numpy(), df["y"].to_numpy()
    linear = LinearRegression().fit(X, Y)
    forest = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=10, random_state=0)

    result = AnalysisResult(
        x_label=x_label, y_label=y_label, x_units=x_units, y_units=y_units, freq=freq, data=df,
        pearson_r=float(pearson.statistic), pearson_p=float(pearson.pvalue),
        spearman_rho=float(spearman.statistic), spearman_p=float(spearman.pvalue),
        lags=lags, best_lag=best_lag, best_lag_r=float(lags[best_lag]),
        slope=float(linear.coef_[0]), intercept=float(linear.intercept_),
        linear_r2_cv=_cv_r2(LinearRegression(), X, Y), rf_r2_cv=_cv_r2(forest, X, Y),
        strength=strength_label(float(spearman.statistic)),
        significant=bool(min(pearson.pvalue, spearman.pvalue) < 0.05),
        notes=notes,
    )

    if forecast_x is not None:
        fx = to_freq(forecast_x.dropna(), freq)
        if fx.empty:
            notes.append("Forecast input was empty; no forecast produced.")
        else:
            forest.fit(X, Y)
            FX = fx.to_numpy().reshape(-1, 1)
            result.forecast = pd.DataFrame(
                {"x": fx.to_numpy(), "y_pred_linear": linear.predict(FX), "y_pred_rf": forest.predict(FX)},
                index=fx.index,
            )
            lo, hi = df["x"].min(), df["x"].max()
            if (fx < lo).any() or (fx > hi).any():
                notes.append("Some forecast inputs lie outside the historical range of x; "
                             "predictions there are extrapolations (Random Forest caps at seen values).")
    return result
