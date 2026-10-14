"""Plotly figures for an AnalysisResult.

Colors follow the entity, never the chart: X is always blue, Y always orange. Two measures with
different units are shown as stacked panels with a shared time axis, never on a dual y-axis.
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from corr_agent.analysis import AnalysisResult

X_COLOR = "#2a78d6"
Y_COLOR = "#eb6834"
FIT_COLOR = "#8a8984"
LINEAR_COLOR = "#1baf7a"
RF_COLOR = "#6250d6"
MUTED_BAR = "#b8c9e0"


def _axis_title(label: str, units: str) -> str:
    return f"{label} ({units})" if units else label


def _layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def timeseries(res: AnalysisResult) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=(res.x_label, res.y_label))
    fig.add_trace(go.Scatter(x=res.data.index, y=res.data["x"], name=res.x_label, mode="lines",
                             line=dict(color=X_COLOR, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=res.data.index, y=res.data["y"], name=res.y_label, mode="lines",
                             line=dict(color=Y_COLOR, width=2)), row=2, col=1)
    fig.update_yaxes(title_text=res.x_units, row=1, col=1)
    fig.update_yaxes(title_text=res.y_units, row=2, col=1)
    fig.update_annotations(font_size=13)
    return _layout(fig, height=440).update_layout(showlegend=False)


def scatter(res: AnalysisResult) -> go.Figure:
    df = res.data
    xs = [df["x"].min(), df["x"].max()]
    fig = go.Figure([
        go.Scatter(x=df["x"], y=df["y"], mode="markers", name=res.period.capitalize() + "s",
                   marker=dict(color=Y_COLOR, size=8, opacity=0.6, line=dict(width=1, color="white")),
                   text=[d.date().isoformat() for d in df.index],
                   hovertemplate="%{text}<br>x = %{x:.3g}<br>y = %{y:.3g}<extra></extra>"),
        go.Scatter(x=xs, y=[res.slope * v + res.intercept for v in xs], mode="lines",
                   name=f"Linear fit (r = {res.pearson_r:.2f})", line=dict(color=FIT_COLOR, width=2, dash="dash"),
                   hoverinfo="skip"),
    ])
    fig.update_xaxes(title_text=_axis_title(res.x_label, res.x_units))
    fig.update_yaxes(title_text=_axis_title(res.y_label, res.y_units))
    return _layout(fig).update_layout(hovermode="closest")


def lags(res: AnalysisResult) -> go.Figure:
    colors = [X_COLOR if k == res.best_lag else MUTED_BAR for k in res.lags.index]
    fig = go.Figure(go.Bar(x=res.lags.index, y=res.lags.values, marker_color=colors,
                           hovertemplate=f"lag %{{x}} {res.period}(s)<br>r = %{{y:.3f}}<extra></extra>"))
    fig.add_hline(y=0, line_width=1, line_color=FIT_COLOR)
    fig.update_xaxes(title_text=f"Lag in {res.period}s (positive: X leads Y)", dtick=1)
    fig.update_yaxes(title_text="Pearson r", range=[-1, 1])
    return _layout(fig, height=320).update_layout(showlegend=False, hovermode="closest")


def forecast(res: AnalysisResult, history_points: int = 60) -> go.Figure | None:
    if res.forecast is None:
        return None
    hist = res.data["y"].iloc[-history_points:]
    fc = res.forecast
    fig = go.Figure([
        go.Scatter(x=hist.index, y=hist.values, name=f"{res.y_label} (observed)", mode="lines",
                   line=dict(color=Y_COLOR, width=2)),
        go.Scatter(x=fc.index, y=fc["y_pred_linear"], name="Forecast: linear regression", mode="lines+markers",
                   line=dict(color=LINEAR_COLOR, width=2, dash="dash"), marker=dict(size=8)),
        go.Scatter(x=fc.index, y=fc["y_pred_rf"], name="Forecast: random forest", mode="lines+markers",
                   line=dict(color=RF_COLOR, width=2, dash="dot"), marker=dict(size=8)),
    ])
    fig.update_yaxes(title_text=_axis_title(res.y_label, res.y_units))
    return _layout(fig)
