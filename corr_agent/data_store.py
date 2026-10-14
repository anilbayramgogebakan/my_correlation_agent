"""Per-session store for fetched time series and analysis results.

Tools put data here and hand the LLM only a short `dataset_id` plus summary
statistics, so raw numbers never round-trip through the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# Frequency codes used across the project: D=daily, B=business-daily, W=weekly,
# M=monthly, Q=quarterly, A=annual.
FREQUENCIES = ("D", "B", "W", "M", "Q", "A")


@dataclass
class Dataset:
    id: str
    label: str
    source: str
    series: pd.Series
    freq: str = "D"
    units: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        s = self.series
        return {
            "dataset_id": self.id,
            "label": self.label,
            "source": self.source,
            "frequency": self.freq,
            "units": self.units,
            "n_points": int(s.size),
            "start": s.index.min().date().isoformat() if s.size else None,
            "end": s.index.max().date().isoformat() if s.size else None,
            "mean": _round(s.mean()),
            "min": _round(s.min()),
            "max": _round(s.max()),
            **{k: v for k, v in self.meta.items() if k in ("location", "article", "series_id", "note")},
        }


def _round(v: float) -> float | None:
    return None if pd.isna(v) else round(float(v), 3)


class DatasetStore:
    def __init__(self) -> None:
        self._datasets: dict[str, Dataset] = {}
        self.analyses: list[Any] = []  # AnalysisResult objects, in creation order

    def add(self, *, label: str, source: str, series: pd.Series, freq: str = "D",
            units: str = "", meta: dict[str, Any] | None = None) -> Dataset:
        if freq not in FREQUENCIES:
            raise ValueError(f"Unknown frequency {freq!r}")
        series = series.dropna().astype(float).sort_index()
        series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
        series.name = label
        ds = Dataset(id=f"ds{len(self._datasets) + 1}", label=label, source=source,
                     series=series, freq=freq, units=units, meta=meta or {})
        self._datasets[ds.id] = ds
        return ds

    def get(self, dataset_id: str) -> Dataset:
        try:
            return self._datasets[dataset_id.strip()]
        except KeyError:
            available = ", ".join(f"{d.id} ({d.label})" for d in self._datasets.values()) or "none"
            raise KeyError(f"Unknown dataset_id {dataset_id!r}. Available: {available}") from None

    def list(self) -> list[Dataset]:
        return list(self._datasets.values())

    def __len__(self) -> int:
        return len(self._datasets)
