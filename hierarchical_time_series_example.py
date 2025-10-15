"""Example of hierarchical time series forecasting in Python.

This script creates a synthetic hierarchical time series consisting of an overall total,
regions, and stores. Each series is forecast individually with a seasonal naive method and
reconciled bottom-up to ensure the hierarchy sums.

Dependencies:
    pip install pandas numpy matplotlib

Usage:
    python hierarchical_time_series_example.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HierarchyNode:
    """Represents a node in a hierarchical time series."""

    name: str
    children: Tuple[str, ...]

    def is_leaf(self) -> bool:
        return len(self.children) == 0


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def simulate_store_series(index: pd.DatetimeIndex, seed: int) -> pd.Series:
    """Simulate seasonal daily demand for a single store."""

    periods = len(index)
    rng = np.random.default_rng(seed)
    trend = np.linspace(20, 40, periods)
    weekly_seasonality = 10 + 5 * np.sin(np.linspace(0, 3 * np.pi, periods))
    noise = rng.normal(scale=3.0, size=periods)
    return pd.Series(trend + weekly_seasonality + noise, index=index)


def build_hierarchy() -> Dict[str, HierarchyNode]:
    """Return the node structure for the hierarchy.

    total
      ├── north
      │     ├── north_store_1
      │     └── north_store_2
      └── south
            ├── south_store_1
            └── south_store_2
    """

    return {
        "total": HierarchyNode(name="total", children=("north", "south")),
        "north": HierarchyNode(name="north", children=("north_store_1", "north_store_2")),
        "south": HierarchyNode(name="south", children=("south_store_1", "south_store_2")),
        "north_store_1": HierarchyNode(name="north_store_1", children=()),
        "north_store_2": HierarchyNode(name="north_store_2", children=()),
        "south_store_1": HierarchyNode(name="south_store_1", children=()),
        "south_store_2": HierarchyNode(name="south_store_2", children=()),
    }


def simulate_hierarchical_data(periods: int = 90) -> pd.DataFrame:
    """Simulate a hierarchical time series with four stores and two regions."""

    hierarchy = build_hierarchy()
    date_index = pd.date_range("2023-01-01", periods=periods, freq="D")

    series = {}
    for idx, leaf in enumerate([node for node in hierarchy if hierarchy[node].is_leaf()]):
        series[leaf] = simulate_store_series(index=date_index, seed=idx)

    df = pd.DataFrame(series)
    df["north"] = df[["north_store_1", "north_store_2"]].sum(axis=1)
    df["south"] = df[["south_store_1", "south_store_2"]].sum(axis=1)
    df["total"] = df[["north", "south"]].sum(axis=1)
    return df[hierarchy.keys()]


# ---------------------------------------------------------------------------
# Forecasting utilities
# ---------------------------------------------------------------------------

def forecast_series(series: pd.Series, horizon: int, seasonal_periods: int = 7) -> pd.Series:
    """Seasonal naive forecast that repeats the last full season."""

    if len(series) < seasonal_periods:
        raise ValueError("Series length must be at least one seasonal period")

    last_season = series.iloc[-seasonal_periods:]
    repeats = int(np.ceil(horizon / seasonal_periods))
    forecast_values = np.tile(last_season.values, repeats)[:horizon]
    forecast_index = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.Series(forecast_values, index=forecast_index)


def bottom_up_reconciliation(
    leaf_forecasts: Dict[str, pd.Series],
    hierarchy: Dict[str, HierarchyNode],
) -> Dict[str, pd.Series]:
    """Aggregate leaf forecasts to internal nodes."""

    reconciled = dict(leaf_forecasts)

    def aggregate(node: str) -> pd.Series:
        node_info = hierarchy[node]
        if node_info.is_leaf():
            return reconciled[node]

        children_series = [aggregate(child) for child in node_info.children]
        reconciled[node] = pd.concat(children_series, axis=1).sum(axis=1)
        return reconciled[node]

    aggregate("total")
    return reconciled


def train_test_split(series: pd.DataFrame, test_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data frame into training and test periods."""

    return series.iloc[:-test_size], series.iloc[-test_size:]


# ---------------------------------------------------------------------------
# Main example
# ---------------------------------------------------------------------------

def main() -> None:
    hierarchy = build_hierarchy()
    data = simulate_hierarchical_data(periods=120)
    train, test = train_test_split(data, test_size=14)

    leaf_nodes = [name for name, node in hierarchy.items() if node.is_leaf()]
    leaf_forecasts: Dict[str, pd.Series] = {}
    for leaf in leaf_nodes:
        leaf_forecasts[leaf] = forecast_series(train[leaf], horizon=len(test), seasonal_periods=7)

    reconciled_forecasts = bottom_up_reconciliation(leaf_forecasts, hierarchy)

    print("Forecasts (first 5 rows):")
    print(pd.DataFrame({node: reconciled_forecasts[node] for node in hierarchy}).head())

    # Plot actual vs. forecast for the total series
    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train["total"], label="Train Total")
    plt.plot(test.index, test["total"], label="Test Total")
    plt.plot(
        reconciled_forecasts["total"].index,
        reconciled_forecasts["total"],
        label="Bottom-Up Forecast",
    )
    plt.title("Hierarchical Time Series Forecasting (Total)")
    plt.legend()
    plt.tight_layout()
    output_path = "hierarchical_total_forecast.png"
    plt.savefig(output_path)
    print(f"Saved total forecast plot to {output_path}")


if __name__ == "__main__":
    main()
