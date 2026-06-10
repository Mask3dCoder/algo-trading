#!/usr/bin/env python3
"""Rolling correlation monitor for strategy attribution exports.

Reads daily strategy attribution CSV files and warns when rolling pairwise
strategy-return correlations exceed a configured threshold. The script is
read-only: it does not change allocations, parameters, or strategy code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_LIVE_ATTRIBUTION = Path("reports/strategy_attribution.csv")
DEFAULT_PAPER_ATTRIBUTION = Path("reports/paper_attribution.csv")


def load_attribution(path: str | Path) -> pd.DataFrame:
    """Load attribution rows from CSV."""
    attribution_path = Path(path)
    if not attribution_path.exists():
        return pd.DataFrame(columns=["date", "strategy", "ret"])
    df = pd.read_csv(attribution_path)
    required = {"date", "strategy", "ret"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["date", "strategy", "ret"])
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    df["strategy"] = df["strategy"].astype(str)
    return df.dropna(subset=["date", "strategy", "ret"])


def compute_correlation_alarms(
    attribution: pd.DataFrame,
    window: int = 20,
    threshold: float = 0.8,
) -> pd.DataFrame:
    """Return rolling pairwise correlation alarms by strategy pair."""
    columns = ["date", "strategy_a", "strategy_b", "correlation", "window"]
    if attribution.empty or window <= 1:
        return pd.DataFrame(columns=columns)

    pivot = attribution.pivot_table(
        index="date",
        columns="strategy",
        values="ret",
        aggfunc="mean",
    ).sort_index()
    if len(pivot) < window or len(pivot.columns) < 2:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, float | int | str]] = []
    for end in range(window, len(pivot) + 1):
        window_df = pivot.iloc[end - window : end]
        corr = window_df.corr()
        strategies = list(corr.columns)
        for i, strategy_a in enumerate(strategies):
            for strategy_b in strategies[i + 1 :]:
                value = corr.loc[strategy_a, strategy_b]
                if pd.notna(value) and value >= threshold:
                    rows.append(
                        {
                            "date": str(pivot.index[end - 1]),
                            "strategy_a": strategy_a,
                            "strategy_b": strategy_b,
                            "correlation": float(value),
                            "window": window,
                        }
                    )
    return pd.DataFrame(rows, columns=columns)


def print_alarms(alarms: pd.DataFrame, threshold: float, window: int) -> None:
    """Print correlation alarms or a clean status line."""
    if alarms.empty:
        print(
            f"Correlation monitor: no pair exceeded {threshold:.2f} "
            f"over rolling {window} days."
        )
        return

    print("Correlation alarms:")
    for _, row in alarms.iterrows():
        print(
            f"{row['date']}: {row['strategy_a']} / {row['strategy_b']} "
            f"correlation={float(row['correlation']):.3f} "
            f"window={int(row['window'])}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rolling strategy correlation monitor")
    parser.add_argument("--paper", action="store_true", help="Use paper attribution default")
    parser.add_argument(
        "--attribution",
        type=Path,
        default=None,
        help="Attribution CSV path. Defaults to live/paper exports.",
    )
    parser.add_argument("--window", type=int, default=20, help="Rolling correlation window")
    parser.add_argument("--threshold", type=float, default=0.8, help="Alarm threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attribution_paths = (
        [args.attribution]
        if args.attribution is not None
        else [DEFAULT_PAPER_ATTRIBUTION if args.paper else DEFAULT_LIVE_ATTRIBUTION]
    )
    attribution = pd.concat([load_attribution(path) for path in attribution_paths], ignore_index=True)
    alarms = compute_correlation_alarms(attribution, window=args.window, threshold=args.threshold)
    print_alarms(alarms, threshold=args.threshold, window=args.window)


if __name__ == "__main__":
    main()
