"""Pre-registered cross-sectional factor experiment (protocol-locked).

Implements ``docs/cross_sectional_factor_protocol.md`` exactly as locked on
2026-08-04 (pre-registration checkpoint git commit ``ed52bae``). This is the
ONLY experiment the freeze permits in the equity daily-bar space, and it is
deliberately not wired to any strategy registry: the three factor variants
are computed directly from fixed definitions, so the harness cannot
accidentally evaluate anything else "by convention".

Design points fixed by the protocol:
* Variants: ``momentum``, ``low_vol``, ``composite`` (exactly 3).
* Long-only top quintile, equal-weighted, fully invested, monthly rebalance
  (21 trading days), no leverage, no shorting.
* Costs: 20 bps per unit of one-way turnover (``0.5 * sum|dw|``) plus
  ``$0.005``/share commission. No financing.
* Gate (Section 8): OOS Sharpe 90% block-bootstrap CI lower bound >= 0.5
  (margin >= 0.6); DSR p <= 0.05 (margin <= 0.04); net-of-cost total return
  > SPY total return (margin: annualized net return beats SPY by >= 1.5pp);
  all must hold or the variant is dead.

Recorded methodology decisions (transparency, not amendments):
* Factor inputs use split-adjusted closes (raw unadjusted bars are preserved;
  splits are removed so a 20:1 price gap cannot fake a -95% momentum print).
  Backward split adjustment preserves price ratios inside any lookback that
  ends before the ex-date, so using the full-window split schedule introduces
  no lookahead into factor values at formation date T.
* Strategy returns are total returns: split-adjusted price returns plus cash-
  credited dividends on ex-dates (pit source). The triage panel is
  total-return-adjusted, so its close returns already include dividends.
* Sharpe uses the codebase convention rf=4% annual (geometrically
  decompounded daily); DSR uses per-observation Sharpe (correct units).
* Walk-forward is anchored with fixed annual test folds over the OOS daily
  return series; the strategy has no estimated parameters, so folds are a
  time-segment robustness check, reported per-fold and pooled.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantcore.data.pit.market.store import ParquetPitStore
from quantcore.research.bootstrap import bootstrap_confidence_interval
from quantcore.research.dsr_published import dsr_components
from quantcore.risk.metrics import TRADING_DAYS

# --------------------------------------------------------------------------
# Frozen protocol constants (Sections 5-8). Do not edit.
# --------------------------------------------------------------------------

VARIANTS: tuple[str, ...] = ("momentum", "low_vol", "composite")
assert VARIANTS == ("momentum", "low_vol", "composite")
assert len(VARIANTS) == 3

LOOKBACK_MOM_FAST_DAYS = 21   # skip the most recent month
LOOKBACK_MOM_SLOW_DAYS = 273  # 12-month cumulative return window start
VOL_WINDOW_DAYS = 126         # trailing 6-month realized volatility
REBAL_DAYS = 21               # monthly rebalance cadence
TOP_QUANTILE = 0.20           # top quintile
COST_BPS_TURNOVER = 20.0      # one-way cost per unit of turnover
COMMISSION_PER_SHARE = 0.005  # dollars
RF_ANNUAL = 0.04              # codebase Sharpe convention (annual)
MIN_PRICE = 2.0               # liquidity filter, as of T
MIN_MARKET_CAP = 1.0e9        # liquidity filter, as of T (proxy, see below)
BURN_IN_MONTHS = 18           # first 18 months reserved as burn-in
N_BOOTSTRAP = 2000
BLOCK_LEN = 63
CI_LEVEL = 0.90

GATE_RAW = {
    "oos_sharpe_ci_lower": 0.5,
    "dsr_p": 0.05,
    "beat_spy": True,
}
GATE_MARGIN = {
    "oos_sharpe_ci_lower": 0.6,
    "dsr_p": 0.04,
    "net_annual_excess_pp": 1.5,
}

PROTOCOL_DOC = "docs/cross_sectional_factor_protocol.md"
PROTOCOL_LOCK_COMMIT = "ed52bae"
RESULTS_PATH = "reports/cross_sectional_factor_results.json"


@dataclass
class ExperimentConfig:
    """Runtime configuration for one experiment run."""

    source: str = "pit"  # "pit" | "triage"
    pit_dir: str | None = None
    triage_csv: str | None = None
    spy_csv: str | None = None
    start: str | None = None
    end: str | None = None
    out: str | None = None
    n_bootstrap: int = N_BOOTSTRAP
    block_len: int = BLOCK_LEN
    triage: bool = False
    seed: int = 42


@dataclass
class VariantResult:
    """One variant's OOS metrics and gate evaluation."""

    variant: str
    oos_returns: np.ndarray
    oos_sharpe: float
    oos_sharpe_ci_lower: float
    oos_sharpe_ci_upper: float
    dsr_p: float
    net_total_return: float
    net_annual_return: float
    spy_total_return: float | None
    annual_excess_pp: float | None
    avg_turnover: float
    trade_count: int
    per_year_sharpe: dict[str, float]
    per_fold_sharpe: list[dict[str, Any]]
    n_oos_days: int
    raw_pass: bool
    margin_pass: bool
    verdict: str
    criteria: dict[str, dict[str, Any]] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def _read_triage(csv_path: str | Path, start: str, end: str) -> pd.DataFrame:
    """Load the long-format triage panel into a wide close matrix."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df[(df["timestamp"] >= pd.Timestamp(start)) & (df["timestamp"] <= pd.Timestamp(end))]
    close = df.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
    close = close.sort_index()
    close = close.loc[:, close.notna().any()]  # drop empty columns
    return close


def _read_spy_fixture(csv_path: str | Path) -> pd.DataFrame:
    """Load the committed Yahoo SPY fixture (used for the triage benchmark)."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    return df[["open", "high", "low", "close", "adj_close"]]


def load_panel(cfg: ExperimentConfig) -> dict[str, Any]:
    """Return a normalized panel dict shared by both sources.

    Keys: ``close`` (wide raw/unadjusted close matrix), ``dates``,
    ``symbols``, ``corp_actions``, ``universe``, ``source``, ``window``,
    ``provenance``.
    """
    start = cfg.start or "2019-01-01"
    end = cfg.end or "2026-08-01"

    if cfg.source == "triage":
        if not cfg.triage_csv or not Path(cfg.triage_csv).exists():
            raise FileNotFoundError(f"triage csv not found: {cfg.triage_csv}")
        close = _read_triage(cfg.triage_csv, start, end)
        return {
            "close": close,
            "dates": close.index,
            "symbols": list(close.columns),
            "corp_actions": pd.DataFrame(),
            "universe": pd.DataFrame(),
            "source": "triage",
            "window": {"start": start, "end": end},
            "provenance": (
                "data/sp500_pit_universe_2018_2024.csv - TOTAL-RETURN-ADJUSTED, "
                "survivorship-reduced, approximate membership. TRIAGE ONLY, "
                "excluded from the gate."
            ),
        }

    if cfg.source != "pit":
        raise ValueError(f"unknown source {cfg.source!r}")

    pit_dir = cfg.pit_dir or str(Path.cwd() / "data" / "pit" / "market")
    store = ParquetPitStore(pit_dir)

    # Universe: latest snapshot (survivorship-free; delisted tail retained).
    snapshots = store.list_shards("universe")
    if not snapshots:
        raise RuntimeError(
            "PIT store has no universe snapshot. Run `bin/polygon_ingest.py universe` first."
        )
    universe = store.read("universe", shards=[snapshots[-1]])
    universe["delisted_utc"] = pd.to_datetime(universe["delisted_utc"], utc=True, errors="coerce")

    # Symbols = universe symbols, plus SPY if bars exist (ETF benchmark).
    symbols = sorted(universe["symbol"].dropna().astype(str).unique().tolist())
    bars = store.read("equity_bars", shards=symbols + ["SPY"])
    if bars.empty:
        raise RuntimeError("PIT store has no equity bars. Run `bin/polygon_ingest.py backfill` first.")
    bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
    bars = bars[(bars["ts"] >= pd.Timestamp(start, tz="UTC")) & (bars["ts"] <= pd.Timestamp(end, tz="UTC"))]
    close = bars.pivot_table(index="ts", columns="symbol", values="close", aggfunc="last")
    close = close.sort_index().tz_localize(None)
    close = close.loc[:, close.notna().any()]

    corp_actions = store.read("corp_actions", shards=["all"])
    if not corp_actions.empty:
        corp_actions["effective_date"] = pd.to_datetime(corp_actions["effective_date"], errors="coerce")
        corp_actions["announced_at"] = pd.to_datetime(corp_actions["announced_at"], utc=True, errors="coerce")

    return {
        "close": close,
        "dates": close.index,
        "symbols": list(close.columns),
        "corp_actions": corp_actions,
        "universe": universe,
        "source": "pit",
        "window": {"start": start, "end": end},
        "provenance": "Massive/Polygon PIT store (raw unadjusted bars + corporate-actions table).",
    }


# --------------------------------------------------------------------------
# Price adjustment
# --------------------------------------------------------------------------

def _split_adjust_factors(
    close: pd.DataFrame,
    corp_actions: pd.DataFrame,
) -> pd.DataFrame:
    """Return cumulative backward split-adjustment factor per symbol/date.

    A split with ratio ``R`` effective at date ``D`` divides all prices
    before ``D`` by ``R``. Factor(t) = prod(1/R for splits with D > t).
    Ratios inside any pre-D lookback are preserved, so using the full
    schedule introduces no lookahead into factor values at formation time.
    """
    factors = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    if corp_actions is None or corp_actions.empty:
        return factors
    splits = corp_actions[corp_actions["action_type"] == "split"]
    if splits.empty:
        return factors
    for _, row in splits.iterrows():
        sym = str(row.get("symbol"))
        if sym not in factors.columns:
            continue
        eff = pd.Timestamp(row.get("effective_date"))
        if pd.isna(eff):
            continue
        ratio = float(row.get("value") or 0.0)
        if ratio <= 0.0:
            continue
        mask = factors.index < eff.normalize()
        factors.loc[mask, sym] *= 1.0 / ratio
    return factors


# --------------------------------------------------------------------------
# Factor computation
# --------------------------------------------------------------------------

def compute_factors(
    close_adj: pd.DataFrame,
    formation_idx: int,
) -> dict[str, pd.Series]:
    """Return momentum / low-vol / composite factor series at ``formation_idx``.

    ``close_adj`` is the split-adjusted close matrix. Factor values are only
    defined for symbols with the full required history (no forward fill).
    """
    c = close_adj
    i = formation_idx

    mom = pd.Series(float("nan"), index=c.columns, dtype=float)
    if i - LOOKBACK_MOM_SLOW_DAYS >= 0 and i - LOOKBACK_MOM_FAST_DAYS >= 0:
        past = c.iloc[i - LOOKBACK_MOM_SLOW_DAYS]
        recent = c.iloc[i - LOOKBACK_MOM_FAST_DAYS]
        mom = (recent / past - 1.0).replace([np.inf, -np.inf], np.nan)

    vol = pd.Series(float("nan"), index=c.columns, dtype=float)
    if i - VOL_WINDOW_DAYS >= 1:
        rets = c.iloc[i - VOL_WINDOW_DAYS : i].pct_change().iloc[1:]
        vol = rets.std(ddof=1)

    # Composite: equal-weight average of sign-corrected cross-sectional z-scores.
    composite = pd.Series(float("nan"), index=c.columns, dtype=float)
    z_mom = _cross_sectional_z(mom, higher_is_better=True)
    z_vol = _cross_sectional_z(vol, higher_is_better=False)
    both = z_mom.notna() & z_vol.notna()
    if both.any():
        composite = (z_mom.where(both) + z_vol.where(both)) / 2.0

    return {"momentum": mom, "low_vol": vol, "composite": composite}


def _cross_sectional_z(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    valid = series.dropna()
    z = pd.Series(float("nan"), index=series.index, dtype=float)
    if len(valid) < 2:
        return z
    mu = valid.mean()
    sd = valid.std(ddof=1)
    if not np.isfinite(sd) or sd <= 1e-12:
        return z
    vals = (valid - mu) / sd
    if not higher_is_better:
        vals = -vals
    z.loc[valid.index] = vals
    return z


# --------------------------------------------------------------------------
# Portfolio construction
# --------------------------------------------------------------------------

def _formation_indices(n_days: int) -> list[int]:
    """Monthly formation dates (every 21 trading days) after warmup."""
    first = LOOKBACK_MOM_SLOW_DAYS + LOOKBACK_MOM_FAST_DAYS
    return list(range(first, n_days, REBAL_DAYS))


def _universe_symbols_asof(
    panel: dict[str, Any],
    date_ts: pd.Timestamp,
) -> set[str]:
    """PIT membership as of ``date_ts`` (close basis)."""
    close = panel["close"]
    live = set(close.columns[close.loc[date_ts].notna()])
    universe = panel.get("universe")
    if universe is None or universe.empty:
        return live
    delisted = pd.to_datetime(universe["delisted_utc"], utc=True, errors="coerce")
    still = universe.loc[delisted.isna() | (delisted > pd.Timestamp(date_ts, tz="UTC"))]
    allowed = set(still["symbol"].astype(str))
    return live & allowed


def _liquidity_mask(
    panel: dict[str, Any],
    close_adj: pd.DataFrame,
    idx: int,
) -> pd.Series:
    """Price >= $2 and market cap >= $1B, both as of formation date."""
    price_ok = close_adj.iloc[idx] >= MIN_PRICE
    universe = panel.get("universe")
    if universe is None or universe.empty:
        return price_ok.fillna(False).astype(bool)
    mcap = universe.set_index("symbol")["market_cap"].astype(float)
    cap_ok = close_adj.iloc[idx].index.to_series().map(
        lambda s: float(mcap.get(s, float("nan")))
    ) >= MIN_MARKET_CAP
    return (price_ok & cap_ok).fillna(False).astype(bool)


def run_experiment(panel: dict[str, Any], cfg: ExperimentConfig) -> dict[str, Any]:
    """Run all three variants over the panel and return the full result dict."""
    close = panel["close"]
    dates = panel["dates"]
    n_days = len(close)
    if n_days < LOOKBACK_MOM_SLOW_DAYS + LOOKBACK_MOM_FAST_DAYS + 30:
        raise RuntimeError(f"panel too short ({n_days} days) for factor warmup")

    corp_actions = panel.get("corp_actions")
    split_factors = _split_adjust_factors(close, corp_actions)
    close_adj = close * split_factors
    daily_rets = close_adj.pct_change()

    div_yields = pd.DataFrame()
    if panel["source"] == "pit" and corp_actions is not None and not corp_actions.empty:
        divs = corp_actions[corp_actions["action_type"] == "dividend"]
        if not divs.empty:
            rows = []
            for _, row in divs.iterrows():
                sym = str(row.get("symbol"))
                eff = pd.Timestamp(row.get("effective_date"))
                amt = row.get("value")
                if sym in close.columns and not pd.isna(eff) and not pd.isna(amt):
                    prev = close_adj.loc[eff.normalize(), sym] if eff.normalize() in close_adj.index else None
                    rows.append({"symbol": sym, "ts": eff.normalize(), "yield": float(amt) / float(prev) if prev and prev > 0 else 0.0})
            div_yields = pd.DataFrame(rows) if rows else pd.DataFrame()

    formation_idxs = _formation_indices(n_days)
    burn_in_start = dates[0] + pd.DateOffset(months=BURN_IN_MONTHS)
    oos_start_idx = next(
        (i for i in formation_idxs if dates[i] >= burn_in_start),
        formation_idxs[0],
    )

    # Precompute portfolio weights and daily returns per variant.
    variant_returns: dict[str, np.ndarray] = {}
    variant_meta: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        port_ret, turnover_by_rebal, trades = _simulate_variant(
            panel=panel,
            close_adj=close_adj,
            daily_rets=daily_rets,
            div_yields=div_yields,
            formation_idxs=formation_idxs,
            oos_start_idx=oos_start_idx,
            variant=variant,
        )
        variant_returns[variant] = port_ret[oos_start_idx:]
        variant_meta[variant] = {
            "turnover": turnover_by_rebal,
            "trades": trades,
            "oos_start": str(dates[oos_start_idx].date()),
        }

    # SPY benchmark (total return over the OOS window).
    spy_total, spy_series = _spy_benchmark(panel, cfg, dates, oos_start_idx)

    results: dict[str, Any] = {}
    for variant in VARIANTS:
        rets = variant_returns[variant]
        meta = variant_meta[variant]
        results[variant] = _evaluate_variant(
            variant=variant,
            returns=rets,
            spy_total=spy_total,
            cfg=cfg,
            meta=meta,
        )

    return {
        "meta": _build_meta(panel, cfg, oos_start_idx, dates, spy_series),
        "spy": {
            "oos_total_return": spy_total,
            "oos_start": str(dates[oos_start_idx].date()),
            "oos_end": str(dates[-1].date()),
        },
        "variants": results,
    }


def _simulate_variant(
    *,
    panel: dict[str, Any],
    close_adj: pd.DataFrame,
    daily_rets: pd.DataFrame,
    div_yields: pd.DataFrame,
    formation_idxs: list[int],
    oos_start_idx: int,
    variant: str,
) -> tuple[np.ndarray, list[float], int]:
    """Simulate one variant: daily returns, turnover list, trade count."""
    n_days = len(close_adj)
    port_ret = np.zeros(n_days)
    prev_weights: dict[str, float] = {}
    trade_count = 0
    turnover_list: list[float] = []

    for fi, i in enumerate(formation_idxs):
        date_ts = close_adj.index[i]
        members = _universe_symbols_asof(panel, date_ts)
        liq = _liquidity_mask(panel, close_adj, i)
        factors = compute_factors(close_adj, i)
        factor = factors[variant]
        eligible = [
            s for s in members
            if liq.get(s, False) and pd.notna(factor.get(s))
        ]
        if not eligible:
            # No trade: hold cash this period; carry no weights.
            prev_weights = {}
            continue

        ranked = pd.Series({s: float(factor[s]) for s in eligible}).sort_values(ascending=False)
        n_keep = max(1, int(np.ceil(TOP_QUANTILE * len(ranked))))
        selected = set(ranked.head(n_keep).index)
        w_new = {s: 1.0 / n_keep for s in selected}

        # Turnover vs carried weights (delisted names drop to cash).
        carried = {s: w for s, w in prev_weights.items() if s in members}
        all_syms = set(carried) | set(w_new)
        turnover = 0.5 * sum(abs(w_new.get(s, 0.0) - carried.get(s, 0.0)) for s in all_syms)
        cost = turnover * COST_BPS_TURNOVER / 10_000.0

        # Commission: 0.005/share on the traded dollar value, unit equity E=1.
        commission = 0.0
        for s in all_syms:
            dw = abs(w_new.get(s, 0.0) - carried.get(s, 0.0))
            if dw <= 0.0:
                continue
            price = close_adj.iloc[i].get(s)
            if pd.notna(price) and price > 0.0:
                shares = dw / price
                commission += COMMISSION_PER_SHARE * shares

        trade_count += sum(1 for s in all_syms if abs(w_new.get(s, 0.0) - carried.get(s, 0.0)) > 1e-12)
        turnover_list.append(turnover)
        prev_weights = w_new

        # Holding period: days i+1 .. next formation (or end).
        end = formation_idxs[fi + 1] if fi + 1 < len(formation_idxs) else n_days
        for t in range(i, min(end, n_days)):
            if t == i:
                port_ret[t] -= cost + commission
                continue
            day_rets = []
            for s in selected:
                r = daily_rets.iloc[t].get(s)
                if pd.notna(r):
                    day_rets.append(float(r))
            if day_rets:
                port_ret[t] += float(np.mean(day_rets))
            if not div_yields.empty:
                ts = daily_rets.index[t]
                for s in selected:
                    row = div_yields[(div_yields["symbol"] == s) & (div_yields["ts"] == ts)]
                    if not row.empty:
                        port_ret[t] += float(row["yield"].iloc[0])

    return port_ret, turnover_list, trade_count


def _spy_benchmark(
    panel: dict[str, Any],
    cfg: ExperimentConfig,
    dates: pd.DatetimeIndex,
    oos_start_idx: int,
) -> tuple[float | None, pd.Series | None]:
    """SPY total return over the OOS window, or None when unavailable."""
    oos_dates = dates[oos_start_idx:]
    if panel["source"] == "pit" and "SPY" in panel["close"].columns:
        spy = panel["close"]["SPY"].dropna()
        spy_rets = spy.pct_change().dropna()
        return float(np.prod(1.0 + spy_rets.loc[oos_dates[1:]]) - 1.0), spy_rets
    if cfg.spy_csv and Path(cfg.spy_csv).exists():
        spy_df = _read_spy_fixture(cfg.spy_csv)
        adj = spy_df["adj_close"].dropna()
        adj = adj.reindex(adj.index.intersection(oos_dates)).dropna()
        if len(adj) >= 2:
            return float(adj.iloc[-1] / adj.iloc[0] - 1.0), adj
    return None, None


# --------------------------------------------------------------------------
# Evaluation and gate
# --------------------------------------------------------------------------

def _evaluate_variant(
    *,
    variant: str,
    returns: np.ndarray,
    spy_total: float | None,
    cfg: ExperimentConfig,
    meta: dict[str, Any],
) -> dict[str, Any]:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 30:
        return {
            "variant": variant,
            "error": f"insufficient OOS days ({len(r)})",
            "verdict": "ERROR",
        }

    rf_daily = (1.0 + RF_ANNUAL) ** (1.0 / TRADING_DAYS) - 1.0
    r_excess = r - rf_daily
    oos_sharpe = float(np.mean(r_excess) / np.std(r_excess, ddof=1) * np.sqrt(TRADING_DAYS))
    ci_lower, ci_upper = bootstrap_confidence_interval(
        r, block_size=cfg.block_len, n_boot=cfg.n_bootstrap, ci=CI_LEVEL, seed=cfg.seed
    )

    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurtosis()) + 3.0  # raw kurtosis
    per_obs_sr = float(np.mean(r_excess) / np.std(r_excess, ddof=1))
    comps = dsr_components(
        observed_sharpe=per_obs_sr,
        num_trials=len(VARIANTS),
        sample_length=len(r),
        skewness=skew,
        kurtosis=kurt,
    )
    dsr_p = comps["p_value"]

    net_total = float(np.prod(1.0 + r) - 1.0)
    years = len(r) / TRADING_DAYS
    net_annual = float((1.0 + net_total) ** (1.0 / years) - 1.0) if net_total > -1.0 else -1.0
    spy_annual = None
    annual_excess_pp = None
    if spy_total is not None:
        spy_annual = float((1.0 + spy_total) ** (1.0 / years) - 1.0) if spy_total > -1.0 else -1.0
        annual_excess_pp = (net_annual - spy_annual) * 100.0

    per_year = _per_year_sharpe(r, meta.get("oos_start", ""), rf_daily)
    per_fold = _walk_forward_folds(r, rf_daily, fold_days=TRADING_DAYS)

    turnover = meta.get("turnover") or []
    avg_turnover = float(np.mean(turnover)) if turnover else 0.0
    trade_count = int(meta.get("trades", 0))

    beat_spy = bool(spy_total is not None and net_total > spy_total)
    raw_pass = (
        ci_lower >= GATE_RAW["oos_sharpe_ci_lower"]
        and dsr_p <= GATE_RAW["dsr_p"]
        and beat_spy
    )
    margin_pass = (
        ci_lower >= GATE_MARGIN["oos_sharpe_ci_lower"]
        and dsr_p <= GATE_MARGIN["dsr_p"]
        and annual_excess_pp is not None
        and annual_excess_pp >= GATE_MARGIN["net_annual_excess_pp"]
    )
    if raw_pass and margin_pass:
        verdict = "PASS"
    elif raw_pass and not margin_pass:
        verdict = "FAIL - insufficient margin"
    else:
        verdict = "FAIL"

    criteria = {
        "oos_sharpe_ci_lower_90": {"value": round(float(ci_lower), 4), "raw_gate": GATE_RAW["oos_sharpe_ci_lower"], "margin_gate": GATE_MARGIN["oos_sharpe_ci_lower"]},
        "dsr_p": {"value": round(float(dsr_p), 6), "raw_gate": GATE_RAW["dsr_p"], "margin_gate": GATE_MARGIN["dsr_p"]},
        "net_total_vs_spy": {"net_total": round(float(net_total), 6), "spy_total": round(float(spy_total), 6) if spy_total is not None else None, "raw_gate": "net_total > spy_total"},
        "net_annual_excess_pp": {"value": round(float(annual_excess_pp), 4) if annual_excess_pp is not None else None, "margin_gate": GATE_MARGIN["net_annual_excess_pp"]},
    }

    return {
        "variant": variant,
        "oos_sharpe": round(float(oos_sharpe), 4),
        "oos_sharpe_ci_90": [round(float(ci_lower), 4), round(float(ci_upper), 4)],
        "dsr_p": round(float(dsr_p), 6),
        "dsr_components": {k: (round(float(v), 6) if isinstance(v, float) else v) for k, v in comps.items()},
        "net_total_return": round(float(net_total), 6),
        "net_annual_return": round(float(net_annual), 6),
        "spy_total_return": round(float(spy_total), 6) if spy_total is not None else None,
        "annual_excess_pp": round(float(annual_excess_pp), 4) if annual_excess_pp is not None else None,
        "avg_turnover": round(float(avg_turnover), 5),
        "trade_count": trade_count,
        "n_oos_days": len(r),
        "oos_return_series": [round(float(x), 10) for x in r],
        "per_year_sharpe": per_year,
        "per_fold_sharpe": per_fold,
        "criteria": criteria,
        "verdict": verdict,
    }


def _per_year_sharpe(r: np.ndarray, oos_start: str, rf_daily: float) -> dict[str, float]:
    if not oos_start:
        return {}
    start = pd.Timestamp(oos_start)
    idx = pd.date_range(start=start, periods=len(r), freq="B")
    s = pd.Series(r, index=idx)
    out: dict[str, float] = {}
    for year, grp in s.groupby(s.index.year):
        if len(grp) < 20:
            continue
        exc = grp.to_numpy() - rf_daily
        sd = float(np.std(exc, ddof=1))
        out[str(year)] = round(float(np.mean(exc) / sd * np.sqrt(TRADING_DAYS)), 4) if sd > 1e-12 else 0.0
    return out


def _walk_forward_folds(r: np.ndarray, rf_daily: float, fold_days: int) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for k in range(0, len(r), fold_days):
        chunk = r[k : k + fold_days]
        if len(chunk) < 20:
            continue
        exc = chunk - rf_daily
        sd = float(np.std(exc, ddof=1))
        sharpe = round(float(np.mean(exc) / sd * np.sqrt(TRADING_DAYS)), 4) if sd > 1e-12 else 0.0
        folds.append({"fold": len(folds) + 1, "start_day": k, "n_days": len(chunk), "oos_sharpe_annualized": sharpe})
    return folds


def _build_meta(
    panel: dict[str, Any],
    cfg: ExperimentConfig,
    oos_start_idx: int,
    dates: pd.DatetimeIndex,
    spy_series: pd.Series | None,
) -> dict[str, Any]:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        rev = "unknown"
    return {
        "protocol": PROTOCOL_DOC,
        "protocol_lock_commit": PROTOCOL_LOCK_COMMIT,
        "gate": {"raw": GATE_RAW, "margin": GATE_MARGIN},
        "variants": list(VARIANTS),
        "costs": {
            "turnover_bps": COST_BPS_TURNOVER,
            "commission_per_share": COMMISSION_PER_SHARE,
            "financing": "none (long-only)",
        },
        "portfolio": {
            "top_quantile": TOP_QUANTILE,
            "equal_weight": True,
            "rebalance_days": REBAL_DAYS,
            "long_only": True,
            "leverage": 0.0,
        },
        "factors": {
            "momentum": f"close[T-{LOOKBACK_MOM_FAST_DAYS}]/close[T-{LOOKBACK_MOM_SLOW_DAYS}]-1",
            "low_vol": f"{VOL_WINDOW_DAYS}-day realized vol, ranked ascending",
            "composite": "equal-weight of sign-corrected z-scores",
        },
        "liquidity_filters": {
            "min_price": MIN_PRICE,
            "min_market_cap": MIN_MARKET_CAP,
            "market_cap_note": "pit: latest universe snapshot's current market cap (PIT cap needs shares_outstanding x price series; recorded limitation). triage: price-only.",
        },
        "methodology_notes": [
            "Factor inputs are split-adjusted closes (splits removed; raw bars preserved). Backward split adjustment preserves pre-ex-date ratios, so no lookahead enters factor values.",
            "Strategy returns are split-adjusted price returns + cash-credited dividends on ex-dates (pit). Triage panel is total-return-adjusted (dividends already in prices).",
            "Sharpe: rf=4% annual, geometrically decompounded daily. DSR uses per-observation Sharpe (published B&LdP 2014 formula, see docs/dsr_audit.md).",
            "Walk-forward: anchored, fixed annual OOS folds over the OOS daily series (no fitted parameters; folds are a time-segment robustness check).",
            "Burn-in: first 18 months of the panel reserved (not evaluated).",
        ],
        "window": panel["window"],
        "oos_start": str(dates[oos_start_idx].date()),
        "oos_end": str(dates[-1].date()),
        "source": panel["source"],
        "triage": bool(cfg.triage),
        "provenance": panel["provenance"],
        "spy_benchmark": {
            "source": "pit store SPY bars + dividends" if spy_series is not None and panel["source"] == "pit" else "committed Yahoo fixture (adj_close)" if spy_series is not None else "unavailable",
        },
        "run": {
            "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "git_head": rev,
            "n_bootstrap": cfg.n_bootstrap,
            "block_len": cfg.block_len,
            "seed": cfg.seed,
        },
    }


def write_results(results: dict[str, Any], out_path: str | Path) -> Path:
    """Serialize results to JSON (NaN-safe)."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_safe(v) for v in obj]
        if isinstance(obj, float):
            if not np.isfinite(obj):
                return None
            return obj
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    payload = _safe(results)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def default_out_path(cfg: ExperimentConfig) -> str:
    if cfg.triage:
        return str(Path("reports") / "cross_sectional_factor_triage_smoke.json")
    return RESULTS_PATH
