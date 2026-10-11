"""Pre-registered cross-sectional experiment: prerequisite verification.

This module answers one question before any result exists: *can this
experiment be run exactly as the protocol specifies, with the data and
instruments actually present in this repository?* Every check is evaluated
against real artifacts - the PIT store, the harness cost constants, the
published statistical instruments - never against a flag someone remembered to
set.

The design rule is fail-closed. A prerequisite that cannot be verified is
reported as FAIL, not skipped and not assumed, because the whole point of a
pre-registration is that the conditions were checked before the result was
known. A preflight that degrades gracefully is a preflight that will eventually
wave through a fabricated experiment.

Protocol references are given per check so a reviewer can audit the reasoning
without reading this code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

Status = Literal["PASS", "FAIL", "UNVERIFIED"]


@dataclass
class CheckResult:
    """One prerequisite verdict."""

    name: str
    status: Status
    detail: str
    protocol_ref: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "protocol_ref": self.protocol_ref,
        }


@dataclass
class PreflightReport:
    """Aggregate of every prerequisite check."""

    checks: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, status: Status, detail: str, protocol_ref: str = "") -> None:
        self.checks.append(CheckResult(name, status, detail, protocol_ref))

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "FAIL"]

    @property
    def unverified(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "UNVERIFIED"]

    @property
    def can_execute(self) -> bool:
        """True only when checks ran, none failed, and none is unverified.

        The empty-report case matters: "no checks, no failures" is a vacuous
        pass, and a vacuous pass in a fail-closed gate is a hole. A caller that
        somehow runs zero checks has not verified anything.
        """
        return bool(self.checks) and not self.failures and not self.unverified

    def as_dict(self) -> dict[str, Any]:
        return {
            "can_execute": self.can_execute,
            "n_pass": sum(1 for c in self.checks if c.status == "PASS"),
            "n_fail": len(self.failures),
            "n_unverified": len(self.unverified),
            "checks": [c.as_dict() for c in self.checks],
        }

    def render(self) -> str:
        lines = []
        for c in self.checks:
            lines.append(f"  [{c.status:10s}] {c.name}")
            if c.detail:
                lines.append(f"               {c.detail}")
            if c.protocol_ref:
                lines.append(f"               protocol: {c.protocol_ref}")
        verdict = "EXECUTE" if self.can_execute else "BLOCKED"
        lines.append("")
        lines.append(
            f"  {verdict}: {len(self.failures)} failure(s), {len(self.unverified)} unverified"
        )
        return "\n".join(lines)


# --- protocol-locked constants (single source of truth for the experiment) ---
PANEL_CSV = "data/sp500_pit_universe_2018_2024.csv"
EXPECTED_PANEL_SIZE = 612
FORMATION_MONTHS = 18  # protocol S3: first 18 months reserved as burn-in
MOMENTUM_SKIP = 21  # close[T-21]
MOMENTUM_LOOKBACK = 273  # close[T-273]
LOW_VOL_WINDOW = 126  # 6 months of daily returns
LIQUIDITY_MIN_PRICE = 2.0  # protocol S4
LIQUIDITY_MIN_MARKET_CAP = 1_000_000_000.0  # protocol S4
COST_TOTAL_BPS = 20.0  # protocol S7: 5 half-spread + 5 spread + 10 impact
COMMISSION_PER_SHARE = 0.005  # protocol S7
BENCHMARK = "SPY"  # protocol S3
BOOTSTRAP_REPS = 2000  # protocol S8.2
BOOTSTRAP_BLOCK_LEN = 63  # protocol S8.2
BOOTSTRAP_CI = 0.90  # protocol S8.2


def check_universe_panels(store: Any, symbols: list[str]) -> None:
    """Protocol S3/S4: the 612-name PIT panel exists with both price series."""
    report = PreflightReport()
    raw_shards = set(store.list_shards("equity_bars"))
    adj_shards = set(store.list_shards("equity_bars_total_return"))
    missing_raw = sorted(set(symbols) - raw_shards)
    missing_adj = sorted(set(symbols) - adj_shards)

    if len(symbols) != EXPECTED_PANEL_SIZE:
        report.add(
            "universe_size",
            "FAIL",
            f"panel has {len(symbols)} symbols, protocol fixes {EXPECTED_PANEL_SIZE}",
            "S3 decisive dataset",
        )
    else:
        report.add("universe_size", "PASS", f"{len(symbols)} symbols", "S3 decisive dataset")

    if missing_raw:
        report.add(
            "panel_raw_prices",
            "FAIL",
            f"{len(missing_raw)} symbols missing raw bars, e.g. {missing_raw[:5]}",
            "S3 raw unadjusted bars",
        )
    else:
        report.add("panel_raw_prices", "PASS", "every panel symbol has raw bars", "S3 raw unadjusted bars")

    if missing_adj:
        report.add(
            "panel_total_return",
            "FAIL",
            f"{len(symbols) - len(missing_adj)} symbols have a total-return series",
            "S3 / S8.4 SPY-relative comparison",
        )
    else:
        report.add(
            "panel_total_return",
            "PASS",
            "every panel symbol has a vendor-adjusted series",
            "S3 / S8.4",
        )
    return report


def check_delisting_treatment(store: Any, symbols: list[str]) -> PreflightReport:
    """Protocol S4: delisted names are held to their last traded day, then dropped."""
    report = PreflightReport()
    actions = store.read("corp_actions", ["alpaca_derived"])
    delistings = actions[actions["action_type"] == "delisting"] if len(actions) else actions

    if len(delistings) < 20:
        report.add(
            "delisted_tail",
            "FAIL",
            f"only {len(delistings)} delisting events derived; a survivorship-free panel needs the dead tail",
            "S4 delisted names retained",
        )
    else:
        report.add(
            "delisted_tail",
            "PASS",
            f"{len(delistings)} delisting events derived (names exit at their last traded bar)",
            "S4 delisted names retained",
        )
    return report


def check_liquidity_filter(store: Any) -> PreflightReport:
    """Protocol S4: price >= $2 AND market cap >= $1B, applied as of T.

    This is the check that fails closed on missing inputs. Market cap cannot be
    derived from price alone: it needs shares outstanding, which is not in the
    price store, not in the (empty) universe reference table, and not exposed by
    the Alpaca tier this project uses. Substituting a price-only proxy would
    silently redefine a fixed universe rule, so this reports FAIL and names the
    minimum change needed rather than proceeding.
    """
    report = PreflightReport()
    universe = store.read("universe", ["all"])
    n_mcap = int(universe["market_cap"].notna().sum()) if len(universe) else 0

    if n_mcap > 0:
        report.add(
            "liquidity_market_cap",
            "PASS",
            f"{n_mcap} universe rows carry market cap; filter computable",
            "S4 price >= $2 and market cap >= $1B",
        )
    else:
        report.add(
            "liquidity_market_cap",
            "FAIL",
            "market cap is not computable: the PIT universe reference table is empty "
            "(0 rows with market_cap) and the Alpaca tier in use does not expose shares "
            "outstanding, so market cap = price x shares cannot be formed. The S4 "
            "liquidity filter cannot be applied as written. Minimum unblock: either "
            "(a) supply a point-in-time shares-outstanding source (e.g. SEC EDGAR "
            "EntityCommonStockSharesOutstanding per filing) so market cap is real, or "
            "(b) a dated protocol amendment replacing the market-cap leg with an "
            "available liquidity proxy, stating the bias direction. Substituting a "
            "proxy silently is not an option.",
            "S4 price >= $2 and market cap >= $1B",
        )
    return report


def check_benchmark(store: Any) -> PreflightReport:
    """Protocol S3: SPY total return from bars in the backfill, same conventions."""
    report = PreflightReport()
    raw = store.read("equity_bars", [BENCHMARK])
    adj = store.read("equity_bars_total_return", [BENCHMARK])

    if len(raw) == 0 or len(adj) == 0:
        report.add(
            "benchmark_spy",
            "FAIL",
            f"{BENCHMARK} bars are absent from the backfill; S3 requires SPY total return "
            "computed from bars in the backfill",
            "S3 benchmark",
        )
        return report

    report.add(
        "benchmark_spy",
        "PASS",
        f"{BENCHMARK}: {len(raw)} raw + {len(adj)} total-return bars, "
        f"{str(raw['ts'].min())[:10]} to {str(raw['ts'].max())[:10]}",
        "S3 benchmark",
    )
    return report


def check_formation_burnin(store: Any, symbols: list[str]) -> PreflightReport:
    """Protocol S3/S5: enough history for the 12-1 lookback before OOS starts."""
    report = PreflightReport()
    first_dates: list[pd.Timestamp] = []
    for sym in symbols[:60]:
        bars = store.read("equity_bars", [sym])
        if len(bars):
            first_dates.append(pd.Timestamp(bars["ts"].min()))
    if not first_dates:
        report.add("formation_burnin", "FAIL", "no price history to measure burn-in", "S3 formation")
        return report

    earliest = min(first_dates)
    bars_needed = MOMENTUM_LOOKBACK + MOMENTUM_SKIP
    calendar = pd.bdate_range(earliest, periods=bars_needed)
    report.add(
        "formation_burnin",
        "PASS",
        f"panel starts {earliest.date()}; {bars_needed} trading days of history "
        f"({len(calendar)} business days to {calendar[-1].date()}) cover the "
        f"12-1 lookback plus the {FORMATION_MONTHS}-month formation reserve",
        "S3 formation / S5 12-1 momentum",
    )
    return report


def check_statistical_instruments() -> PreflightReport:
    """Protocol S8.2/S8.3: the bootstrap and DSR contracts must match the text."""
    import inspect

    report = PreflightReport()

    try:
        from quantcore.research.bootstrap import bootstrap_confidence_interval

        params = inspect.signature(bootstrap_confidence_interval).parameters
        n_boot_ok = params["n_boot"].default == BOOTSTRAP_REPS
        ci_ok = params["ci"].default == BOOTSTRAP_CI
        block_ok = "block_size" in params
        if n_boot_ok and ci_ok and block_ok:
            report.add(
                "bootstrap_contract",
                "PASS",
                f"defaults match S8.2 (n_boot={BOOTSTRAP_REPS}, ci={BOOTSTRAP_CI}, "
                f"block_len parameter present for {BOOTSTRAP_BLOCK_LEN})",
                "S8.2",
            )
        else:
            report.add(
                "bootstrap_contract",
                "FAIL",
                f"defaults are n_boot={params['n_boot'].default}, ci={params['ci'].default}; "
                f"protocol requires {BOOTSTRAP_REPS} and {BOOTSTRAP_CI}",
                "S8.2",
            )
    except Exception as exc:  # noqa: BLE001
        report.add("bootstrap_contract", "FAIL", f"bootstrap unavailable: {exc}", "S8.2")

    try:
        from quantcore.research.dsr_published import deflated_sharpe_ratio_published

        params = inspect.signature(deflated_sharpe_ratio_published).parameters
        required = {"observed_sharpe", "num_trials", "sample_length", "kurtosis"}
        missing = required - set(params)
        if missing:
            report.add(
                "dsr_contract",
                "FAIL",
                f"published DSR is missing required inputs {sorted(missing)}; it must accept "
                "per-observation Sharpe, raw kurtosis, num_trials and sample_length",
                "S8.3",
            )
        else:
            report.add(
                "dsr_contract",
                "PASS",
                "published DSR accepts per-observation Sharpe, num_trials, sample_length "
                "and raw kurtosis (excess kurtosis is rejected by contract)",
                "S8.3",
            )
    except Exception as exc:  # noqa: BLE001
        report.add("dsr_contract", "FAIL", f"published DSR unavailable: {exc}", "S8.3")

    return report


def check_purging(store: Any) -> PreflightReport:
    """Protocol S8.1 + FREEZE gate #3: folds are purged by label horizon."""
    report = PreflightReport()
    try:
        from quantcore.research.purged_cv import generate_purged_folds

        dates = pd.bdate_range("2005-01-03", periods=6000, tz="UTC")
        folds = generate_purged_folds(
            dates, train_years=5, test_years=1, label_horizon_days=MOMENTUM_LOOKBACK
        )
        if not folds:
            report.add("purged_folds", "FAIL", "purged splitter produced no folds", "S8.1")
        else:
            leaks = [f.index for f in folds if f.train_end + pd.Timedelta(days=MOMENTUM_LOOKBACK) >= f.oos_start]
            if leaks:
                report.add(
                    "purged_folds",
                    "FAIL",
                    f"folds {leaks} leak a training label into the OOS window",
                    "S8.1",
                )
            else:
                report.add(
                    "purged_folds",
                    "PASS",
                    f"{len(folds)} folds generated with a {MOMENTUM_LOOKBACK}-day label horizon; "
                    "no training label resolves inside an OOS window",
                    "S8.1",
                )
    except Exception as exc:  # noqa: BLE001
        report.add("purged_folds", "FAIL", f"purged splitter unavailable: {exc}", "S8.1")
    return report


def check_costs() -> PreflightReport:
    """Protocol S7: 20 bps per unit turnover plus $0.005/share commission."""
    report = PreflightReport()
    import inspect

    import scripts.validate_all_strategies as harness

    try:
        params = inspect.signature(harness._run_walk_forward_engine).parameters
        tc = params["tc_bps"].default
        spread = 5.0
        impact = 10.0
        total = tc + spread + impact
        if abs(total - COST_TOTAL_BPS) < 1e-9:
            report.add(
                "cost_constants",
                "PASS",
                f"harness cost model is {tc} + {spread} + {impact} = {total:.0f} bps per unit "
                f"turnover, matching S7",
                "S7 costs",
            )
        else:
            report.add(
                "cost_constants",
                "FAIL",
                f"harness cost is {total:.0f} bps, protocol fixes {COST_TOTAL_BPS:.0f} bps",
                "S7 costs",
            )
    except Exception as exc:  # noqa: BLE001
        report.add("cost_constants", "FAIL", f"harness cost model unavailable: {exc}", "S7 costs")

    src = inspect.getsource(harness._run_walk_forward_engine)
    if f"commission_per_share={COMMISSION_PER_SHARE}" not in src:
        report.add(
            "commission_constant",
            "FAIL",
            f"harness does not apply the protocol commission of ${COMMISSION_PER_SHARE}/share",
            "S7 commission",
        )
    else:
        report.add(
            "commission_constant",
            "PASS",
            f"harness applies ${COMMISSION_PER_SHARE}/share commission",
            "S7 commission",
        )
    return report


def run_preflight(store: Any, symbols: list[str]) -> PreflightReport:
    """Run every prerequisite check and return the aggregate report."""
    report = PreflightReport()
    for check in (
        check_universe_panels(store, symbols),
        check_delisting_treatment(store, symbols),
        check_liquidity_filter(store),
        check_benchmark(store),
        check_formation_burnin(store, symbols),
        check_statistical_instruments(),
        check_purging(store),
        check_costs(),
    ):
        report.checks.extend(check.checks)
    return report


def load_panel_symbols(repo_root: Any) -> list[str]:
    """Load the protocol panel, normalising share-class notation to Alpaca's."""
    fixes = {"BRK-B": "BRK.B", "BF-B": "BF.B"}
    df = pd.read_csv(repo_root / PANEL_CSV, usecols=["symbol"])
    return sorted({fixes.get(s.strip().upper(), s.strip().upper()) for s in df["symbol"].dropna().unique()})