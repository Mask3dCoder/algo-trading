"""Launch paper trading for one or more strategies simultaneously.

Starts one paper_trade.py process per strategy.

Execution modes:
  --execution standard   Default. Direct order submission to broker.
  --execution smart      SmartExecutionRouter: limit optimization,
                         spread capture, TWAP splitting, slippage tracking,
                         cost monitoring to reports/execution_costs.csv.

Usage:
    python scripts/launch_paper_trading.py
    python scripts/launch_paper_trading.py --strategies sentiment_momentum,stat_arb_mean_reversion
    python scripts/launch_paper_trading.py --strategies time_series_momentum,carry
    python scripts/launch_paper_trading.py --duration 5d --data-source yahoo
    python scripts/launch_paper_trading.py --live --scale single_share
    python scripts/launch_paper_trading.py --duration 1d --execution smart
    python scripts/launch_paper_trading.py --duration 5d --execution smart --broker ibkr
    python scripts/launch_paper_trading.py --strategies macro_cross_sectional_momentum,macro_state_selective_momentum,pairs_trading_intraday --execution smart --duration 30d
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import signal
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# BLUECHIP 100 — same universe used in walk-forward validation sweep
BLUECHIP_SYMBOLS = (
    "AAPL,MSFT,JNJ,PFE,PG,KO,PEP,WMT,T,VZ,"
    "XOM,CVX,MRK,ABT,BMY,LLY,IBM,CSCO,INTC,"
    "BA,GE,MMM,DIS,HD,NKE,MCD,SBUX,YUM,"
    "WFC,BAC,C,USB,PNC,TFC,COF,SCHW,"
    "JPM,GS,MS,BLK,AXP,V,MA,COST,PCAR,"
    "ADM,SHW,CLX,GIS,KMB,SYY,HSY,MKC,"
    "EMR,ETN,PH,ROK,DOV,ITW,CAT,DE,HON,"
    "UPS,FDX,RTX,LMT,NOC,GD,TDG,"
    "AON,MMC,AJG,TRV,ALL,MET,PRU,"
    "AVB,EQR,EXR,PSA,AMT,EQIX,CCI,"
    "NSC,UNP,CSX,F,"
    "CMCSA,ORCL,ACN,INFY,ADI,TXN,QCOM"
)

DEPLOY_STRATEGIES = [
    "macro_cross_sectional_momentum",
    "macro_state_selective_momentum",
    "pairs_trading_intraday",
]

FACTORY_CONFIGS_DIR = REPO_ROOT / "quantcore" / "strategy_factory" / "configs"

_BAR_SIZE_MINUTES: dict[str, int] = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "1h": 60,
}


def _bar_size_to_minutes(bar_size: str) -> int:
    """Convert bar size string to integer minutes."""
    return _BAR_SIZE_MINUTES.get(bar_size, 5)


def is_factory_config(name: str) -> bool:
    """Check if a strategy name matches a YAML factory config."""
    return (FACTORY_CONFIGS_DIR / f"{name}.yaml").is_file()


def is_intraday_strategy(name: str) -> bool:
    """Check if a factory config specifies intraday data source."""
    config_path = FACTORY_CONFIGS_DIR / f"{name}.yaml"
    if not config_path.is_file():
        return False
    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("data_source") == "intraday"
    except Exception:
        return False


def get_config_bar_size(name: str) -> str | None:
    """Get bar_size from factory config, if specified."""
    config_path = FACTORY_CONFIGS_DIR / f"{name}.yaml"
    if not config_path.is_file():
        return None
    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("bar_size")
    except Exception:
        return None


def launch_paper_trading(
    strategies: list[str],
    duration: str = "5d",
    data_source: str = "mock",
    broker: str = "mock",
    capital: float = 100_000.0,
    symbols: str = BLUECHIP_SYMBOLS,
    live_mode: bool = False,
    scale: str = "full",
    execution_mode: str = "standard",
    bar_size: str = "5min",
) -> list[subprocess.Popen]:
    """Launch paper trading processes for each strategy."""
    processes = []
    paper_script = str(REPO_ROOT / "scripts" / "paper_trade.py")

    for idx, strat in enumerate(strategies):
        # Detect factory config vs framework strategy
        if is_factory_config(strat):
            strat_flag = "--factory-config"
        else:
            strat_flag = "--framework"

        # Determine effective data source: intraday strategies override global setting
        effective_data_source = data_source
        effective_bar_size = bar_size
        if is_intraday_strategy(strat):
            effective_data_source = "intraday"
            config_bar = get_config_bar_size(strat)
            if config_bar:
                effective_bar_size = config_bar

        cmd = [
            sys.executable, paper_script,
            "--duration", duration,
            strat_flag, strat,
            "--data-source", effective_data_source,
            "--broker", broker,
            "--capital", str(int(capital)),
            "--symbols", symbols,
            "--session-id", strat,
        ]
        if effective_data_source == "intraday":
            # paper_trade.py uses --resample (integer minutes)
            bar_minutes = _bar_size_to_minutes(effective_bar_size)
            if bar_minutes > 1:
                cmd.extend(["--resample", str(bar_minutes)])
        if live_mode:
            cmd.extend(["--live", "--scale", scale, "--yes"])

        log_file = REPO_ROOT / "reports" / f"paper_{strat}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Each strategy needs a unique IBKR client ID
        env = os.environ.copy()
        env["IBKR_CLIENT_ID"] = str(1 + idx)

        # Smart execution: inject via environment variable
        if execution_mode == "smart":
            env["QUANTCORE_SMART_EXECUTION"] = "1"
            env["QUANTCORE_COST_REPORT"] = str(REPO_ROOT / "reports" / "execution_costs.csv")

        exec_label = " [SMART]" if execution_mode == "smart" else ""
        mode_label = "[factory]" if strat_flag == "--factory-config" else "[framework]"
        print(f"[launch] Starting {strat} {mode_label}{exec_label}")
        print(f"  Command: {' '.join(cmd[:6])}...")
        print(f"  Log:     {log_file}")
        print(f"  IBKR client ID: {1 + idx}")

        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT),
                env=env,
            )
        processes.append((strat, proc))
        print(f"  PID:     {proc.pid}")

    return processes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch paper trading for both DEPLOY strategies",
    )
    parser.add_argument(
        "--duration", "-d", default="5d",
        help="Run duration (default: 5d)",
    )
    parser.add_argument(
        "--data-source", default="mock",
        choices=["mock", "alpaca", "yahoo"],
        help="Data source (default: mock)",
    )
    parser.add_argument(
        "--broker", default="mock",
        choices=["mock", "alpaca", "ibkr"],
        help="Broker (default: mock)",
    )
    parser.add_argument(
        "--capital", type=float, default=100_000,
        help="Starting capital per strategy (default: 100000)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Enable live trading mode (real broker, with warnings)",
    )
    parser.add_argument(
        "--scale", default="full",
        choices=["single_share", "fractional", "full"],
        help="Scale mode for live trading (default: full)",
    )
    parser.add_argument(
        "--execution", "-e", default="standard",
        choices=["standard", "smart"],
        help="Execution mode (default: standard). "
             "'smart' enables limit optimization, spread capture, "
             "TWAP splitting, and cost monitoring.",
    )
    parser.add_argument(
        "--strategies", default=None,
        help="Comma-separated list of strategy names to launch. "
             "Can be framework registry names (e.g., 'sentiment_momentum') "
             "or factory config names (e.g., 'time_series_momentum', 'carry'). "
             "Factory configs are auto-detected from YAML files in configs/. "
             "Intraday strategies (e.g., 'pairs_trading_intraday') automatically "
             "switch to intraday data feed. "
             "If not provided, launches the default DEPLOY strategies.",
    )
    parser.add_argument(
        "--bar-size", default="5min",
        choices=["1min", "5min", "15min", "30min", "1h"],
        help="Bar size for intraday strategies (default: 5min). "
             "Overridden by per-config bar_size for intraday factory configs.",
    )
    args = parser.parse_args()

    # Resolve strategy list
    if args.strategies:
        strategies = list(dict.fromkeys(
            s.strip() for s in args.strategies.split(",") if s.strip()
        ))
    else:
        strategies = DEPLOY_STRATEGIES

    # Classify strategies for display
    intraday_strats = [s for s in strategies if is_intraday_strategy(s)]
    daily_strats = [s for s in strategies if not is_intraday_strategy(s)]

    print("=" * 60)
    print("  QUANTCORE — PAPER TRADING LAUNCHER")
    print("=" * 60)
    print(f"  Strategies:  {', '.join(strategies)}")
    if intraday_strats:
        print(f"    Intraday:  {', '.join(intraday_strats)} ({args.bar_size})")
    if daily_strats:
        print(f"    Daily:     {', '.join(daily_strats)}")
    print(f"  Universe:    BLUECHIP 100 ({len(BLUECHIP_SYMBOLS.split(','))} symbols)")
    print(f"  Duration:    {args.duration}")
    print(f"  Data source: {args.data_source} (intraday strategies auto-switch)")
    print(f"  Broker:      {args.broker}")
    print(f"  Capital:     ${args.capital:,.0f} per strategy")
    if args.live:
        print(f"  Mode:        LIVE ({args.scale})")
        print("  *** REAL MONEY WARNING ***")
    else:
        print("  Mode:        PAPER")
    print(f"  Execution:   {args.execution}")
    print("=" * 60)

    processes = launch_paper_trading(
        strategies=strategies,
        duration=args.duration,
        data_source=args.data_source,
        broker=args.broker,
        capital=args.capital,
        live_mode=args.live,
        scale=args.scale,
        execution_mode=args.execution,
        bar_size=args.bar_size,
    )

    print(f"\n[launch] {len(processes)} strategies running. Press Ctrl+C to stop all.\n")

    # Handle graceful shutdown
    def shutdown(signum, frame):
        print("\n[launch] Shutting down all strategies...")
        for name, proc in processes:
            if proc.poll() is None:
                print(f"  Terminating {name} (PID {proc.pid})")
                proc.terminate()
        for name, proc in processes:
            proc.wait(timeout=10)
            print(f"  {name} exited with code {proc.returncode}")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Wait for all processes
    try:
        for name, proc in processes:
            proc.wait()
            print(f"[launch] {name} exited with code {proc.returncode}")
    except KeyboardInterrupt:
        shutdown(None, None)

    return 0


if __name__ == "__main__":
    sys.exit(main())
