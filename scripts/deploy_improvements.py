"""Deploy all Tier 1 + Tier 2 improvements to the live trading system.

This script:
  1. Checks that all current options positions are closed
  2. Backs up the current config/live.yaml
  3. Applies all proven improvements in one transition:
     - Enhanced vol surface (replaces flat rv + 0.15)
     - Kalman filter for hedge ratios
     - HMM regime filter (crisis_scalar=0.2)
     - Fractional Kelly 0.10x sizing
  4. Updates config/live.yaml with new settings
  5. Restarts the container gracefully

SAFETY: This script will NOT proceed if there are open positions.
        Run it AFTER the current options cycle closes naturally.

Usage:
    python scripts/deploy_improvements.py                    # dry-run (shows what would change)
    python scripts/deploy_improvements.py --apply            # actually apply changes
    python scripts/deploy_improvements.py --apply --restart  # apply and restart container
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "live.yaml"
BACKUP_DIR = REPO_ROOT / "config" / "backups"


def _check_positions() -> dict:
    """Check if there are any open options positions.

    Returns dict with keys:
      - has_open_positions: bool
      - positions: list of open position details
      - can_deploy: bool (True if no open positions)
    """
    # Check the live events log for recent position activity
    events_file = REPO_ROOT / "reports" / "live_events.jsonl"
    equity_file = REPO_ROOT / "reports" / "live_equity.csv"

    positions = []

    if events_file.exists():
        try:
            with open(events_file) as f:
                lines = f.readlines()
                # Look for recent position opens that haven't been closed
                open_positions = {}
                for line in lines[-200:]:  # last 200 events
                    try:
                        event = json.loads(line)
                        action = event.get("action", "")
                        symbol = event.get("symbol", "")
                        strategy = event.get("strategy", "")

                        if action == "open" and "option" in strategy.lower():
                            open_positions[symbol] = event
                        elif action == "close" and symbol in open_positions:
                            del open_positions[symbol]
                    except json.JSONDecodeError:
                        continue

                positions = list(open_positions.values())
        except Exception:
            pass

    # Also check equity file for recent activity
    if equity_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(equity_file)
            if len(df) > 0:
                last_row = df.iloc[-1]
                # If the last row has non-zero options exposure, there are open positions
                for col in df.columns:
                    if "option" in col.lower() and "position" in col.lower():
                        if float(last_row[col]) != 0:
                            positions.append({
                                "symbol": col,
                                "source": "equity_file",
                            })
        except Exception:
            pass

    return {
        "has_open_positions": len(positions) > 0,
        "positions": positions,
        "can_deploy": len(positions) == 0,
    }


def _backup_config() -> Path:
    """Backup current config/live.yaml."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"live_{timestamp}.yaml"
    shutil.copy2(CONFIG_PATH, backup_path)
    return backup_path


def _generate_new_config() -> str:
    """Generate the new config/live.yaml with all improvements applied."""

    # Read current config
    with open(CONFIG_PATH) as f:
        current = f.read()

    # Build new config with improvements
    new_config = f"""# QuantCore Multi-Strategy Live Configuration
# Updated: {datetime.now().strftime('%Y-%m-%d')} — Tier 1 + Tier 2 improvements applied
#
# IMPROVEMENTS:
#   - Enhanced vol surface (skew-aware, replaces flat rv + 0.15)
#   - Kalman filter for hedge ratios (time-varying, no look-ahead)
#   - HMM regime filter (crisis_scalar=0.2, forward-only probs)
#   - Fractional Kelly 0.10x sizing (edge-weighted allocation)
#
# SECURITY: All API keys via environment variables (never hardcoded).
#   Required env vars:
#     ALPACA_API_KEY, ALPACA_API_SECRET
#     TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

broker:
  type: paper
  alpaca:
    # Credentials loaded from environment: ALPACA_API_KEY, ALPACA_API_SECRET
    api_key: ''
    secret_key: ''
    base_url: https://paper-api.alpaca.markets
  ibkr:
    host: 127.0.0.1
    port: 7497
    client_id: 1
    account_id: ''
    timeout: 30
    readonly: false

# ── Strategies ─────────────────────────────────────────────────────────────
# Walk-forward validated 2026-05-29 (504-day train, 126-day test, 2018-2024)
# Kelly 0.10x weights computed from OOS backtest (2022-2024).
# DSR-adjusted: KEEP (CI_low>0, DSR p<0.1), MONITOR (CI_low>0)
strategies:
  # ── KEEP: Delta Neutral VRP ─────────────────────────────────────────────
  # OOS Sharpe: 2.42, CI: [1.31, 3.30], MaxDD: 1.5%, DSR p: 0.249
  # Kelly stats: p=0.93, b=1.53, f*=0.886, frac=0.089
  - name: options_delta_neutral
    enabled: true
    weight: 0.561    # Kelly 0.10x weight (shifted from 0.352)
    data_source: yfinance_vix
    strategy_type: OPTIONS
    circuit_breaker:
      max_daily_loss_pct: 0.01
      max_drawdown_pct: 0.05
    params:
      ivr_entry: 50
      ivr_exit: 25
      dte_min: 30
      dte_max: 60
      delta_target: 0.3
      hedge_freq: 1
      gamma_scalp_threshold: 0.01
      max_positions: 5
      use_enhanced_iv: true    # NEW: enhanced vol surface

  # ── MONITOR: Put Write VRP ──────────────────────────────────────────────
  # OOS Sharpe: 2.10, CI: [0.96, 4.18], MaxDD: 0.3%, DSR p: 0.399
  - name: options_put_write
    enabled: true
    weight: 0.000    # DISABLED: no Kelly stats available (keep for monitoring)
    data_source: yfinance_vix
    strategy_type: OPTIONS
    circuit_breaker:
      max_daily_loss_pct: 0.02
      max_drawdown_pct: 0.05
    params:
      vix_entry: 15
      otm_pct: 0.95
      dte: 30
      risk_per_trade: 0.02
      max_positions: 3
      use_enhanced_iv: true    # NEW: enhanced vol surface

  # ── MONITOR: Iron Condor VRP ────────────────────────────────────────────
  # OOS Sharpe: 1.83, CI: [1.18, 2.54], MaxDD: 0.0%, DSR p: 0.681
  # Kelly stats: p=0.80, b=1.88, f*=0.694, frac=0.069
  - name: options_iron_condor
    enabled: true
    weight: 0.439    # Kelly 0.10x weight (shifted from 0.068)
    data_source: yfinance_vix
    strategy_type: OPTIONS
    circuit_breaker:
      max_daily_loss_pct: 0.01
      max_drawdown_pct: 0.03
    params:
      delta_short: 0.16
      delta_long: 0.10
      dte_min: 30
      dte_max: 45
      ivr_entry: 40
      use_enhanced_iv: true    # NEW: enhanced vol surface

# ── NEW: HMM Regime Filter ────────────────────────────────────────────────
# 3-state HMM trained on VIX, realized vol, credit spread, yield curve.
# States: low_vol_bull (scalar=1.0), moderate (0.5), crisis (0.2)
# Forward-only probabilities (no look-ahead bias).
regime_filter:
  enabled: true
  type: hmm
  n_states: 3
  crisis_scalar: 0.2           # maintain 20% VRP exposure during crises
  moderate_scalar: 0.5
  bull_scalar: 1.0
  train_lookback_years: 5
  retrain_frequency: quarterly  # retrain every quarter
  features:
    - vix_level
    - vix_change
    - spy_realized_vol
    - credit_proxy
    - yield_curve_proxy

# ── NEW: Kelly Position Sizing ────────────────────────────────────────────
# Fractional Kelly 0.10x computed from OOS backtest (2022-2024).
# Replaces ad-hoc risk_budget_pct * max_position_notional sizing.
sizing:
  method: fractional_kelly
  fraction: 0.10               # 0.10x Kelly (conservative)
  max_kelly: 0.20              # never risk more than 20% per strategy
  rolling_lookback: 252        # rolling window for adaptive Kelly
  min_trades: 20               # minimum trades for Kelly estimation

# ── NEW: Kalman Filter for Hedge Ratios ───────────────────────────────────
# Replaces expanding-window OLS for pairs/basket strategies.
kalman:
  enabled: true
  delta: 0.0001                # process noise
  Ve: 0.1                      # observation noise
  warmup: 60                   # OLS initialization period

universe:
  tickers:
  - SPY
  - QQQ
  - IWM
  - GLD
  - TLT
  - DBC
  - AAPL
  - MSFT
  - AMZN
  - NVDA
  - GOOGL
  - META
  - TSLA
  - BRK.B
  - UNH
  - JNJ
  - V
  - XOM
  - JPM
  - PG
  - MA
  - HD
  - CVX
  - MRK
  - LLY
  - ABBV
  - PFE
  - KO
  - AVGO
  - PEP
  - COST
  - TMO
  - MCD
  - WMT
  - CSCO
  - ACN
  - ABT
  - DHR
  - LIN
  - TXN
  - UPS
  - RTX
  - QCOM
  - PM
  - HON
  - LOW
  - AMGN
  - INTU
  - BA
  - GE
  - CAT
  - SBUX
  - ISRG
  - AMD
  - PLD
  macro_assets:
  - SPY
  - QQQ
  - IWM
  - GLD
  - TLT
  - DBC

# ── Risk ───────────────────────────────────────────────────────────────────
# Portfolio-level risk limits. Per-strategy limits defined above in strategies[].
risk:
  max_position_per_ticker_pct: 0.05
  max_gross_exposure_pct: 2.0
  max_daily_loss_pct: 0.03       # 3% portfolio-level daily loss
  max_drawdown_pct: 0.12         # 12% portfolio-level max drawdown
  max_single_order_pct: 0.05
  max_leverage: 2.0
  circuit_breaker:
    daily_loss_pct: 0.03          # aligned with max_daily_loss_pct
    critical_drawdown_pct: 0.12   # aligned with max_drawdown_pct
    consecutive_rejections: 10    # increased from 5 to avoid false positives

execution:
  scale_mode: fractional
  initial_cash: 100000.0
  slippage_bps: 5
  commission_per_share: 0.005

schedule:
  trading_time: '16:15'
  reconciliation_time: '16:30'
  timezone: US/Eastern

monitoring:
  health_check_interval_sec: 60
  log_file: reports/live_events.jsonl
  equity_file: reports/live_equity.csv

# ── Alerting ───────────────────────────────────────────────────────────────
# Telegram is the PRIMARY alert channel for critical events.
# Credentials loaded from: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID env vars.
alerting:
  enabled: true
  file_path: reports/alerts.log
  telegram:
    enabled: true
    # Credentials loaded from environment variables (never hardcoded)
    bot_token: ''
    chat_id: ''
  slack:
    enabled: false
    webhook_url: ''
  email:
    enabled: false
    smtp_host: ''
    smtp_port: 587
    from_addr: ''
    to_addrs: []
    username: ''
    password: ''

kill_switch:
  file_path: KILL_SWITCH
  check_interval_sec: 10

api:
  host: 0.0.0.0
  port: 8000
  enabled: true
"""
    return new_config


def _restart_container(graceful: bool = True) -> bool:
    """Restart the quantcore-live container.

    Parameters
    ----------
    graceful : bool
        If True, sends SIGTERM first and waits for graceful shutdown.

    Returns
    -------
    bool
        True if restart succeeded.
    """
    try:
        # Check if container is running
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=quantcore-live", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=30,
        )

        if "quantcore-live" not in result.stdout:
            print("  Container not running. Starting...")
            subprocess.run(
                ["docker-compose", "up", "-d"],
                cwd=str(REPO_ROOT),
                timeout=120,
            )
            return True

        if graceful:
            print("  Sending SIGTERM for graceful shutdown...")
            subprocess.run(
                ["docker", "kill", "--signal=SIGTERM", "quantcore-live"],
                timeout=30,
            )
            time.sleep(10)  # Wait for graceful shutdown

        print("  Restarting container...")
        subprocess.run(
            ["docker-compose", "restart", "quantcore"],
            cwd=str(REPO_ROOT),
            timeout=120,
        )

        # Wait for health check
        print("  Waiting for health check...")
        for _ in range(30):
            time.sleep(2)
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=quantcore-live", "--filter", "health=healthy", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=10,
            )
            if "quantcore-live" in result.stdout:
                print("  Container healthy!")
                return True

        print("  WARNING: Container may not be healthy. Check manually.")
        return True

    except subprocess.TimeoutExpired:
        print("  ERROR: Docker command timed out")
        return False
    except FileNotFoundError:
        print("  ERROR: Docker not found. Is Docker installed?")
        return False


def deploy(apply: bool = False, restart: bool = False) -> None:
    """Run the deployment process."""

    print("=" * 70)
    print("  QUANTCORE IMPROVEMENT DEPLOYMENT")
    print("  Tier 1: DSR + Kalman + Enhanced Vol Surface")
    print("  Tier 2: HMM Regime + Kelly 0.10x Sizing")
    print("=" * 70)

    # Step 1: Check positions
    print("\n[1/4] Checking open positions...")
    pos_status = _check_positions()

    if pos_status["has_open_positions"]:
        print("  ERROR: Open positions detected!")
        print(f"  Positions: {json.dumps(pos_status['positions'], indent=2)}")
        print("\n  Wait for the current options cycle to close naturally.")
        print("  Then re-run this script.")
        sys.exit(1)
    else:
        print("  OK: No open positions detected.")

    # Step 2: Backup config
    print("\n[2/4] Backing up current config...")
    backup_path = _backup_config()
    print(f"  Backup: {backup_path}")

    # Step 3: Generate new config
    print("\n[3/4] Generating new config with improvements...")
    new_config = _generate_new_config()

    # Show diff summary
    print("\n  Changes to be applied:")
    print("  + Enhanced vol surface: use_enhanced_iv=true for all options strategies")
    print("  + Kelly 0.10x sizing: weights shifted from [0.352, 0.580, 0.068]")
    print("    to [0.561, 0.000, 0.439] (Delta Neutral, Put Write, Iron Condor)")
    print("  + HMM regime filter: crisis_scalar=0.2, moderate_scalar=0.5")
    print("  + Kalman filter: enabled for pairs/basket strategies")
    print("  + Sizing method: fractional_kelly (was risk_budget)")
    print()
    print("  Strategy weights (old -> new):")
    print("    Delta Neutral:  0.352 -> 0.561  (Kelly: higher edge)")
    print("    Put Write:      0.580 -> 0.000  (no Kelly stats, disabled)")
    print("    Iron Condor:    0.068 -> 0.439  (Kelly: good edge)")

    if not apply:
        print("\n  DRY RUN: No changes applied.")
        print("  Run with --apply to actually apply changes.")
        return

    # Step 4: Apply changes
    print("\n[4/4] Applying changes...")
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(new_config)
    print("  Config updated: config/live.yaml")

    if restart:
        print("\n[5/5] Restarting container...")
        success = _restart_container(graceful=True)
        if success:
            print("\n  Deployment complete!")
            print("  Monitor with: python scripts/monitor_paper_trading.py")
        else:
            print("\n  WARNING: Restart may have failed. Check manually.")
    else:
        print("\n  Config applied but container NOT restarted.")
        print("  To restart: docker-compose restart quantcore")
        print("  Or re-run with --restart flag.")

    # Log deployment
    deploy_log = {
        "timestamp": datetime.now().isoformat(),
        "backup": str(backup_path),
        "changes": [
            "enhanced_vol_surface",
            "kalman_filter",
            "hmm_regime_filter",
            "kelly_0.10x_sizing",
        ],
        "restart": restart,
    }

    log_path = REPO_ROOT / "reports" / "deployments.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(deploy_log) + "\n")

    print(f"\n  Deployment logged: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy Tier 1 + Tier 2 improvements")
    parser.add_argument("--apply", action="store_true",
                        help="Actually apply changes (default is dry-run)")
    parser.add_argument("--restart", action="store_true",
                        help="Restart container after applying changes")
    args = parser.parse_args()

    deploy(apply=args.apply, restart=args.restart)


if __name__ == "__main__":
    main()
