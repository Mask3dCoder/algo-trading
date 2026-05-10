#!/usr/bin/env python
"""
QuantCore — IBKR Connection Diagnostic
Run this to verify TWS is configured correctly and QuantCore can connect.
TWS must be running and logged in before running this script.
"""
import os
import sys
import time
from pathlib import Path

# Ensure quantcore is importable from the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv()

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def check(label, fn):
    try:
        result = fn()
        print(f"  {PASS} {label}: {result}")
        return True
    except Exception as e:
        print(f"  {FAIL} {label}: {e}")
        return False


def main():
    print()
    print("QuantCore -- IBKR Diagnostic")
    print("=" * 50)
    failures = 0

    # Check 1: Environment variables
    print("\n[1] Environment Variables")
    host = os.environ.get("IBKR_HOST", "127.0.0.1")
    port = int(os.environ.get("IBKR_PORT", "7497"))
    client_id = int(os.environ.get("IBKR_CLIENT_ID", "1"))

    port_modes = {7497: "TWS Paper", 7496: "TWS Live", 4002: "Gateway Paper", 4001: "Gateway Live"}
    mode = port_modes.get(port, "Unknown")
    is_paper = port in (7497, 4002)

    print(f"  {PASS} IBKR_HOST: {host}")
    print(f"  {PASS} IBKR_PORT: {port} ({mode})")
    print(f"  {PASS} IBKR_CLIENT_ID: {client_id}")

    if not is_paper:
        print(f"  {WARN} WARNING: Port {port} indicates LIVE trading -- real money at risk")

    # Check 2: TWS Connection
    print("\n[2] TWS Connection")
    try:
        import ib_insync
        print(f"  {PASS} ib_insync version: {ib_insync.__version__}")
    except ImportError:
        print(f"  {FAIL} ib_insync not installed: pip install ib_insync")
        sys.exit(1)

    from ib_insync import IB
    ib = IB()

    try:
        ib.connect(host, port, clientId=client_id, timeout=10)
        print(f"  {PASS} Connected to TWS at {host}:{port}")
    except Exception as e:
        print(f"  {FAIL} Cannot connect to TWS: {e}")
        print(f"       Is TWS running and logged in?")
        print(f"       Is API enabled? (Edit > Global Configuration > API > Settings)")
        print(f"       Is socket port set to {port}?")
        sys.exit(1)

    # Check 3: Account
    print("\n[3] Account")
    accounts = ib.managedAccounts()
    for acct in accounts:
        prefix = acct[:2]
        acct_type = "PAPER" if prefix == "DU" else "LIVE"
        flag = PASS if (acct_type == "PAPER") == is_paper else WARN
        print(f"  {flag} Account: {acct} ({acct_type})")
        if acct_type == "LIVE" and is_paper:
            print(f"       {WARN} Live account detected on paper port -- check TWS login mode")
            failures += 1

    summary = ib.accountSummary()
    for item in summary:
        if item.tag in ("NetLiquidation", "NetLiquidationByCurrency") and item.currency == "BASE":
            print(f"  {PASS} Portfolio value: {float(item.value):,.2f} (base currency)")
        if item.tag == "AvailableFunds" and item.currency == "BASE":
            print(f"  {PASS} Available funds: {float(item.value):,.2f} (base currency)")
    # Per-currency breakdown
    for item in summary:
        if item.tag == "NetLiquidationByCurrency" and item.currency == "USD":
            print(f"         USD-denominated: ${float(item.value):,.2f}")
        if item.tag == "NetLiquidationByCurrency" and item.currency == "GBP":
            print(f"         GBP-denominated: £{float(item.value):,.2f}")

    # Check 4: Market data (basic test)
    print("\n[4] Market Data")
    try:
        from ib_insync import Stock
        contract = Stock("SPY", "SMART", "USD")
        ib.qualifyContracts(contract)
        ticker = ib.reqMktData(contract, "", True, False)
        ib.sleep(2)
        if ticker.last and ticker.last > 0:
            print(f"  {PASS} SPY last price: ${ticker.last:.2f}")
        elif ticker.close and ticker.close > 0:
            print(f"  {WARN} SPY close price: ${ticker.close:.2f} (delayed or market closed)")
        else:
            print(f"  {WARN} SPY price not available (market may be closed or no data subscription)")
        ib.cancelMktData(contract)
    except Exception as e:
        print(f"  {WARN} Market data check failed: {e}")
        print(f"       This may be normal if market is closed or you lack data subscriptions")
        print(f"       Use --data-source alpaca for free real-time data")

    # Check 5: Order permissions (read-only check)
    print("\n[5] Order Permissions")
    open_orders = ib.openOrders()
    print(f"  {PASS} Can read orders: {len(open_orders)} open orders")
    positions = ib.positions()
    for p in positions:
        if float(p.position) != 0:
            print(f"         Position: {p.contract.symbol} x{p.position} @ ${p.avgCost:.2f}")
    print(f"  {PASS} Can read positions: {len(positions)} positions")

    if not ib.client.isConnected():
        print(f"  {FAIL} Connection lost during checks")
        failures += 1

    # Check 6: Latency
    print("\n[6] API Latency")
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        ib.accountSummary()
        times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    flag = PASS if avg < 100 else WARN
    print(f"  {flag} Round-trip latency (avg of 3): {avg:.1f}ms")

    # Check 7: QuantCore IBKR broker (import + instantiate)
    print("\n[7] QuantCore IBKR Broker")
    try:
        from quantcore.brokers.ibkr_broker import IBKRBroker
        broker = IBKRBroker(host=host, port=port, client_id=(client_id + 10))
        print(f"  {PASS} IBKRBroker instantiated")
        print(f"       Mode: {'PAPER' if broker.is_paper else 'LIVE'}")
        print(f"       Host: {broker.host}:{broker.port}")
    except Exception as e:
        print(f"  {FAIL} IBKRBroker failed: {e}")
        failures += 1

    ib.disconnect()
    print(f"  {PASS} Disconnected cleanly")

    # Summary
    print("\n" + "=" * 50)
    if failures == 0:
        print(f"{PASS} All checks passed -- QuantCore can connect to IBKR")
        mode_str = "PAPER" if is_paper else "LIVE"
        print(f"   Mode: {mode_str} ({mode})")
        print(f"   Recommended run command:")
        print(f"   python scripts/paper_trade.py --duration 5d --strategy momentum --data-source alpaca --broker ibkr")
    else:
        print(f"{FAIL} {failures} check(s) failed -- resolve before trading")
        sys.exit(1)
    print()


if __name__ == "__main__":
    main()
