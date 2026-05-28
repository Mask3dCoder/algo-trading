#!/usr/bin/env python3
"""Manual IBKR connection smoketest.

Verifies connectivity to TWS/IB Gateway, fetches account info
and market prices, then optionally places and cancels a 1-share
paper order.

Usage:
    python scripts/test_ibkr_connection.py              # connect + account + prices
    python scripts/test_ibkr_connection.py --place-order  # also place + cancel a test order

Environment variables:
    IBKR_HOST       (default: 127.0.0.1)
    IBKR_PORT       (default: 7497 — TWS Paper)
    IBKR_CLIENT_ID  (default: 1)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ibkr_smoketest")

_PORT_MODES = {
    7497: "TWS Paper",
    7496: "TWS Live",
    4002: "Gateway Paper",
    4001: "Gateway Live",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="IBKR connection smoketest")
    parser.add_argument(
        "--place-order", action="store_true",
        help="Place a 1-share market order on AAPL (paper only) then cancel it.",
    )
    args = parser.parse_args()

    host = os.environ.get("IBKR_HOST", "127.0.0.1")
    port = int(os.environ.get("IBKR_PORT", "7497"))
    client_id = int(os.environ.get("IBKR_CLIENT_ID", "1"))
    mode = _PORT_MODES.get(port, f"Unknown ({port})")
    is_paper = port in (7497, 4002)

    print("=" * 60)
    print("  IBKR Connection Smoketest")
    print("=" * 60)
    print(f"  Host:      {host}")
    print(f"  Port:      {port} ({mode})")
    print(f"  Client ID: {client_id}")
    print(f"  Mode:      {'PAPER' if is_paper else 'LIVE — REAL MONEY'}")
    print()

    if not is_paper and args.place_order:
        print("ERROR: --place-order is only allowed on paper trading ports (7497/4002).")
        print("Set IBKR_PORT=7497 for TWS Paper or IBKR_PORT=4002 for Gateway Paper.")
        return 1

    # --- Step 1: Connect ---
    print("[1/4] Connecting to IBKR...")
    try:
        from quantcore.live.ibkr_gateway import IBKRExecutionGateway

        gw = IBKRExecutionGateway(host=host, port=port, client_id=client_id)
        gw.connect()
        print(f"  OK — connected (account={gw._broker.account_id})")
    except Exception as e:
        print(f"  FAIL — {e}")
        return 1

    # --- Step 2: Account info ---
    print("[2/4] Fetching account info...")
    try:
        acct = gw.get_account()
        print(f"  Cash:            ${acct.get('cash', 0):>14,.2f}")
        print(f"  Equity:          ${acct.get('equity', 0):>14,.2f}")
        print(f"  Portfolio value: ${acct.get('portfolio_value', 0):>14,.2f}")
        print(f"  Buying power:    ${acct.get('buying_power', 0):>14,.2f}")
    except Exception as e:
        print(f"  FAIL — {e}")
        gw.disconnect()
        return 1

    # --- Step 3: Market data ---
    print("[3/4] Fetching market prices (AAPL, MSFT, SPY)...")
    try:
        prices = gw.get_current_prices(["AAPL", "MSFT", "SPY"])
        for sym, px in prices.items():
            print(f"  {sym}: ${px:,.2f}")
        if not prices:
            print("  (no prices returned — market may be closed)")
    except Exception as e:
        print(f"  FAIL — {e}")
        gw.disconnect()
        return 1

    # --- Step 4: Order test (optional) ---
    if args.place_order:
        print("[4/4] Placing 1-share market order on AAPL...")
        try:
            from quantcore.core.types import Order, Side, OrderType

            order = Order(
                symbol="AAPL",
                side=Side.BUY,
                type=OrderType.MARKET,
                quantity=1,
            )
            order_id = gw.submit_order(order)
            print(f"  Submitted — broker order ID: {order_id}")

            time.sleep(2)

            cancelled = gw.cancel_order(order_id)
            print(f"  Cancel requested: {'OK' if cancelled else 'FAIL'}")

            positions = gw.get_positions()
            print(f"  Current positions: {len(positions)}")
        except Exception as e:
            print(f"  FAIL — {e}")
            gw.disconnect()
            return 1
    else:
        print("[4/4] Order test skipped (use --place-order to enable)")

    # --- Cleanup ---
    gw.disconnect()
    print()
    print("=" * 60)
    print("  Smoketest PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
