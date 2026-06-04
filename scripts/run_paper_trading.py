#!/usr/bin/env python3
"""Launch script for QuantCore paper/live trading.

DEPRECATED - this script is kept for local development only.

For production (and paper trading validation), use Docker::

    # Build and start (reads config/live.yaml, routes to ProductionRunner):
    docker compose up -d

    # Verify health after 60s:
    curl http://localhost:8000/health

    # Watch logs:
    docker logs quantcore-live -f

    # Check fills after 16:30 ET:
    python scripts/monitor_paper_trading.py

    # Stop:
    docker compose down

The container reads the following environment variables (set in .env):
    ALPACA_API_KEY, ALPACA_API_SECRET    - Alpaca paper credentials
    ALPACA_BASE_URL                      - https://paper-api.alpaca.markets (default)
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID - alert channel

Docker configuration is in docker-compose.yml; environment defaults are in
.env (copy from .env.example if needed).

Strategies enabled for paper trading are in config/live.yaml under
strategies: and are the three options VRP strategies:
    options_delta_neutral, options_put_write, options_iron_condor

NO new strategies may be added during the 5-10 day paper validation run.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "DEPRECATED: use 'docker compose up -d' instead of this script.\n"
        "See docstring or README for environment setup.\n"
        "\n"
        "Quick start:\n"
        "  cp .env.example .env     # set ALPACA_API_KEY / ALPACA_API_SECRET\n"
        "  docker compose up -d     # starts ProductionRunner\n"
        "  curl http://localhost:8000/health\n"
        "  docker logs quantcore-live -f\n"
        "\n"
        "For a one-shot connectivity check:\n"
        "  python -m quantcore.live.production_runner --dry-run\n",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())