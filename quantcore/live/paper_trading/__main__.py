"""Entry point for `python -m quantcore.live.paper_trading`.

Delegates to the runner CLI::

    python -m quantcore.live.paper_trading --dry-run
    python -m quantcore.live.paper_trading --config config/paper_trading.yaml
"""
from __future__ import annotations

import sys

# Re-export runner CLI so -m works
from quantcore.live.paper_trading.runner import main

if __name__ == "__main__":
    sys.exit(main())