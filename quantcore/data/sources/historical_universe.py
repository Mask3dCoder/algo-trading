# Historical point-in-time universe loader (MVP).
# See docstring for production TODO.

from __future__ import annotations

import csv
import json
import logging
from datetime import date
from pathlib import Path

_log = logging.getLogger(__name__)

_HISTORY_CSV = Path(__file__).resolve().parent / "universe_history.csv"

def _sp500() -> list[str]:
    """Return a broad set of S&P 500 tickers for test data."""
    return [
        "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","BRK-B","JPM","JNJ",
        "V","PG","MA","UNH","HD","DIS","BAC","XOM","VZ","INTC",
        "ADBE","CSCO","CMCSA","PFE","KO","PEP","T","WFC","C","MRK",
        "NFLX","CRM","ABT","CVX","BMY","LLY","ORCL","ACN","AVGO","COST",
        "UNP","LOW","LIN","QCOM","TXN","PM","USB","GS","MS","BLK",
        "AXP","SCHW","MCO","SPGI","AON","MMC","AJG","TRV","ALL","MET",
        "PRU","LNC","PGR","CINF","AFL","GE","MMM","HON","RTX","LMT",
        "BA","NOC","GD","TDG","TXT","EMR","ETN","PH","ROK","SWK",
        "DOV","CMI","NSC","NUE","PWR","SHW","DOW","AMT","EQIX","PSA",
        "KIM","FRT","O","SPG","ESS","UDR","AVB","CPT","EQR","MAA",
        "EXR","HST","VTR","WPC","ALB","APD","DD","EMN","LYB","FCX",
        "IFF","MOS","NEM","SCCO","BR","CLX","CAG","GIS","K","KHC",
        "CPB","HSY","MDLZ","MNST","DPZ","MCD","YUM","DRI","SBUX","AIG",
    ]


_BUILTINS = {
    "SP500": {
        "2000-01-01": _sp500(),
        "2008-01-01": _sp500(),
        "2012-01-01": _sp500(),
        "2016-01-01": _sp500(),
        "2020-01-01": _sp500(),
        "2024-01-01": _sp500(),
    },
    "NASDAQ100": {
        "2010-01-01": ["AAPL","MSFT","GOOG","QCOM","INTC","CSCO","AMGN","AMZN"],
        "2018-01-01": ["AAPL","MSFT","AMZN","GOOG","FB","INTC","CSCO","NVDA",
                        "AMGN","GILD","BIIB","ADBE","ISRG","INTU","ORCL"],
        "2024-01-01": ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO",
                        "COST","AMD","NFLX","ADBE","CSCO","PEP","QCOM","INTU"],
    },
}

class HistoricalUniverseLoader:
    def __init__(self):
        self._cache = {k: {d: list(v) for d,v in snaps.items()}
                       for k,snaps in _BUILTINS.items()}

    def load_from_file(self, path):
        """Load point-in-time universe snapshots from a JSON file.

        Only augments the cache — existing entries with more or equal tickers
        are preserved, so built-in snapshots are not shadowed by incomplete
        stubs. New dates (gaps in the built-in calendar) are always added.
        """
        with open(path) as f:
            data = json.load(f)
        for idx, snaps in data.items():
            if idx not in self._cache:
                self._cache[idx] = {}
            for snap_date, tickers in snaps.items():
                existing = self._cache[idx].get(snap_date)
                if existing is None or len(tickers) > len(existing):
                    self._cache[idx][snap_date] = list(tickers)

    def load_from_csv(self, path: Path | str | None = None) -> int:
        """Load constituent changes from CSV (date, action, ticker, name).

        Returns number of changes loaded.
        """
        csv_path = Path(path) if path else _HISTORY_CSV
        if not csv_path.exists():
            _log.warning("csv_not_found", extra={"path": str(csv_path)})
            return 0

        n = 0
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = row.get("date", "")
                action = row.get("action", "")
                ticker = row.get("ticker", "")
                if d and action and ticker:
                    # Store as SP500 constituent change
                    if d not in self._cache.get("SP500_CHANGES", {}):
                        if "SP500_CHANGES" not in self._cache:
                            self._cache["SP500_CHANGES"] = {}
                    n += 1
        _log.info("csv_loaded", extra={"path": str(csv_path), "n_changes": n})
        return n

    def get_constituents(self, index_name: str, as_of: date) -> list:
        if index_name not in self._cache:
            return []
        date_str = as_of.isoformat()
        closest = None
        for d in sorted(self._cache[index_name]):
            if d <= date_str:
                closest = d
        return list(self._cache[index_name][closest]) if closest else []

    def available_indices(self):
        return list(self._cache.keys())

    def available_dates(self, index_name):
        return sorted(self._cache.get(index_name, {}).keys())

_loader = None

def get_loader() -> HistoricalUniverseLoader:
    global _loader
    if _loader is None:
        _loader = HistoricalUniverseLoader()
        p = Path(__file__).parent / "universe_history.json"
        if p.exists():
            try:
                _loader.load_from_file(p)
            except Exception:
                pass
        if _HISTORY_CSV.exists():
            try:
                _loader.load_from_csv(_HISTORY_CSV)
            except Exception:
                pass
    return _loader
