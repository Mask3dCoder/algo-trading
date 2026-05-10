# Historical point-in-time universe loader (MVP).
# See docstring for production TODO.

from __future__ import annotations
import json
import logging
from datetime import date
from pathlib import Path

_log = logging.getLogger(__name__)

_BUILTINS = {
    "SP500": {
        "2008-01-01": ["AAPL","MSFT","GE","C","BAC","XOM","CVX","JNJ","PG",
                        "WFC","JPM","AIG","IBM","T","VZ","HD","WMT","MCD",
                        "PFE","MRK","ABT","AMGN","BMY","LLY","KO","PEP"],
        "2012-01-01": ["AAPL","MSFT","XOM","GE","CVX","PFE","KO","PEP","PM",
                        "T","VZ","ORCL","CSCO","INTC","QCOM","GOOG","AMZN",
                        "DIS","BAC","ABT","MRK","ABBV","BMY","LLY","UNH"],
        "2016-01-01": ["AAPL","MSFT","XOM","JNJ","BRK-B","WFC","JPM","GE",
                        "CVX","PG","T","VZ","GOOG","GOOGL","AMZN","FB",
                        "DIS","PFE","BAC","ABT","KO","PEP","MRK","ABBV"],
        "2020-01-01": ["AAPL","MSFT","AMZN","GOOGL","FB","BRK-B","JPM","JNJ",
                        "V","PG","NVDA","HD","MA","UNH","BAC","VZ","ADBE",
                        "T","NFLX","CSCO","XOM","WMT","CRM","ABBV","ACN"],
        "2024-01-01": ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","BRK-B","LLY",
                        "AVGO","JPM","TSLA","UNH","V","XOM","MA","JNJ","PG",
                        "HD","ORCL","COST","MRK","ABBV","CVX","BAC","KO"],
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
        with open(path) as f:
            data = json.load(f)
        for idx,snaps in data.items():
            if idx not in self._cache:
                self._cache[idx] = {}
            self._cache[idx].update(snaps)

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
    return _loader
