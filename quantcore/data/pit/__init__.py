"""Point-in-Time PIT database package.

The lightweight generic ``quantcore.data.point_in_time.PointInTimeStore``
implements the case-wide ``(as_of, valid_from)`` invariant for in-memory
feature data and is the right tool for unit tests and small ingest jobs.

This package is the **SEC EDGAR-XBRL specific, PostgreSQL-bound counterpart**.
It is designed for backtests that need to eliminate *lookahead bias* against
SEC filings and their amendments at microsecond precision.
"""

__all__ = [
    "Company",
    "CompaniesIngestor",
    "DDL_STATEMENTS",
    "FactObservation",
    "FactsIngestor",
    "Filing",
    "FilingsIngestor",
    "IngestConfig",
    "PITQuery",
    "RestatementChain",
    "SECClient",
    "XBRLConcept",
    "create_all",
    "create_engine_from_url",
    "run_ddl",
]


def __getattr__(name: str):
    if name in __all__:
        from . import blueprint

        return getattr(blueprint, name)
    raise AttributeError(f"module 'quantcore.data.pit' has no attribute {name!r}")
