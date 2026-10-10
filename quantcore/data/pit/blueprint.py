"""SEC EDGAR XBRL Point-in-Time database blueprint.

This module is the **SEC-specific, PostgreSQL-bound** counterpart to the
lightweight generic ``quantcore.data.point_in_time.PointInTimeStore``.

It implements the core invariant required to eliminate lookahead bias:

    Every fact carries both
      1. validity period: fiscal quarter/year the data represents
      2. publication timestamp: exact microsecond the filing hit EDGAR

and restatements are handled as append-only history, so a backtest running at
``asof_ts`` sees exactly what was available to the market at that time.

Usage
-----
```bash
python -m quantcore.data.pit.blueprint sync --cik 320193 --ticker AAPL
python -m quantcore.data.pit.blueprint pit --cik 320193 --concept us-gaap:Revenues --asof 2023-12-31
```
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pandas as pd
import requests
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session

SEC_USER_AGENT = "QuantCore PIT research bot contact:research@example.com"

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{:010d}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{:010d}.json"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_RATE_LIMIT_SECONDS = 0.15


class Base(DeclarativeBase):
    """Base class for all PIT database models."""


class Company(Base):
    """Static company registry keyed by CIK."""

    __tablename__ = "companies"

    cik = Column(String(10), primary_key=True)
    name = Column(String(255), nullable=False)
    ticker = Column(String(20), index=True)
    sic = Column(String(10), index=True)
    sic_description = Column(String(255))
    state_of_incorporation = Column(String(2))
    fiscal_year_end = Column(String(5))
    last_synced_at = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Filing(Base):
    """One row per EDGAR submission.

    ``accepted_at`` is the publication timestamp: the instant the filing hit
    EDGAR and became available to the market.
    """

    __tablename__ = "filings"

    accession_number = Column(String(25), primary_key=True)
    cik = Column(String(10), ForeignKey("companies.cik"), nullable=False, index=True)
    form_type = Column(String(10), nullable=False, index=True)
    filed_date = Column(Date, nullable=False, index=True)
    period_of_report = Column(Date, index=True)
    accepted_at = Column(DateTime(timezone=True), nullable=False, index=True)
    primary_document = Column(String(255))
    is_amendment = Column(Boolean, nullable=False, default=False)
    amends_accession = Column(String(25), ForeignKey("filings.accession_number"))
    raw_submission = Column(JSONB)
    ingested_at = Column(DateTime(timezone=True), server_default=func.now())


class XBRLConcept(Base):
    """XBRL taxonomy concept registry."""

    __tablename__ = "concepts"

    namespace = Column(String(64), primary_key=True)
    qname = Column(String(255), primary_key=True)
    label = Column(String(500))
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class FactObservation(Base):
    """One as-reported fact observation.

    PIT invariant:
        backtest at ``asof_ts`` sees rows where
        ``filing_published_at <= asof_ts`` and
        ``period_end <= asof_ts``.

    Restatements are append-only: a 10-K/A or 10-Q/A creates new rows with
    ``is_restatement=True`` and later supersedes the prior observation.
    """

    __tablename__ = "facts_observations"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    cik = Column(String(10), ForeignKey("companies.cik"), nullable=False, index=True)
    concept = Column(String(255), nullable=False, index=True)
    period_type = Column(String(16), nullable=False)
    period_start = Column(Date)
    period_end = Column(Date, nullable=False, index=True)
    unit = Column(String(32), nullable=False)
    value_numeric = Column(Numeric(38, 6))
    value_raw_text = Column(Text)
    decimals = Column(Integer)
    scale = Column(Integer)
    context_id = Column(String(128))
    accession_number = Column(String(25), ForeignKey("filings.accession_number"), index=True)
    filing_published_at = Column(DateTime(timezone=True), nullable=False, index=True)
    is_restatement = Column(Boolean, nullable=False, default=False)
    is_current = Column(Boolean, nullable=False, default=True, index=True)
    superseded_by_filing_published_at = Column(DateTime(timezone=True))
    ingested_at = Column(DateTime(timezone=True), server_default=func.now())


class RestatementChain(Base):
    """Audit trail of restatement supersessions."""

    __tablename__ = "restatement_chain"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    cik = Column(String(10), ForeignKey("companies.cik"), nullable=False, index=True)
    concept = Column(String(255), nullable=False, index=True)
    superseded_accession = Column(String(25), ForeignKey("filings.accession_number"), nullable=False)
    superseding_accession = Column(String(25), ForeignKey("filings.accession_number"), nullable=False)
    original_period_end = Column(Date, nullable=False)
    new_value = Column(Numeric(38, 6))
    superseded_at = Column(DateTime(timezone=True), nullable=False)
    detected_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint(
            "superseded_accession",
            "superseding_accession",
            "concept",
            "original_period_end",
            name="uq_restatement_chain",
        ),
    )


class PITQueryLog(Base):
    """Optional observability table for PIT query auditing."""

    __tablename__ = "pit_query_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    query_label = Column(String(128))
    asof_ts = Column(DateTime(timezone=True), nullable=False)
    rows_returned = Column(Integer)
    cik = Column(String(10), index=True)
    concept = Column(String(255), index=True)
    executed_at = Column(DateTime(timezone=True), server_default=func.now())
    lag_ms = Column(Integer)


DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS companies (
        cik VARCHAR(10) PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        ticker VARCHAR(20),
        sic VARCHAR(10),
        sic_description VARCHAR(255),
        state_of_incorporation VARCHAR(2),
        fiscal_year_end VARCHAR(5),
        last_synced_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS filings (
        accession_number VARCHAR(25) PRIMARY KEY,
        cik VARCHAR(10) NOT NULL REFERENCES companies(cik),
        form_type VARCHAR(10) NOT NULL,
        filed_date DATE NOT NULL,
        period_of_report DATE,
        accepted_at TIMESTAMPTZ NOT NULL,
        primary_document VARCHAR(255),
        is_amendment BOOLEAN NOT NULL DEFAULT FALSE,
        amends_accession VARCHAR(25) REFERENCES filings(accession_number),
        raw_submission JSONB,
        ingested_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS concepts (
        namespace VARCHAR(64),
        qname VARCHAR(255),
        label VARCHAR(500),
        description TEXT,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (namespace, qname)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS facts_observations (
        id BIGSERIAL PRIMARY KEY,
        cik VARCHAR(10) NOT NULL REFERENCES companies(cik),
        concept VARCHAR(255) NOT NULL,
        period_type VARCHAR(16) NOT NULL,
        period_start DATE,
        period_end DATE NOT NULL,
        unit VARCHAR(32) NOT NULL,
        value_numeric NUMERIC(38, 6),
        value_raw_text TEXT,
        decimals INTEGER,
        scale INTEGER,
        context_id VARCHAR(128),
        accession_number VARCHAR(25) REFERENCES filings(accession_number),
        filing_published_at TIMESTAMPTZ NOT NULL,
        is_restatement BOOLEAN NOT NULL DEFAULT FALSE,
        is_current BOOLEAN NOT NULL DEFAULT TRUE,
        superseded_by_filing_published_at TIMESTAMPTZ,
        ingested_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS restatement_chain (
        id BIGSERIAL PRIMARY KEY,
        cik VARCHAR(10) NOT NULL REFERENCES companies(cik),
        concept VARCHAR(255) NOT NULL,
        superseded_accession VARCHAR(25) NOT NULL REFERENCES filings(accession_number),
        superseding_accession VARCHAR(25) NOT NULL REFERENCES filings(accession_number),
        original_period_end DATE NOT NULL,
        new_value NUMERIC(38, 6),
        superseded_at TIMESTAMPTZ NOT NULL,
        detected_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT uq_restatement_chain UNIQUE (
            superseded_accession,
            superseding_accession,
            concept,
            original_period_end
        )
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS pit_query_log (
        id BIGSERIAL PRIMARY KEY,
        query_label VARCHAR(128),
        asof_ts TIMESTAMPTZ NOT NULL,
        rows_returned INTEGER,
        cik VARCHAR(10),
        concept VARCHAR(255),
        executed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        lag_ms INTEGER
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_companies_ticker ON companies(ticker);
    CREATE INDEX IF NOT EXISTS idx_filings_cik ON filings(cik);
    CREATE INDEX IF NOT EXISTS idx_filings_form_type ON filings(form_type);
    CREATE INDEX IF NOT EXISTS idx_filings_accepted_at ON filings(accepted_at);
    CREATE INDEX IF NOT EXISTS idx_filings_amendment ON filings(is_amendment);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_pit_lookup
        ON facts_observations (cik, concept, period_end, filing_published_at DESC);
    CREATE INDEX IF NOT EXISTS idx_pit_current
        ON facts_observations (cik, concept, period_end)
        WHERE is_current = TRUE;
    CREATE INDEX IF NOT EXISTS idx_pit_filing_published_at
        ON facts_observations (filing_published_at);
    CREATE INDEX IF NOT EXISTS idx_pit_accession
        ON facts_observations (accession_number);
    """,
]


@dataclass(frozen=True)
class IngestConfig:
    """Runtime configuration for the SEC ingest pipeline."""

    database_url: str = "postgresql+psycopg2://quantcore:quantcore@localhost:5432/quantcore"
    rate_limit_seconds: float = SEC_RATE_LIMIT_SECONDS
    user_agent: str = SEC_USER_AGENT
    chunk_size: int = 1000


def create_engine_from_url(database_url: str | None = None, **kwargs: Any) -> Engine:
    """Create a PostgreSQL engine with sane defaults."""

    url = database_url or os.environ.get("PIT_DATABASE_URL") or IngestConfig().database_url
    return create_engine(url, pool_pre_ping=True, future=True, **kwargs)


def create_all(engine: Engine) -> None:
    """Create the PIT schema from SQLAlchemy ORM metadata."""

    Base.metadata.create_all(engine)


def run_ddl(engine: Engine) -> None:
    """Create the PIT schema from explicit PostgreSQL DDL statements."""

    with engine.begin() as conn:
        for statement in DDL_STATEMENTS:
            conn.execute(text(statement))


def parse_sec_datetime(value: str) -> datetime:
    """Parse SEC timestamp strings into timezone-aware UTC datetimes."""

    if not value:
        raise ValueError("Missing SEC timestamp")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def clean_cik(cik: int | str) -> str:
    """Return a zero-padded 10-digit CIK string."""

    return f"{int(cik):010d}"


def accession_nodashes(accession: str) -> str:
    """Convert SEC accession number to archive path component."""

    return accession.replace("-", "")


def is_amendment(form_type: str) -> bool:
    """Return True if the form is an amendment."""

    return form_type.upper().endswith("/A")


class SECRateLimiter:
    """Simple sliding-window rate limiter for SEC fair-access compliance."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self.last_request_at = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self.last_request_at
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_at = time.monotonic()


class SECClient:
    """Thin SEC HTTP client with rate limiting and user-agent headers."""

    def __init__(self, user_agent: str = SEC_USER_AGENT, rate_limit_seconds: float = SEC_RATE_LIMIT_SECONDS) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov",
        })
        self.rate_limiter = SECRateLimiter(rate_limit_seconds)

    def get_json(self, url: str) -> dict[str, Any]:
        self.rate_limiter.wait()
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self.session.close()


class CompaniesIngestor:
    """Ingest SEC company ticker registry."""

    def __init__(self, client: SECClient) -> None:
        self.client = client

    def ingest(self, session: Session) -> None:
        payload = self.client.get_json(COMPANY_TICKERS_URL)
        rows = []
        for item in payload.values():
            rows.append({
                "cik": clean_cik(item["cik_str"]),
                "name": item["title"],
                "ticker": item["ticker"],
            })

        if rows:
            session.execute(
                insert(Company),
                rows,
            ).on_conflict_do_update(
                index_elements=["cik"],
                set_={
                    "name": insert(Company).c.name,
                    "ticker": insert(Company).c.ticker,
                    "last_synced_at": func.now(),
                },
            )
            session.commit()


class FilingsIngestor:
    """Ingest SEC Submissions API payload into ``filings``."""

    def __init__(self, client: SECClient) -> None:
        self.client = client

    def ingest_cik(self, session: Session, cik: int | str) -> None:
        cik_padded = clean_cik(cik)
        payloads = self._fetch_all_submission_pages(cik_padded)

        rows = []
        for payload in payloads:
            for filing in payload["filings"]["recent"]:
                accession = filing["accessionNumber"]
                form_type = filing["form"]
                rows.append({
                    "accession_number": accession,
                    "cik": cik_padded,
                    "form_type": form_type,
                    "filed_date": filing["filingDate"],
                    "period_of_report": filing.get("reportDate"),
                    "accepted_at": parse_sec_datetime(filing["acceptedAt"]),
                    "primary_document": filing.get("primaryDocument"),
                    "is_amendment": is_amendment(form_type),
                    "raw_submission": filing,
                })

        if rows:
            session.execute(
                insert(Filing),
                rows,
            ).on_conflict_do_update(
                index_elements=["accession_number"],
                set_={
                    "form_type": insert(Filing).c.form_type,
                    "filed_date": insert(Filing).c.filed_date,
                    "period_of_report": insert(Filing).c.period_of_report,
                    "accepted_at": insert(Filing).c.accepted_at,
                    "primary_document": insert(Filing).c.primary_document,
                    "is_amendment": insert(Filing).c.is_amendment,
                    "raw_submission": insert(Filing).c.raw_submission,
                    "ingested_at": func.now(),
                },
            )
            session.commit()

    def _fetch_all_submission_pages(self, cik_padded: str) -> list[dict[str, Any]]:
        """Fetch recent and historical Submissions API pages."""

        payloads = []
        primary = self.client.get_json(SUBMISSIONS_URL.format(cik_padded))
        payloads.append(primary)

        for file_meta in primary.get("filings", {}).get("files", []):
            name = file_meta["name"]
            payloads.append(self.client.get_json(f"https://data.sec.gov/submissions/{name}"))

        return payloads


class FactsIngestor:
    """Ingest SEC Company Facts JSON payload into ``facts_observations``."""

    def __init__(self, client: SECClient) -> None:
        self.client = client

    def ingest_cik(self, session: Session, cik: int | str) -> None:
        cik_padded = clean_cik(cik)
        payload = self.client.get_json(COMPANYFACTS_URL.format(cik_padded))

        concept_rows = []
        for namespace, concepts in payload["facts"].items():
            for qname, meta in concepts.items():
                concept_rows.append({
                    "namespace": namespace,
                    "qname": f"{namespace}:{qname}",
                    "label": meta.get("label"),
                    "description": meta.get("description"),
                })

        if concept_rows:
            session.execute(
                insert(XBRLConcept),
                concept_rows,
            ).on_conflict_do_update(
                index_elements=["namespace", "qname"],
                set_={
                    "label": insert(XBRLConcept).c.label,
                    "description": insert(XBRLConcept).c.description,
                },
            )

        accessions = []
        for namespace, concepts in payload["facts"].items():
            for qname, meta in concepts.items():
                for unit, observations in meta.get("units", {}).items():
                    for obs in observations:
                        accession = obs.get("accn")
                        if accession:
                            accessions.append(accession)

        filing_by_accession = self._fetch_filing_metadata(session, sorted(set(accessions)))

        fact_rows = []
        for namespace, concepts in payload["facts"].items():
            for qname, meta in concepts.items():
                for unit, observations in meta.get("units", {}).items():
                    for obs in observations:
                        accession = obs.get("accn")
                        if not accession:
                            continue

                        filing_meta = filing_by_accession.get(accession)
                        if filing_meta is None:
                            filing_published_at = datetime.combine(
                                date.fromisoformat(obs.get("filed", "1900-01-01")),
                                datetime.min.time(),
                                tzinfo=UTC,
                            )
                            is_restatement = is_amendment(obs.get("form", ""))
                        else:
                            filing_published_at, is_restatement = filing_meta

                        period_start = obs.get("start")
                        period_end = obs.get("end")
                        value = obs.get("val")

                        fact_rows.append({
                            "cik": cik_padded,
                            "concept": f"{namespace}:{qname}",
                            "period_type": "duration" if period_start else "instant",
                            "period_start": date.fromisoformat(period_start) if period_start else None,
                            "period_end": date.fromisoformat(period_end),
                            "unit": unit,
                            "value_numeric": Decimal(str(value)) if isinstance(value, (int, float)) else None,
                            "value_raw_text": str(value) if not isinstance(value, (int, float)) else None,
                            "decimals": obs.get("decimals"),
                            "scale": obs.get("scale"),
                            "context_id": obs.get("contextId"),
                            "accession_number": accession,
                            "filing_published_at": filing_published_at,
                            "is_restatement": is_restatement,
                        })

        if fact_rows:
            session.execute(insert(FactObservation), fact_rows)
            self._mark_superseded(session, cik_padded)
            session.commit()

    def _fetch_filing_metadata(self, session: Session, accessions: list[str]) -> dict[str, tuple[datetime, bool]]:
        """Bulk-load filing publication timestamps for CompanyFacts accessions."""

        metadata: dict[str, tuple[datetime, bool]] = {}
        if not accessions:
            return metadata

        for start in range(0, len(accessions), 1000):
            chunk = accessions[start:start + 1000]
            rows = session.execute(
                select(Filing.accession_number, Filing.accepted_at, Filing.is_amendment)
                .where(Filing.accession_number.in_(chunk))
            ).all()
            for accession, accepted_at, is_amendment in rows:
                metadata[accession] = (accepted_at, is_amendment)

        return metadata

    def _mark_superseded(self, session: Session, cik: str) -> None:
        """Mark older facts as superseded by newer restatements."""

        insert_audit_stmt = text(
            """
            INSERT INTO restatement_chain (
                cik,
                concept,
                superseded_accession,
                superseding_accession,
                original_period_end,
                new_value,
                superseded_at
            )
            SELECT
                older.cik,
                older.concept,
                older.accession_number,
                newer.accession_number,
                older.period_end,
                newer.value_numeric,
                newer.filing_published_at
              FROM facts_observations older
              JOIN facts_observations newer
                ON newer.cik = older.cik
               AND newer.concept = older.concept
               AND newer.period_end = older.period_end
               AND newer.filing_published_at = (
                    SELECT MIN(newer2.filing_published_at)
                      FROM facts_observations newer2
                     WHERE newer2.cik = older.cik
                       AND newer2.concept = older.concept
                       AND newer2.period_end = older.period_end
                       AND newer2.filing_published_at > older.filing_published_at
               )
             WHERE older.cik = :cik
               AND older.filing_published_at < newer.filing_published_at
            ON CONFLICT DO NOTHING;
            """
        )

        update_stmt = text(
            """
            UPDATE facts_observations older
               SET is_current = FALSE,
                   superseded_by_filing_published_at = (
                       SELECT MIN(newer.filing_published_at)
                         FROM facts_observations newer
                        WHERE newer.cik = older.cik
                          AND newer.concept = older.concept
                          AND newer.period_end = older.period_end
                          AND newer.filing_published_at > older.filing_published_at
                   )
             WHERE older.cik = :cik
               AND EXISTS (
                   SELECT 1
                     FROM facts_observations newer
                    WHERE newer.cik = older.cik
                      AND newer.concept = older.concept
                      AND newer.period_end = older.period_end
                      AND newer.filing_published_at > older.filing_published_at
               );
            """
        )

        session.execute(insert_audit_stmt, {"cik": cik})
        session.execute(update_stmt, {"cik": cik})


class PITQuery:
    """Point-in-time query facade.

    The core query uses ``DISTINCT ON`` to pick the latest observation that was
    published before ``asof_ts`` for each (cik, concept, period_end) tuple.
    """

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def get_asof(
        self,
        cik: int | str,
        concept: str,
        asof_ts: datetime,
        unit: str | None = None,
    ) -> pd.DataFrame:
        """Return the latest observation known at ``asof_ts``."""

        cik_padded = clean_cik(cik)
        asof_ts = parse_sec_datetime(asof_ts.isoformat()) if isinstance(asof_ts, str) else asof_ts

        stmt = text(
            """
            SELECT DISTINCT ON (cik, concept, period_end)
                   cik,
                   concept,
                   period_type,
                   period_start,
                   period_end,
                   unit,
                   value_numeric,
                   value_raw_text,
                   accession_number,
                   filing_published_at,
                   is_restatement,
                   is_current
              FROM facts_observations
             WHERE cik = :cik
               AND concept = :concept
               AND filing_published_at <= :asof_ts
               AND period_end <= :asof_ts::date
               AND is_current = TRUE
               AND (:unit IS NULL OR unit = :unit)
             ORDER BY cik, concept, period_end DESC, filing_published_at DESC;
            """
        )

        with self.engine.connect() as conn:
            df = pd.read_sql_query(
                stmt,
                conn,
                params={
                    "cik": cik_padded,
                    "concept": concept,
                    "asof_ts": asof_ts,
                    "unit": unit,
                },
            )

        return df

    def get_asof_many(
        self,
        cik: int | str,
        concepts: Sequence[str],
        asof_ts: datetime,
    ) -> pd.DataFrame:
        """Return the latest observations for multiple concepts at ``asof_ts``."""

        cik_padded = clean_cik(cik)
        asof_ts = parse_sec_datetime(asof_ts.isoformat()) if isinstance(asof_ts, str) else asof_ts

        stmt = text(
            """
            SELECT DISTINCT ON (cik, concept, period_end)
                   cik,
                   concept,
                   period_type,
                   period_start,
                   period_end,
                   unit,
                   value_numeric,
                   value_raw_text,
                   accession_number,
                   filing_published_at,
                   is_restatement,
                   is_current
              FROM facts_observations
             WHERE cik = :cik
               AND concept = ANY(:concepts)
               AND filing_published_at <= :asof_ts
               AND period_end <= :asof_ts::date
               AND is_current = TRUE
             ORDER BY cik, concept, period_end DESC, filing_published_at DESC;
            """
        )

        with self.engine.connect() as conn:
            df = pd.read_sql_query(
                stmt,
                conn,
                params={
                    "cik": cik_padded,
                    "concepts": list(concepts),
                    "asof_ts": asof_ts,
                },
            )

        return df

    def get_history(
        self,
        cik: int | str,
        concept: str,
        period_end: date,
    ) -> pd.DataFrame:
        """Return the full restatement history for one fact."""

        cik_padded = clean_cik(cik)
        stmt = text(
            """
            SELECT cik,
                   concept,
                   period_type,
                   period_start,
                   period_end,
                   unit,
                   value_numeric,
                   value_raw_text,
                   accession_number,
                   filing_published_at,
                   is_restatement,
                   is_current,
                   superseded_by_filing_published_at
              FROM facts_observations
             WHERE cik = :cik
               AND concept = :concept
               AND period_end = :period_end
             ORDER BY filing_published_at;
            """
        )

        with self.engine.connect() as conn:
            df = pd.read_sql_query(
                stmt,
                conn,
                params={
                    "cik": cik_padded,
                    "concept": concept,
                    "period_end": period_end,
                },
            )

        return df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SEC EDGAR XBRL Point-in-Time database")
    sub = parser.add_subparsers(dest="command", required=True)

    sync = sub.add_parser("sync", help="Ingest one CIK")
    sync.add_argument("--cik", required=True)
    sync.add_argument("--ticker", required=False)
    sync.add_argument("--database-url", default=os.environ.get("PIT_DATABASE_URL"))

    pit = sub.add_parser("pit", help="Query PIT value as of a timestamp")
    pit.add_argument("--cik", required=True)
    pit.add_argument("--concept", required=True)
    pit.add_argument("--asof", required=True)
    pit.add_argument("--unit", required=False)
    pit.add_argument("--database-url", default=os.environ.get("PIT_DATABASE_URL"))

    history = sub.add_parser("history", help="Show restatement history")
    history.add_argument("--cik", required=True)
    history.add_argument("--concept", required=True)
    history.add_argument("--period-end", required=True)
    history.add_argument("--database-url", default=os.environ.get("PIT_DATABASE_URL"))

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    engine = create_engine_from_url(args.database_url)
    create_all(engine)

    if args.command == "sync":
        client = SECClient()
        CompaniesIngestor(client).ingest(Session(engine))
        FilingsIngestor(client).ingest_cik(Session(engine), args.cik)
        FactsIngestor(client).ingest_cik(Session(engine), args.cik)
        client.close()

    elif args.command == "pit":
        query = PITQuery(engine)
        df = query.get_asof(args.cik, args.concept, args.asof, args.unit)
        print(df.to_string(index=False))

    elif args.command == "history":
        query = PITQuery(engine)
        df = query.get_history(args.cik, args.concept, date.fromisoformat(args.period_end))
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
