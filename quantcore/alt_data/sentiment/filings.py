"""SEC EDGAR filing ingestion with Loughran-McDonald sentiment dictionary."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class FilingType(str, Enum):
    K10 = "10-K"
    Q10 = "10-Q"
    K8 = "8-K"
    S1 = "S-1"


@dataclass
class FilingDocument:
    """An SEC EDGAR filing with extracted sections."""
    accession_number: str = ""
    cik: str = ""
    company_name: str = ""
    ticker: str = ""
    filing_type: FilingType = FilingType.K10
    filing_date: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    extracted_items: dict[str, str] = field(default_factory=dict)


class LMDictionary:
    """Loughran-McDonald financial sentiment dictionary.

    Contains word lists for negative, positive, uncertainty, litigious,
    constraining, strong_modal, and weak_modal categories.
    """

    negative = {
        "abandon", "abdicate", "aberrant", "abnormal", "abolish", "abrupt",
        "abuse", "accident", "adverse", "allegation", "annul", "anomaly",
        "arrears", "bankruptcy", "breach", "bribe", "catastrophe", "cease",
        "collapse", "complaint", "compromise", "condemn", "conspiracy",
        "corrupt", "crisis", "damage", "default", "deficiency", "deficit",
        "delinquency", "destroy", "deteriorate", "disaster", "discontinue",
        "dispute", "disruption", "downgrade", "embezzle", "erosion",
        "evasion", "expire", "exploit", "failure", "fault", "felony",
        "fictitious", "fine", "forfeit", "fraud", "hardship", "impair",
        "implicate", "improper", "inability", "inadequate", "incapacity",
        "indictment", "ineffective", "infraction", "injunction", "insolvency",
        "instability", "interruption", "investigation", "involuntary",
        "irregularity", "jeopardy", "lack", "lapse", "liability", "lien",
        "liquidation", "litigation", "loss", "malfeasance", "mismanagement",
        "misrepresent", "negligence", "noncompliance", "nullify", "obsolete",
        "offense", "omission", "onerous", "overdue", "penalty", "peril",
        "pessimistic", "plaintiff", "plea", "precarious", "prohibition",
        "protest", "questionable", "recession", "recourse", "refusal",
        "rejection", "relinquish", "renege", "repossess", "repudiate",
        "rescind", "resignation", "restructure", "revoke", "sanction",
        "scandal", "scrutiny", "seizure", "shortage", "shrinkage", "slowdown",
        "stagnation", "subpoena", "suspension", "termination", "theft",
        "threat", "tort", "turmoil", "uncollectible", "undercapitalized",
        "underperform", "unfavorable", "unsafe", "unstable", "unsustainable",
        "violation", "void", "warn", "waste", "weakness", "worsening",
        "write-down", "write-off",
    }

    positive = {
        "able", "abundance", "accomplish", "achieve", "advance", "advantage",
        "alliance", "auspicious", "award", "beneficial", "best", "booming",
        "breakthrough", "collaboration", "competitive", "complement",
        "confident", "constructive", "convenient", "cyclical", "definitive",
        "deliver", "dependable", "desirable", "distinctive", "dividend",
        "durable", "ease", "efficient", "enhance", "enormous", "enthusiasm",
        "exceed", "excellence", "exceptional", "exclusive", "expand",
        "favorable", "flexibility", "fortunate", "friendly", "gain",
        "generous", "growth", "guidance", "improve", "increase",
        "industry-leading", "innovation", "integrity", "leadership",
        "lucrative", "merit", "momentum", "opportunity", "optimistic",
        "outperform", "patent", "peerless", "pioneer", "popular",
        "preeminent", "premier", "premium", "prestige", "productivity",
        "progress", "prominent", "prosperity", "rebound", "reliable",
        "resilient", "revitalize", "robust", "satisfaction", "scalable",
        "shareholder", "simplify", "solution", "stability", "strength",
        "substantial", "superior", "sustainable", "synergy", "turnaround",
        "unmatched", "unprecedented", "upside", "valuable", "versatile",
        "vibrant", "visionary", "win", "worthwhile",
    }

    uncertainty = {
        "anticipate", "approximate", "assumption", "believe", "cautious",
        "conceivable", "contingency", "could", "depend", "estimate",
        "eventual", "expectation", "fluctuate", "forecast", "indefinite",
        "indeterminate", "likelihood", "maybe", "might", "nearby", "nearly",
        "occasional", "pending", "perhaps", "possible", "potentially",
        "predict", "preliminary", "probable", "projection", "prospect",
        "random", "reassess", "reconsider", "roughly", "seem", "somewhat",
        "speculate", "subject to", "suggest", "susceptible", "tend",
        "tentative", "theoretical", "unanticipated", "unclear",
        "undetermined", "unexpected", "unforeseen", "unknown", "unlikely",
        "unpredictable", "unproven", "unsure", "vague", "variability",
        "volatile",
    }

    litigious = {
        "abrogate", "acquit", "adjudicate", "affidavit", "allege", "appeal",
        "appellate", "arbitrate", "assertion", "attest", "claimant",
        "class action", "complainant", "confess", "contend", "contest",
        "contractual", "court", "criminal", "decree", "defend", "defendant",
        "depose", "disclaimer", "discovery", "doctrine", "docket", "enjoin",
        "evidence", "exonerate", "finding", "grievance", "guilty", "hearing",
        "heretofore", "illegal", "impound", "indict", "infringement",
        "injunction", "innocent", "interrogatory", "judge", "judicial",
        "jurisdiction", "jury", "justify", "lawsuit", "legal", "legislate",
        "legitimate", "litigant", "malpractice", "mandamus", "mediate",
        "negligence", "notarize", "null", "oath", "objection", "opinion",
        "overrule", "petition", "plaintiff", "pleading", "precedent",
        "prosecute", "regulation", "remand", "respondent", "ruling",
        "statute", "stipulation", "subpoena", "sue", "testament", "testify",
        "theretofore", "tort", "transcript", "tribunal", "void", "waiver",
        "warrant", "whereas", "witness", "writ",
    }

    def score(self, text: str) -> dict[str, float]:
        """Score text for all LM sentiment categories."""
        words = set(re.findall(r"\b[a-z][a-z-]+\b", text.lower()))
        total = max(len(words), 1)
        return {
            "negative": len(words & self.negative) / total,
            "positive": len(words & self.positive) / total,
            "uncertainty": len(words & self.uncertainty) / total,
            "litigious": len(words & self.litigious) / total,
        }

    def sentiment_score(self, text: str) -> float:
        """Net sentiment = (positive - negative) / total words."""
        s = self.score(text)
        return s["positive"] - s["negative"]


class FilingIngestor:
    """Fetch and analyze SEC EDGAR filings."""

    def __init__(self, user_agent: str = "QuantCore/0.1 (research)") -> None:
        self.user_agent = user_agent
        self._session: Any = None

    def _get_session(self) -> Any:
        import requests
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": self.user_agent})
        return self._session

    def close(self) -> None:
        if self._session:
            self._session.close()
            self._session = None

    def fetch_filing(self, cik: str, filing_type: FilingType) -> FilingDocument | None:
        """Fetch the latest filing of a given type for a CIK."""
        cik_clean = cik.lstrip("0")
        url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcompany&CIK={cik_clean}&type={filing_type.value}&count=1&output=atom"
        )
        try:
            resp = self._get_session().get(url, timeout=15)
            resp.raise_for_status()
        except Exception:
            return None
        return FilingDocument(cik=cik, filing_type=filing_type)

    def extract_sections(self, text: str) -> dict[str, str]:
        """Extract key sections from a filing's full text."""
        sections: dict[str, str] = {}
        patterns = {
            "risk_factors": r"Item\s+1A[\.\s].*?Risk\s+Factors(.*?)(?=Item\s+\d+[A-Z]?[\.\s])",
            "mda": r"Item\s+7[\.\s].*?(?:Management.*?Analysis|Financial\s+Condition)(.*?)(?=Item\s+\d+[A-Z]?[\.\s])",
            "financials": r"Item\s+8[\.\s].*?Financial\s+Statements(.*?)(?=Item\s+\d+[A-Z]?[\.\s]|$)",
        }
        for name, pattern in patterns.items():
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                sections[name] = match.group(1).strip()[:5000]
        return sections

    def analyze_sentiment(
        self, filing: FilingDocument, lm_dict: LMDictionary | None = None,
    ) -> dict[str, dict[str, float]]:
        """Score sentiment for each extracted section."""
        lm = lm_dict or LMDictionary()
        results: dict[str, dict[str, float]] = {}
        for name, section_text in filing.sections.items():
            results[name] = lm.score(section_text)
        return results


def compute_filing_sentiment_trend(filings: list[FilingDocument]) -> dict[str, list[float]]:
    """Track LM sentiment over a series of filings."""
    lm = LMDictionary()
    trends: dict[str, list[float]] = {}
    for f in filings:
        for name, text in f.sections.items():
            trends.setdefault(name, []).append(lm.sentiment_score(text))
    return trends
