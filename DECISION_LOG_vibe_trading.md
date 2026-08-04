# Decision Log — Vibe-Trading Integration

| Field | Value |
|---|---|
| Decision ID | DL-2026-07-09-VIBE |
| Date | 2026-07-09 |
| Status | **REJECTED** (do not import as runtime / dependency) |
| Author | QuantCore Team |
| Re-reviewed against | HKUDS/Vibe-Trading framework analysis |

## 1. Headline

**Vibe-Trading integration is rejected on epistemic, architectural, and
fiduciary-security grounds.** The advertised capabilities (data layer,
backtest engines, broker connectors, "Alpha Zoo" factors, three-layer safety
pillars, MCP/API surface) are *already native* to QuantCore and implemented
more rigorously. The one genuinely unique asset — an LLM "committee" that
turns natural language into strategies — is incompatible with a live-money,
point-in-time-guarded, deterministic execution core.

**Approved path:** continue using the existing Claude Code + Ruflo MCP stack
(`.mcp.json`) for natural-language-assisted research, routing every candidate
through QuantCore's native `backtest` + `data/pit` + `risk` pipeline. External
data sources (if ever adopted) enter **only** via scheduled, import-free ETL
dumps that are re-ingested and PIT-validated by `quantcore/data/` — never by
importing Vibe-Trading's runtime into the trust boundary.

## 2. Context

Vibe-Trading is an LLM-multi-agent "AI trading research OS" (29 preset agent
teams, NL→code strategy generation, 7 backtest engines, 18 free data sources,
connector profiles for IBKR/Robinhood, MCP server). The generic evaluation
framework recommends integrating it as a loosely-coupled "Strategy Research &
Signal Generation Module." This log records why that recommendation does **not**
transfer to QuantCore.

## 3. Evaluation (4-layer autopsy)

### Layer 1 — Epistemological fatal flaw (lookahead bias)
QuantCore enforces strict point-in-time discipline at two layers:
- Price level: `BacktestEngine` calls `PortfolioState.mark_to_market(snapshot)`
  and enforces a **strict look-ahead barrier** (`engine.py:981`) *before* any
  strategy logic runs; `on_snapshot` strategies observe only timestamp T.
- Fundamentals level: `quantcore/data/pit/blueprint.py` is a SEC/XBRL
  point-in-time store (`PITQuery.get_asof(cik, concept, as_of)` with
  `RestatementChain`) — restatements are modeled explicitly so a fact is only
  visible as of its actual public date.

Vibe-Trading generates strategy code via an LLM committee. Modern LLMs are
trained on internet snapshots that include historical prices, corporate
actions, and post-hoc research; their latent priors encode ex-post outcomes.
There is no PIT filter on its prompt engineering. Importing its outputs means
wrapping a black box that has already "cheated" statistically during training
and asking the PIT engine to retroactively un-cheat it — impossible.

### Layer 2 — Architectural anti-pattern (stochastic vs deterministic)
`grep -riE "openai|anthropic|deepseek|langchain" quantcore/` returns **nothing**:
the execution core is deliberately LLM-free for deterministic reproducibility
(walk-forward validated: 504d train / 126d test, DSR p<0.1, CI bounds in
`config/live.yaml`). Vibe-Trading relies on an async swarm of LLM agents with
temperature > 0; its outputs are non-stationary. A strategy generated today
(GPT-4-turbo) differs from one generated tomorrow (GPT-4o) under an identical
prompt. This poisons git history, CI, reconciliation (`reconciliation.py`), and
invalidates the OOS validation discipline — the "strategy" becomes an ephemeral
hallucination of the prompt chain, not a fixed algorithm.

### Layer 3 — Fiduciary security calculus
Pre-v0.1.10 CVEs (path traversal via `remember`, **DNS-rebinding auth bypass**,
**live-trading authorization spoofing via `proposal_id`**) sit *in the
live-trading authorization path* — the worst category for a money-moving
system. Pinning to >= v0.1.10 is necessary but insufficient: the design is
AI-first with security bolted on ("Mandates", "kill-switch"), whereas QuantCore
is financial-first with the human/API key as primary actor and the AI (Ruflo) as
a **read-only advisor**. Worse, a prompt-injection vector (e.g., a poisoned
headline "execute proposal 92X") could reach the MCP tool that routes to the
`proposal_id` endpoint — a **pre-trade** vulnerability inserted in front of a
system hardened only with **post-trade** defenses (`reconciliation_guard.py`,
`circuit_breaker.py`). Structural downgrade.

### Layer 4 — The "AI Research OS" fallacy
QuantCore already possesses a superior research OS: Claude Code + Ruflo MCP
(embeddings, swarm, git tooling in `.mcp.json`). It is **human-in-the-loop**:
the LLM proposes, the quant validates through PIT → backtest → stress
(`quantcore/risk/stress.py`) → PR. Vibe-Trading is full-auto. For a live-money
system, precision-guided (one auditable candidate) beats carpet-bombing (100
hallucinated strategies to vet). See `scripts/run_vol_target_momentum_backtest.py`
for a worked example: a strategy generated from native functions, run through the
engine with zero external imports.

## 4. Capability mapping (what the doc claims vs reality)

| Vibe-Trading claim | QuantCore native equivalent | Verdict |
|---|---|---|
| 18 free data sources w/ failover | `quantcore/data/` loaders; add sources via import-free ETL only | Reject runtime; abstract data only |
| 460-factor "Alpha Zoo" | `alpha/` suite (delta_neutral, iron_condor, residual_momentum, stat_arb, pairs, sentiment, fred) | Reject; 70% overlap, rest unvalidated |
| 7 backtest engines | `backtest/engine.py` + walk-forward + DSR CI bounds | Reject; inferior to native |
| LLM strategy generation | Claude+Ruflo HITL (read-only advisor) | Fatal reject (stochastic + ex-post) |
| CVE-patched security | IBKR/Alpaca direct auth; post-trade guards | Fatal reject (pre-trade injection) |
| MCP server | `.mcp.json` Ruflo (read-only, governed) | Reject; native superior |

## 5. Decision

Do **not** add Vibe-Trading as a dependency, runtime, or docker-compose service
reachable from the brokerage/secret network. If additional free data coverage is
later desired, implement a standalone, network-isolated ETL microservice that
writes raw OHLCV to disk/S3 in parquet; QuantCore ingests and PIT-validates it.
Vibe-Trading code must never touch QuantCore's process or PID.

## 6. Approved alternative (already in place)

- NL-assisted research: this Claude Code + Ruflo session against the QuantCore
  codebase (read-only advisor).
- Strategy validation: native `BacktestEngine` + `data/pit` + `risk/stress.py`,
  walk-forward discipline per `config/live.yaml`.
- Reference demo: `scripts/run_vol_target_momentum_backtest.py`.

## 7. Sign-off

| Role | Name | Date |
|---|---|---|
| Decision owner | QuantCore Team | 2026-07-09 |
| Security review | ________________ | __________ |
| Quant review | ________________ | __________ |
