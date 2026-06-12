from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.monitor_paper_trading import (  # noqa: E402
    check_live_strategy_fill_activity,
    load_config,
    parse_args,
    resolve_paths,
    write_attribution_snapshot,
)


def test_parse_args_paper_switches_to_paper_paths() -> None:
    args = parse_args(["--paper"])
    paths = resolve_paths(str(args.config) if args.config else None, paper=args.paper)

    assert paths.config_path == PROJECT_ROOT / "config" / "paper.yaml"
    assert paths.events_path == PROJECT_ROOT / "reports" / "paper_events.jsonl"
    assert paths.equity_path == PROJECT_ROOT / "reports" / "paper_equity.csv"
    assert paths.monitor_log == PROJECT_ROOT / "reports" / "paper_monitoring_log.csv"
    assert paths.attribution_output == PROJECT_ROOT / "reports" / "paper_attribution.csv"


def test_parse_args_config_override() -> None:
    args = parse_args(["--config", "config/paper.yaml"])
    paths = resolve_paths(str(args.config) if args.config else None, paper=args.paper)

    assert paths.config_path == PROJECT_ROOT / "config" / "paper.yaml"
    assert paths.events_path == PROJECT_ROOT / "reports" / "paper_events.jsonl"
    assert paths.equity_path == PROJECT_ROOT / "reports" / "paper_equity.csv"


def test_load_config_reads_paper_strategies() -> None:
    strategies = load_config(PROJECT_ROOT / "config" / "paper.yaml")

    assert [strategy["name"] for strategy in strategies] == [
        "equity_stat_arb",
        "options_tail_hedge",
    ]


def test_write_attribution_snapshot_creates_paper_attribution(tmp_path: Path) -> None:
    output = tmp_path / "paper_attribution.csv"
    strategy_metrics = [
        {
            "name": "equity_stat_arb",
            "allocation_weight": 0.333333,
            "metrics": {"last_pnl": 12.34, "status": "running"},
        }
    ]
    metrics = {"n_days": 1, "current_equity": 1000.0, "daily_pnl": 10.0}

    write_attribution_snapshot(strategy_metrics, metrics, output)

    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "date,strategy,ret,pnl,allocation_weight,status" in text
    assert "equity_stat_arb" in text
    assert "0.333333" in text


def test_write_attribution_snapshot_empty_when_no_equity(tmp_path: Path) -> None:
    output = tmp_path / "paper_attribution.csv"

    write_attribution_snapshot([], {"n_days": 0}, output)

    assert output.exists()
    assert output.read_text(encoding="utf-8").splitlines() == [
        "date,strategy,ret,pnl,allocation_weight,status"
    ]


def test_check_live_strategy_fill_activity_alerts_zero_fills() -> None:
    now = datetime.now(tz=UTC).isoformat()
    strategies = [{"name": "opening_gap_fade", "allocation_weight": 0.1}]
    events = [
        {
            "event": "fill",
            "strategy": "opening_gap_fade",
            "timestamp": "2026-01-01T00:00:00Z",
        }
    ]

    alerts = check_live_strategy_fill_activity(strategies, events, window_days=5)

    assert alerts == ["opening_gap_fade: zero fills in rolling 5-day window"]

    recent_events = [*events, {"event": "fill", "strategy": "opening_gap_fade", "timestamp": now}]
    assert check_live_strategy_fill_activity(strategies, recent_events, window_days=5) == []
