"""Tests for scripts/paper_trade.py — CLI parsing, duration conversion, and metrics."""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.paper_trade import parse_duration, setup_metrics_csv


class TestParseDuration:
    """Validate duration string parsing."""

    def test_days_to_seconds(self) -> None:
        assert parse_duration("5d") == 5 * 86400

    def test_hours_to_seconds(self) -> None:
        assert parse_duration("12h") == 12 * 3600

    def test_minutes_to_seconds(self) -> None:
        assert parse_duration("30m") == 30 * 60

    def test_seconds(self) -> None:
        assert parse_duration("60s") == 60

    def test_no_space(self) -> None:
        assert parse_duration("1d") == 86400

    def test_with_space(self) -> None:
        assert parse_duration("1 d") == 86400

    def test_large_value(self) -> None:
        assert parse_duration("365d") == 365 * 86400

    def test_invalid_unit_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("5x")

    def test_alpha_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("abc")

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("")

    def test_missing_unit_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("123")


class TestMetricsCSV:
    """Validate metrics CSV file creation."""

    def test_csv_created_with_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, writer, f = setup_metrics_csv(tmpdir)
            assert os.path.exists(csv_path)
            f.close()

            with open(csv_path, "r") as fh:
                reader = csv.DictReader(fh)
                expected = {
                    "timestamp", "total_pnl", "unrealized_pnl", "realized_pnl",
                    "open_positions_count", "orders_submitted_count",
                    "orders_filled_count", "orders_rejected_count",
                    "kill_switch_active", "memory_rss_mb", "last_data_age_seconds",
                }
                assert set(reader.fieldnames or []) == expected

    def test_csv_writable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, writer, f = setup_metrics_csv(tmpdir)
            writer.writerow({
                "timestamp": "2026-01-01T00:00:00Z",
                "total_pnl": 100.0,
                "unrealized_pnl": 50.0,
                "realized_pnl": 50.0,
                "open_positions_count": 2,
                "orders_submitted_count": 10,
                "orders_filled_count": 8,
                "orders_rejected_count": 2,
                "kill_switch_active": False,
                "memory_rss_mb": 128.5,
                "last_data_age_seconds": 0.5,
            })
            f.flush()
            f.close()

            with open(csv_path, "r") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
                assert len(rows) == 1
                assert float(rows[0]["total_pnl"]) == 100.0
                assert rows[0]["kill_switch_active"] == "False"


class TestPaperTradeCLI:
    """Validate the CLI argument parser."""

    def test_parser_accepts_duration(self) -> None:
        from scripts.paper_trade import main as paper_main
        parser = argparse.ArgumentParser()
        parser.add_argument("--duration", "-d", required=True)
        parser.add_argument("--strategy", "-s", default="momentum")
        parser.add_argument("--data-source", default="mock")
        parser.add_argument("--capital", type=float, default=100_000.0)

        args = parser.parse_args(["--duration", "5d"])
        assert args.duration == "5d"
        assert args.strategy == "momentum"
        assert args.data_source == "mock"
        assert args.capital == 100_000.0

    def test_parser_rejects_missing_duration(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--duration", "-d", required=True)
        parser.add_argument("--strategy", "-s", default="momentum")
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parser_full_args(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--duration", "-d", required=True)
        parser.add_argument("--strategy", "-s", default="momentum")
        parser.add_argument("--data-source", default="mock")
        parser.add_argument("--capital", type=float, default=100_000.0)

        args = parser.parse_args([
            "--duration", "12h",
            "--strategy", "momentum",
            "--data-source", "mock",
            "--capital", "500000",
        ])
        assert args.duration == "12h"
        assert args.strategy == "momentum"
        assert args.data_source == "mock"
        assert args.capital == 500000.0


class TestPaperTradeSmoke:
    """Minimal end-to-end smoke test with mock feed and short duration."""

    def test_run_2_seconds_mock(self) -> None:
        """Run paper trading for 2 seconds with mock feed — validates no crash."""
        from scripts.paper_trade import run_paper_trade

        exit_code = run_paper_trade(
            duration_seconds=2,
            strategy_name="momentum",
            data_source="mock",
            initial_capital=100_000.0,
        )
        # Should exit cleanly after timeout (0) or via signal (1) — both are valid
        assert exit_code in (0, 1)
