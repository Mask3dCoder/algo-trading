"""Pytest integration for adversarial test scenarios.

Run with:  python -m pytest tests/adversarial/test_adversarial.py -v
"""

from tests.adversarial.harness import AdversarialHarness


def test_broker_fill_loss() -> None:
    """Scenario 1: Broker drops a fill, SUT diverges — must be detected."""
    h = AdversarialHarness()
    result = h.run_scenario("broker_fill_loss", seed=42)
    assert result.passed, (
        f"Broker fill loss scenario FAILED: "
        f"divergences={len(result.divergences)}, sis={result.sis:.4f}"
    )
    assert len(result.divergences) > 0, "Should detect dropped fill divergence"


def test_duplicate_fill_idempotent() -> None:
    """Scenario 2: Duplicate fill must be idempotently rejected."""
    h = AdversarialHarness()
    result = h.run_scenario("duplicate_fill_attack", seed=42)
    assert result.passed, (
        f"Duplicate fill scenario FAILED: "
        f"divergences={len(result.divergences)}, sis={result.sis:.4f}"
    )
    assert len(result.divergences) == 0, "Duplicate fill should not change state"


def test_cash_overcommit_blocked() -> None:
    """Scenario 3: Cash commitment must block orders exceeding available cash."""
    h = AdversarialHarness()
    result = h.run_scenario("cash_overcommit", seed=42)
    assert result.passed, (
        f"Cash overcommit scenario FAILED: "
        f"divergences={len(result.divergences)}, sis={result.sis:.4f}"
    )


def test_disconnect_silent_fill_detected() -> None:
    """Scenario 4: Fills during broker outage must be detected on reconnect."""
    h = AdversarialHarness()
    result = h.run_scenario("disconnect_silent_fill", seed=42)
    assert result.passed, (
        f"Disconnect silent fill scenario FAILED: "
        f"divergences={len(result.divergences)}, sis={result.sis:.4f}"
    )
    assert len(result.divergences) > 0, "Should detect divergence from silent fill"


def test_restart_reconstruction_matches() -> None:
    """Scenario 5: SUT restart must reconstruct state matching broker."""
    h = AdversarialHarness()
    result = h.run_scenario("restart_active_trading", seed=42)
    assert result.passed, (
        f"Restart reconstruction scenario FAILED: "
        f"divergences={len(result.divergences)}, sis={result.sis:.4f}"
    )
    assert len(result.divergences) == 0, "Reconstruction should match broker perfectly"


def test_all_scenarios_report() -> None:
    """Run all 5 scenarios and verify overall results."""
    h = AdversarialHarness()
    results = h.run_all_scenarios(seed=42)
    report = h.report_summary(results)
    assert all(r.passed for r in results.values()), (
        f"Not all scenarios passed:\n{report}"
    )
