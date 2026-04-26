"""Tests for the /health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from quantcore.api.main import app
from quantcore.core import health_state

client = TestClient(app)


class TestHealthEndpoint:
    """Validate the /health endpoint returns correct schema and values."""

    def test_health_returns_200(self) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_status_is_ok(self) -> None:
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_required_fields_present(self) -> None:
        resp = client.get("/health")
        data = resp.json()
        required = [
            "status", "uptime_seconds", "last_data_timestamp",
            "open_positions_count", "last_order_timestamp",
            "kill_switch_active", "memory_rss_mb", "timestamp",
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_uptime_is_non_negative(self) -> None:
        resp = client.get("/health")
        data = resp.json()
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0

    def test_kill_switch_is_boolean(self) -> None:
        resp = client.get("/health")
        data = resp.json()
        assert isinstance(data["kill_switch_active"], bool)

    def test_health_unauthenticated(self) -> None:
        """Health endpoint does not require authentication."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_responds_to_state_changes(self) -> None:
        """Verify the endpoint reflects state updates in real time."""
        health_state.set_kill_switch_active(True)
        health_state.set_open_positions_count(5)
        health_state.record_data_point()
        health_state.record_order_submitted()

        resp = client.get("/health")
        data = resp.json()

        assert data["kill_switch_active"] is True
        assert data["open_positions_count"] == 5
        assert data["last_data_timestamp"] is not None
        assert data["last_order_timestamp"] is not None

        # Reset for other tests
        health_state.set_kill_switch_active(False)
        health_state.set_open_positions_count(0)

    def test_memory_rss_mb_is_number(self) -> None:
        resp = client.get("/health")
        data = resp.json()
        mem = data["memory_rss_mb"]
        # Should be a positive number when psutil is available
        assert mem is None or (isinstance(mem, float) and mem > 0)
