"""Unit tests for paper_trade.py feed/broker factory and normalization helpers."""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


class TestGetAccountDict:
    def test_from_dict(self):
        from scripts.paper_trade import _get_account_dict

        broker = MagicMock(spec=["get_account"])
        broker.get_account.return_value = {
            "cash": 50000.0,
            "equity": 100000.0,
            "portfolio_value": 100000.0,
            "total_commission": 25.0,
        }

        result = _get_account_dict(broker)
        assert result["cash"] == 50000.0
        assert result["equity"] == 100000.0
        assert result["portfolio_value"] == 100000.0
        assert result["total_commission"] == 25.0

    def test_from_account_info(self):
        from quantcore.execution.brokers.base import AccountInfo
        from scripts.paper_trade import _get_account_dict

        acct_info = AccountInfo(
            account_id="DU123",
            cash=50000.0,
            equity=100000.0,
            portfolio_value=100000.0,
            margin_used=25000.0,
        )

        broker = MagicMock(spec=["get_account"])
        broker.get_account.return_value = acct_info

        result = _get_account_dict(broker)
        assert result["cash"] == 50000.0
        assert result["equity"] == 100000.0
        assert result["portfolio_value"] == 100000.0
        assert result["total_commission"] == 25000.0  # margin_used mapped

    def test_uses_sync_wrapper_if_available(self):
        from scripts.paper_trade import _get_account_dict

        broker = MagicMock(spec=["get_account_sync"])
        broker.get_account_sync.return_value = {
            "cash": 75000.0,
            "equity": 85000.0,
            "portfolio_value": 85000.0,
            "total_commission": 10.0,
        }

        result = _get_account_dict(broker)
        assert result["cash"] == 75000.0
        broker.get_account_sync.assert_called_once()


class TestGetBrokerPositions:
    def test_from_dict(self):
        from quantcore.core.types import Position
        from scripts.paper_trade import _get_broker_positions

        pos = Position(
            symbol="AAPL", quantity=Decimal("100"),
            avg_cost=Decimal("150"), market_price=Decimal("155"),
        )
        broker = MagicMock(spec=["get_positions"])
        broker.get_positions.return_value = {"AAPL": pos}

        result = _get_broker_positions(broker)
        assert isinstance(result, dict)
        assert result["AAPL"].symbol == "AAPL"
        assert result["AAPL"].quantity == Decimal("100")

    def test_from_list(self):
        from quantcore.core.types import Position
        from scripts.paper_trade import _get_broker_positions

        pos = Position(
            symbol="SPY", quantity=Decimal("50"),
            avg_cost=Decimal("400"), market_price=Decimal("410"),
        )
        broker = MagicMock(spec=["get_positions"])
        broker.get_positions.return_value = [pos]

        result = _get_broker_positions(broker)
        assert isinstance(result, dict)
        assert result["SPY"].symbol == "SPY"
        assert result["SPY"].quantity == Decimal("50")

    def test_empty_list(self):
        from scripts.paper_trade import _get_broker_positions

        broker = MagicMock(spec=["get_positions"])
        broker.get_positions.return_value = []

        result = _get_broker_positions(broker)
        assert result == {}

    def test_uses_sync_wrapper_if_available(self):
        from quantcore.core.types import Position
        from scripts.paper_trade import _get_broker_positions

        pos = Position(
            symbol="QQQ", quantity=Decimal("25"),
            avg_cost=Decimal("300"), market_price=Decimal("310"),
        )
        broker = MagicMock(spec=["get_positions_sync"])
        broker.get_positions_sync.return_value = [pos]

        result = _get_broker_positions(broker)
        assert result["QQQ"].symbol == "QQQ"
        broker.get_positions_sync.assert_called_once()


class TestUpdateBrokerState:
    def test_paper_broker_calls_update_methods(self):
        from scripts.paper_trade import _update_broker_state

        broker = MagicMock()
        broker.update_market_data = MagicMock()
        broker.update_equity = MagicMock()

        prices = {"AAPL": 150.0}
        _update_broker_state(broker, prices, broker_mode="mock")

        broker.update_market_data.assert_called_once_with(prices)
        broker.update_equity.assert_called_once()

    def test_ibkr_tracks_drawdown(self):
        from scripts.paper_trade import _update_broker_state

        broker = MagicMock()
        broker.drawdown_tracker = MagicMock()
        broker.get_account_sync.return_value = {
            "cash": 40000.0,
            "equity": 105000.0,
            "portfolio_value": 105000.0,
            "total_commission": 10.0,
        }

        _update_broker_state(broker, {}, broker_mode="ibkr")

        broker.drawdown_tracker.update.assert_called_once_with(105000.0)
        # Should not call PaperBroker methods
        assert not hasattr(broker.update_market_data, "called") or \
            not broker.update_market_data.called

    def test_ibkr_no_drawdown_tracker(self):
        from scripts.paper_trade import _update_broker_state

        broker = MagicMock()
        broker.drawdown_tracker = None

        _update_broker_state(broker, {}, broker_mode="ibkr")
        # Should not raise


class TestOrderToBrokerOrder:
    def test_buy_market_order(self):
        from quantcore.core.types import Order, OrderType, Side
        from scripts.paper_trade import _order_to_broker_order

        order = Order(
            id="test-1", symbol="AAPL", side=Side.BUY,
            type=OrderType.MARKET, quantity=Decimal("100"),
            strategy_id="test",
        )
        bo = _order_to_broker_order(order)
        assert bo.symbol == "AAPL"
        assert bo.side == "buy"
        assert bo.order_type == "market"
        assert bo.quantity == Decimal("100")

    def test_sell_limit_order(self):
        from quantcore.core.types import Order, OrderType, Side
        from scripts.paper_trade import _order_to_broker_order

        order = Order(
            id="test-2", symbol="SPY", side=Side.SELL,
            type=OrderType.LIMIT, quantity=Decimal("50"),
            limit_price=Decimal("450.0"), strategy_id="test",
        )
        bo = _order_to_broker_order(order)
        assert bo.symbol == "SPY"
        assert bo.side == "sell"
        assert bo.order_type == "limit"
        assert bo.limit_price == Decimal("450.0")


# ---------------------------------------------------------------------------
# Factory: create_feed_and_broker
# ---------------------------------------------------------------------------


class TestCreateFeedAndBroker:
    @pytest.fixture
    def risk_cfg(self):
        from quantcore.risk.config_loader import RiskConfig
        cfg = MagicMock(spec=RiskConfig)
        cfg.drawdown = MagicMock()
        return cfg

    @pytest.fixture
    def logger(self):
        return MagicMock()

    def test_mock_mock(self, risk_cfg, logger):
        from scripts.paper_trade import MockDataFeed, create_feed_and_broker

        feed, broker = create_feed_and_broker(
            data_source="mock", broker_mode="mock",
            symbols=["STK0000"], risk_cfg=risk_cfg, logger=logger,
        )
        assert isinstance(feed, MockDataFeed)

    def test_mock_alpaca(self, risk_cfg, logger):
        from scripts.paper_trade import create_feed_and_broker

        feed, broker = create_feed_and_broker(
            data_source="mock", broker_mode="alpaca",
            symbols=["STK0000"], risk_cfg=risk_cfg, logger=logger,
        )
        # PaperBroker
        assert hasattr(broker, "update_market_data")

    def test_unknown_data_source_raises(self, risk_cfg, logger):
        from scripts.paper_trade import create_feed_and_broker

        with pytest.raises(ValueError, match="Unknown data source"):
            create_feed_and_broker(
                data_source="unknown", broker_mode="mock",
                symbols=["SPY"], risk_cfg=risk_cfg, logger=logger,
            )

    def test_unknown_broker_mode_raises(self, risk_cfg, logger):
        from scripts.paper_trade import create_feed_and_broker

        with pytest.raises(ValueError, match="Unknown broker mode"):
            create_feed_and_broker(
                data_source="mock", broker_mode="unknown",
                symbols=["SPY"], risk_cfg=risk_cfg, logger=logger,
            )

    def test_yahoo_not_available_raises(self, risk_cfg, logger):
        from scripts.paper_trade import create_feed_and_broker

        with pytest.raises(ValueError, match="Yahoo data feed not available"):
            create_feed_and_broker(
                data_source="yahoo", broker_mode="mock",
                symbols=["SPY"], risk_cfg=risk_cfg, logger=logger,
            )
