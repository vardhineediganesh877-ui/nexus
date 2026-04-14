"""
Tests for NEXUS Telegram signal alerts.

Covers:
- AlertManager evaluation (rules, cooldown, classification)
- Alert formatting
- TelegramBot command processing
- AlertManager stats
- Edge cases
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from src.models import (
    TradeSignal, SignalSide, SignalStrength, AgentOpinion
)
from src.telegram.bot import TelegramBot
from src.telegram.alerts import AlertManager, AlertRule, AlertLevel, Alert
from src.config import NexusConfig


# ─── Fixtures ───────────────────────────────────────────────

def make_signal(
    symbol="BTC/USDT",
    side=SignalSide.BUY,
    strength=SignalStrength.BUY,
    confidence=0.75,
    agents_agree=True,
    entry_price=50000.0,
    stop_loss=49000.0,
    take_profit=52000.0,
) -> TradeSignal:
    """Create a test trade signal."""
    signal = TradeSignal(
        symbol=symbol,
        exchange="binance",
        timeframe="1h",
        side=side,
        strength=strength,
        confidence=confidence,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_reward_ratio=3.0 if stop_loss and take_profit else 0.0,
    )

    opinions = [
        AgentOpinion(
            agent_name="technical",
            signal=side,
            strength=strength,
            confidence=confidence,
            reasoning="Test technical analysis",
            indicators={"rsi": 35, "macd": "bullish"},
        ),
        AgentOpinion(
            agent_name="sentiment",
            signal=side,
            strength=strength,
            confidence=confidence * 0.9,
            reasoning="Test sentiment analysis",
        ),
    ]

    if not agents_agree:
        opinions.append(
            AgentOpinion(
                agent_name="risk",
                signal=SignalSide.SELL if side == SignalSide.BUY else SignalSide.BUY,
                strength=SignalStrength.SELL,
                confidence=0.6,
                reasoning="Contrarian risk opinion",
            )
        )

    signal.opinions = opinions
    return signal


def make_config() -> NexusConfig:
    """Create a test config."""
    return NexusConfig()


# ─── AlertManager Tests ────────────────────────────────────

class TestAlertEvaluation:
    """Test signal evaluation and filtering."""

    def test_hold_signal_not_actionable(self):
        """HOLD signals should never trigger alerts."""
        mgr = AlertManager()
        signal = make_signal(side=SignalSide.HOLD, strength=SignalStrength.NEUTRAL)
        result = mgr.evaluate(signal)
        assert result is None

    def test_buy_signal_above_threshold(self):
        """BUY signal above min_confidence should trigger alert."""
        mgr = AlertManager(rules=[AlertRule(min_confidence=0.6)])
        signal = make_signal(confidence=0.75)
        alert = mgr.evaluate(signal)
        assert alert is not None
        assert alert.signal.symbol == "BTC/USDT"
        assert alert.level in (AlertLevel.MEDIUM, AlertLevel.HIGH, AlertLevel.CRITICAL)

    def test_buy_signal_below_threshold(self):
        """Signal below min_confidence should be skipped."""
        mgr = AlertManager(rules=[AlertRule(min_confidence=0.8)])
        signal = make_signal(confidence=0.5)
        result = mgr.evaluate(signal)
        assert result is None

    def test_multiple_rules_any_match(self):
        """Any matching rule should trigger alert (OR logic)."""
        mgr = AlertManager(rules=[
            AlertRule(min_confidence=0.9),  # too high
            AlertRule(min_confidence=0.5),  # matches
        ])
        signal = make_signal(confidence=0.7)
        alert = mgr.evaluate(signal)
        assert alert is not None

    def test_disabled_rule_skipped(self):
        """Disabled rules should not trigger."""
        mgr = AlertManager(rules=[AlertRule(min_confidence=0.3, enabled=False)])
        signal = make_signal(confidence=0.9)
        result = mgr.evaluate(signal)
        assert result is None

    def test_sell_signal_actionable(self):
        """SELL signals should also trigger alerts."""
        mgr = AlertManager(rules=[AlertRule(min_confidence=0.6)])
        signal = make_signal(side=SignalSide.SELL, confidence=0.8)
        alert = mgr.evaluate(signal)
        assert alert is not None
        assert "🔴" in alert.message or "SELL" in alert.message

    def test_no_rules_defaults(self):
        """Default rule set should work."""
        mgr = AlertManager()  # Default: min_confidence=0.65
        signal = make_signal(confidence=0.7)
        alert = mgr.evaluate(signal)
        assert alert is not None


class TestAlertCooldown:
    """Test per-symbol cooldown prevents spam."""

    def test_cooldown_blocks_duplicate(self):
        """Same symbol within cooldown window should be blocked."""
        mgr = AlertManager(rules=[AlertRule(min_confidence=0.5)])
        mgr._cooldown_seconds = 60

        signal1 = make_signal(symbol="ETH/USDT", confidence=0.8)
        alert1 = mgr.evaluate(signal1)
        assert alert1 is not None

        # Second signal same symbol — should be blocked
        signal2 = make_signal(symbol="ETH/USDT", confidence=0.9)
        alert2 = mgr.evaluate(signal2)
        assert alert2 is None

    def test_cooldown_allows_different_symbols(self):
        """Different symbols should not be affected by each other's cooldown."""
        mgr = AlertManager(rules=[AlertRule(min_confidence=0.5)])
        mgr._cooldown_seconds = 300

        s1 = make_signal(symbol="BTC/USDT", confidence=0.8)
        a1 = mgr.evaluate(s1)
        assert a1 is not None

        s2 = make_signal(symbol="ETH/USDT", confidence=0.8)
        a2 = mgr.evaluate(s2)
        assert a2 is not None

    def test_cooldown_expires(self):
        """Alerts should work after cooldown expires."""
        mgr = AlertManager(rules=[AlertRule(min_confidence=0.5)])
        mgr._cooldown_seconds = 0  # Instant expiry

        s1 = make_signal(symbol="BTC/USDT", confidence=0.8)
        mgr.evaluate(s1)

        s2 = make_signal(symbol="BTC/USDT", confidence=0.8)
        a2 = mgr.evaluate(s2)
        assert a2 is not None


class TestAlertClassification:
    """Test alert level classification."""

    def test_strong_buy_agents_agree_high_confidence(self):
        """Strong buy + agree + high conf = CRITICAL."""
        mgr = AlertManager()
        signal = make_signal(
            side=SignalSide.BUY,
            strength=SignalStrength.STRONG_BUY,
            confidence=0.85,
            agents_agree=True,
        )
        alert = mgr.evaluate(signal)
        assert alert is not None
        assert alert.level == AlertLevel.CRITICAL

    def test_strong_buy_mixed_agents(self):
        """Strong buy but mixed agents = HIGH (not CRITICAL)."""
        mgr = AlertManager()
        signal = make_signal(
            side=SignalSide.BUY,
            strength=SignalStrength.STRONG_BUY,
            confidence=0.85,
            agents_agree=False,
        )
        alert = mgr.evaluate(signal)
        assert alert is not None
        assert alert.level == AlertLevel.HIGH

    def test_regular_buy(self):
        """Regular buy signal = MEDIUM."""
        mgr = AlertManager()
        signal = make_signal(
            side=SignalSide.BUY,
            strength=SignalStrength.BUY,
            confidence=0.7,
        )
        alert = mgr.evaluate(signal)
        assert alert is not None
        assert alert.level == AlertLevel.MEDIUM


class TestAlertFormatting:
    """Test alert message formatting."""

    def test_alert_has_emoji(self):
        """Alert messages should contain emoji."""
        mgr = AlertManager()
        signal = make_signal(confidence=0.8)
        alert = mgr.evaluate(signal)
        assert alert is not None
        assert "🟢" in alert.message or "🔴" in alert.message

    def test_alert_has_symbol(self):
        """Alert should contain the symbol."""
        mgr = AlertManager()
        signal = make_signal(symbol="SOL/USDT", confidence=0.8)
        alert = mgr.evaluate(signal)
        assert "SOL/USDT" in alert.message

    def test_alert_has_confidence(self):
        """Alert should show confidence percentage."""
        mgr = AlertManager()
        signal = make_signal(confidence=0.85)
        alert = mgr.evaluate(signal)
        assert "85%" in alert.message

    def test_alert_has_agent_breakdown(self):
        """Alert should show agent opinions."""
        mgr = AlertManager()
        signal = make_signal(confidence=0.8)
        alert = mgr.evaluate(signal)
        assert "technical" in alert.message.lower() or "Technical" in alert.message

    def test_alert_to_dict(self):
        """Alert to_dict should be serializable."""
        mgr = AlertManager()
        signal = make_signal(confidence=0.8)
        alert = mgr.evaluate(signal)
        d = alert.to_dict()
        assert isinstance(d, dict)
        assert "level" in d
        assert "symbol" in d


class TestAlertSending:
    """Test alert sending mechanics."""

    @pytest.mark.asyncio
    async def test_send_with_callback(self):
        """Alert should call send_fn."""
        mock_fn = AsyncMock()
        mgr = AlertManager(send_fn=mock_fn)
        signal = make_signal(confidence=0.8)

        alert = mgr.evaluate(signal)
        assert alert is not None

        result = await mgr.send(alert)
        assert result is True
        assert alert.sent is True
        mock_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_without_callback(self):
        """Alert without send_fn should return False."""
        mgr = AlertManager()
        signal = make_signal(confidence=0.8)

        alert = mgr.evaluate(signal)
        result = await mgr.send(alert)
        assert result is False
        assert alert.error == "no_send_fn"

    @pytest.mark.asyncio
    async def test_send_failure_handled(self):
        """Send failure should be caught gracefully."""
        mock_fn = AsyncMock(side_effect=Exception("Network error"))
        mgr = AlertManager(send_fn=mock_fn)
        signal = make_signal(confidence=0.8)

        alert = mgr.evaluate(signal)
        result = await mgr.send(alert)
        assert result is False
        assert alert.error == "Network error"

    @pytest.mark.asyncio
    async def test_process_signal_end_to_end(self):
        """process_signal should evaluate + send."""
        mock_fn = AsyncMock()
        mgr = AlertManager(
            rules=[AlertRule(min_confidence=0.5)],
            send_fn=mock_fn,
        )
        signal = make_signal(confidence=0.8)
        alert = await mgr.process_signal(signal)
        assert alert is not None
        assert alert.sent is True
        mock_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_signals_batch(self):
        """process_signals should handle multiple signals."""
        mock_fn = AsyncMock()
        mgr = AlertManager(send_fn=mock_fn)

        signals = [
            make_signal(symbol="BTC/USDT", confidence=0.8),
            make_signal(symbol="ETH/USDT", confidence=0.9),
        ]
        alerts = await mgr.process_signals(signals)
        assert len(alerts) == 2
        assert all(a.sent for a in alerts)


class TestAlertManagerStats:
    """Test statistics tracking."""

    def test_stats_initial(self):
        """Stats should start at zero."""
        mgr = AlertManager()
        stats = mgr.stats
        assert stats["sent"] == 0
        assert stats["skipped"] == 0
        assert stats["queued"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_operations(self):
        """Stats should track sent and skipped counts."""
        mock_fn = AsyncMock()
        mgr = AlertManager(rules=[AlertRule(min_confidence=0.5)], send_fn=mock_fn)

        # Send one
        await mgr.process_signal(make_signal(symbol="BTC/USDT", confidence=0.8))

        # Skip one (hold)
        await mgr.process_signal(make_signal(side=SignalSide.HOLD))

        # Skip another (same symbol cooldown)
        await mgr.process_signal(make_signal(symbol="BTC/USDT", confidence=0.8))

        stats = mgr.stats
        assert stats["sent"] == 1
        assert stats["skipped"] == 2


# ─── TelegramBot Command Tests ─────────────────────────────

class TestTelegramBotCommands:
    """Test command processing (no API calls, pure logic)."""

    def test_help_command(self):
        """Help should list all commands."""
        config = make_config()
        bot = TelegramBot(config)
        response = bot.process_message("/help")
        assert "analyze" in response
        assert "scan" in response
        assert "portfolio" in response

    def test_start_command(self):
        """Start should return help."""
        config = make_config()
        bot = TelegramBot(config)
        response = bot.process_message("/start")
        assert "NEXUS" in response

    def test_unknown_command(self):
        """Unknown command should show help hint."""
        config = make_config()
        bot = TelegramBot(config)
        response = bot.process_message("/foobar")
        assert "Unknown" in response or "help" in response.lower()

    def test_analyze_no_args(self):
        """Analyze without args should show usage."""
        config = make_config()
        bot = TelegramBot(config)
        response = bot.process_message("/analyze")
        assert "Usage" in response

    def test_analyze_auto_usdt(self):
        """Analyze should auto-append /USDT."""
        config = make_config()
        bot = TelegramBot(config)
        # This will try to call signal_engine.analyze which needs mocking
        # Just test the parsing logic
        with patch.object(bot.signal_engine, 'analyze') as mock_analyze:
            mock_signal = make_signal()
            mock_signal.to_telegram = MagicMock(return_value="Signal result")
            mock_analyze.return_value = mock_signal

            response = bot.process_message("/analyze BTC")
            mock_analyze.assert_called_once_with("BTC/USDT", "mexc", "1h")

    def test_command_aliases(self):
        """Short aliases should work."""
        config = make_config()
        bot = TelegramBot(config)

        with patch.object(bot.signal_engine, 'analyze') as m:
            m.return_value.to_telegram.return_value = "ok"
            bot.process_message("/a BTC")
            m.assert_called_once()

    def test_status_command(self):
        """Status should show system info."""
        config = make_config()
        bot = TelegramBot(config)
        with patch.object(bot.executor, 'get_portfolio_summary') as m:
            m.return_value = {
                "paper_mode": True,
                "total_trades": 0,
                "open_positions": 0,
                "closed_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
            }
            response = bot.process_message("/status")
            assert "NEXUS" in response
            assert "Paper" in response

    def test_error_handling(self):
        """Bot should handle errors gracefully."""
        config = make_config()
        bot = TelegramBot(config)
        with patch.object(bot.signal_engine, 'analyze', side_effect=Exception("test error")):
            response = bot.process_message("/analyze BTC")
            assert "Error" in response


# ─── Integration Tests ─────────────────────────────────────

class TestAlertIntegration:
    """Integration: AlertManager + TelegramBot."""

    @pytest.mark.asyncio
    async def test_signal_to_alert_to_message(self):
        """Full flow: signal → alert → formatted message."""
        mock_fn = AsyncMock()
        mgr = AlertManager(
            rules=[AlertRule(min_confidence=0.5)],
            send_fn=mock_fn,
        )

        signal = make_signal(
            symbol="ETH/USDT",
            side=SignalSide.BUY,
            strength=SignalStrength.STRONG_BUY,
            confidence=0.9,
            agents_agree=True,
            entry_price=3500.0,
            stop_loss=3400.0,
            take_profit=3700.0,
        )

        alert = await mgr.process_signal(signal)
        assert alert is not None
        assert alert.sent is True
        assert "ETH/USDT" in alert.message
        assert "90%" in alert.message
        assert "CRITICAL" in alert.message

        # Verify message was sent
        assert mock_fn.call_count == 1
        sent_msg = mock_fn.call_args[0][0]
        assert "ETH/USDT" in sent_msg

    @pytest.mark.asyncio
    async def test_scan_with_alerts(self):
        """Simulate a scan that generates alerts for strong signals."""
        mock_fn = AsyncMock()
        mgr = AlertManager(send_fn=mock_fn)

        signals = [
            make_signal(symbol="BTC/USDT", side=SignalSide.BUY, confidence=0.9),
            make_signal(symbol="ETH/USDT", side=SignalSide.HOLD),  # Skipped
            make_signal(symbol="SOL/USDT", side=SignalSide.SELL, confidence=0.8),
            make_signal(symbol="DOGE/USDT", side=SignalSide.BUY, confidence=0.3),  # Below threshold
        ]

        alerts = await mgr.process_signals(signals)
        # BTC and SOL should alert, ETH is hold, DOGE below threshold
        assert len(alerts) == 2
        symbols = {a.signal.symbol for a in alerts}
        assert "BTC/USDT" in symbols
        assert "SOL/USDT" in symbols


class TestAlertRuleDefaults:
    """Test various rule configurations."""

    def test_default_rule_values(self):
        """Default AlertRule should have sensible values."""
        rule = AlertRule()
        assert rule.min_confidence == 0.65
        assert rule.enabled is True
        assert rule.agents_must_agree is False

    def test_custom_rule(self):
        """Custom rules should override defaults."""
        rule = AlertRule(min_confidence=0.9, agents_must_agree=True)
        assert rule.min_confidence == 0.9
        assert rule.agents_must_agree is True

    def test_alert_level_enum(self):
        """AlertLevel enum should have all levels."""
        assert AlertLevel.CRITICAL.value == "critical"
        assert AlertLevel.HIGH.value == "high"
        assert AlertLevel.MEDIUM.value == "medium"
        assert AlertLevel.LOW.value == "low"
