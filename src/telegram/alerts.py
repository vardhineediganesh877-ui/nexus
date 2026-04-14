"""
NEXUS Telegram Alert System — Automatic signal notifications.

Sends alerts when:
- Strong buy/sell signals detected during scan
- Confidence exceeds configurable threshold
- Risk agent flags danger

Two modes:
1. Push mode: Called by the scan loop, sends immediately
2. Queue mode: Signals queued, batch-sent on interval
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Coroutine, Deque, List, Optional

from ..models import TradeSignal, SignalSide, SignalStrength

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert priority levels"""
    CRITICAL = "critical"  # Strong signal + agents agree + high confidence
    HIGH = "high"          # Strong signal or very high confidence
    MEDIUM = "medium"      # Regular actionable signal
    LOW = "low"            # Hold / informational


@dataclass
class AlertRule:
    """Configurable rule for when to trigger alerts"""
    min_confidence: float = 0.65
    min_strength: str = "buy"  # buy, strong_buy, sell, strong_sell
    agents_must_agree: bool = False
    enabled: bool = True
    quiet_hours: tuple = ()  # (start_hour, end_hour) UTC, e.g. (0, 6)


@dataclass
class Alert:
    """A formatted alert ready to send"""
    level: AlertLevel
    signal: TradeSignal
    message: str
    timestamp: float = field(default_factory=time.time)
    sent: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "signal_id": self.signal.id,
            "symbol": self.signal.symbol,
            "side": self.signal.side.value,
            "confidence": round(self.signal.confidence, 3),
            "sent": self.sent,
            "error": self.error,
        }


class AlertManager:
    """
    Evaluates signals and decides whether to send alerts.

    Usage:
        manager = AlertManager(config)
        alert = manager.evaluate(signal)
        if alert:
            await manager.send(alert)
    """

    def __init__(self, rules: Optional[List[AlertRule]] = None,
                 send_fn: Optional[Callable[[str], Coroutine]] = None):
        self.rules = rules or [AlertRule()]
        self.send_fn = send_fn
        self._queue: Deque[Alert] = deque(maxlen=100)
        self._sent_count = 0
        self._skipped_count = 0
        self._cooldown: dict = {}  # symbol -> last_alert_time
        self._cooldown_seconds = 300  # 5 min between alerts per symbol

    def evaluate(self, signal: TradeSignal) -> Optional[Alert]:
        """Evaluate a signal against alert rules. Returns Alert if actionable."""
        if signal.side == SignalSide.HOLD:
            self._skipped_count += 1
            return None

        # Check cooldown — don't spam same symbol
        now = time.time()
        last = self._cooldown.get(signal.symbol, 0)
        if now - last < self._cooldown_seconds:
            logger.debug(f"Alert cooldown for {signal.symbol}, skipping")
            self._skipped_count += 1
            return None

        # Set cooldown immediately after check passes (even if send fails later)
        self._cooldown[signal.symbol] = time.time()

        # Check against rules
        actionable = False
        for rule in self.rules:
            if not rule.enabled:
                continue
            if signal.confidence >= rule.min_confidence:
                actionable = True
                break

        if not actionable:
            self._skipped_count += 1
            return None

        # Determine alert level
        level = self._classify_alert(signal)

        # Build message
        message = self._format_alert(signal, level)

        alert = Alert(
            level=level,
            signal=signal,
            message=message,
        )

        self._queue.append(alert)
        return alert

    async def send(self, alert: Alert) -> bool:
        """Send an alert via the configured send function."""
        if not self.send_fn:
            logger.warning("No send_fn configured, alert not sent")
            alert.error = "no_send_fn"
            return False

        try:
            await self.send_fn(alert.message)
            alert.sent = True
            self._sent_count += 1
            logger.info(f"Alert sent: {alert.signal.symbol} {alert.signal.side.value}")
            return True
        except Exception as e:
            alert.error = str(e)
            logger.error(f"Failed to send alert: {e}")
            return False

    async def process_signal(self, signal: TradeSignal) -> Optional[Alert]:
        """Evaluate + send in one call. Returns the alert if sent."""
        alert = self.evaluate(signal)
        if alert:
            await self.send(alert)
        return alert

    async def process_signals(self, signals: List[TradeSignal]) -> List[Alert]:
        """Process multiple signals, return alerts that were sent."""
        alerts = []
        for signal in signals:
            alert = await self.process_signal(signal)
            if alert and alert.sent:
                alerts.append(alert)
        return alerts

    def _classify_alert(self, signal: TradeSignal) -> AlertLevel:
        """Classify alert severity."""
        if (signal.strength in (SignalStrength.STRONG_BUY, SignalStrength.STRONG_SELL)
                and signal.agents_agree and signal.confidence >= 0.8):
            return AlertLevel.CRITICAL
        elif signal.strength in (SignalStrength.STRONG_BUY, SignalStrength.STRONG_SELL):
            return AlertLevel.HIGH
        elif signal.confidence >= 0.65:
            return AlertLevel.MEDIUM
        else:
            return AlertLevel.LOW

    def _format_alert(self, signal: TradeSignal, level: AlertLevel) -> str:
        """Format a signal into a Telegram-ready alert message."""
        level_emoji = {
            AlertLevel.CRITICAL: "🚨",
            AlertLevel.HIGH: "⚡",
            AlertLevel.MEDIUM: "📊",
            AlertLevel.LOW: "ℹ️",
        }
        side_emoji = {"buy": "🟢", "sell": "🔴"}

        lines = [
            f"{level_emoji.get(level, '📊')} *NEXUS Signal Alert* [{level.value.upper()}]",
            "",
            f"{side_emoji.get(signal.side.value, '⚪')} *{signal.symbol}* — {signal.side.value.upper()}",
            f"Strength: {signal.strength.value.replace('_', ' ').title()}",
            f"Confidence: {signal.confidence:.0%}",
            f"Exchange: {signal.exchange.upper()} | TF: {signal.timeframe}",
        ]

        if signal.agents_agree:
            lines.append("🤝 All agents agree")
        else:
            lines.append("⚠️ Mixed signals")

        if signal.entry_price:
            lines.append(f"Entry: `${signal.entry_price:,.4f}`")
        if signal.stop_loss:
            lines.append(f"Stop: `${signal.stop_loss:,.4f}`")
        if signal.take_profit:
            lines.append(f"Target: `${signal.take_profit:,.4f}`")
        if signal.risk_reward_ratio > 0:
            lines.append(f"R:R: {signal.risk_reward_ratio:.1f}x")

        lines.append("")
        lines.append("📊 _Agent Breakdown:_")
        for o in signal.opinions:
            agent_emoji = {
                "technical": "📉", "sentiment": "🗣️",
                "fundamental": "🏛️", "risk": "🛡️"
            }
            lines.append(
                f"  {agent_emoji.get(o.agent_name, '🤖')} {o.agent_name.title()}: "
                f"{o.signal.value.upper()} ({o.confidence:.0%})"
            )

        return "\n".join(lines)

    @property
    def stats(self) -> dict:
        return {
            "sent": self._sent_count,
            "skipped": self._skipped_count,
            "queued": len(self._queue),
            "cooldowns": len(self._cooldown),
        }
