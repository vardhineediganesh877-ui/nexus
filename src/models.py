"""
NEXUS Signal Types — The core data structures for the entire platform.
Every module speaks this language.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Any
from datetime import datetime
import uuid


class SignalSide(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class SignalStrength(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class AgentOpinion:
    """One agent's analysis result"""
    agent_name: str  # technical, sentiment, fundamental, risk
    signal: SignalSide
    strength: SignalStrength
    confidence: float  # 0.0 - 1.0
    reasoning: str
    indicators: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "agent": self.agent_name,
            "signal": self.signal.value,
            "strength": self.strength.value,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "indicators": self.indicators,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TradeSignal:
    """Consensus signal from all agents"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    symbol: str = ""
    exchange: str = ""
    timeframe: str = ""
    side: SignalSide = SignalSide.HOLD
    strength: SignalStrength = SignalStrength.NEUTRAL
    confidence: float = 0.0  # Weighted average of agent confidences
    opinions: List[AgentOpinion] = field(default_factory=list)
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size_pct: float = 0.0  # Recommended % of portfolio
    risk_reward_ratio: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def agents_agree(self) -> bool:
        """Do all agents agree on direction?"""
        if not self.opinions:
            return False
        sides = {o.signal for o in self.opinions if o.signal != SignalSide.HOLD}
        return len(sides) <= 1

    @property
    def technical_score(self) -> float:
        for o in self.opinions:
            if o.agent_name == "technical":
                return o.confidence * (1 if o.signal == SignalSide.BUY else -1)
        return 0.0

    @property
    def sentiment_score(self) -> float:
        for o in self.opinions:
            if o.agent_name == "sentiment":
                return o.confidence * (1 if o.signal == SignalSide.BUY else -1)
        return 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timeframe": self.timeframe,
            "side": self.side.value,
            "strength": self.strength.value,
            "confidence": round(self.confidence, 3),
            "opinions": [o.to_dict() for o in self.opinions],
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_size_pct": round(self.position_size_pct, 3),
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "agents_agree": self.agents_agree,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_telegram(self) -> str:
        """Format signal for Telegram message"""
        emoji = {"strong_buy": "🟢🟢", "buy": "🟢", "neutral": "⚪", "sell": "🔴", "strong_sell": "🔴🔴"}
        side_emoji = {"buy": "📈", "sell": "📉", "hold": "⏸️"}
        
        lines = [
            f"{side_emoji.get(self.side.value, '❓')} *{self.symbol}* — {self.side.value.upper()}",
            f"Strength: {emoji.get(self.strength.value, '')} {self.strength.value.replace('_', ' ').title()}",
            f"Confidence: {self.confidence:.0%}",
            f"Agents agree: {'✅' if self.agents_agree else '⚠️ Mixed'}",
        ]
        
        if self.entry_price:
            lines.append(f"Entry: ${self.entry_price:,.4f}")
        if self.stop_loss:
            lines.append(f"Stop: ${self.stop_loss:,.4f}")
        if self.take_profit:
            lines.append(f"Target: ${self.take_profit:,.4f}")
            if self.stop_loss:
                rr = (self.take_profit - self.entry_price) / (self.entry_price - self.stop_loss) if self.entry_price and self.stop_loss else 0
                lines.append(f"R:R: {rr:.1f}x")
        
        # Agent breakdown
        lines.append("\n📊 *Agent Analysis:*")
        for o in self.opinions:
            agent_emoji = {"technical": "📉", "sentiment": "🗣️", "fundamental": "🏛️", "risk": "🛡️"}
            lines.append(f"  {agent_emoji.get(o.agent_name, '🤖')} {o.agent_name.title()}: {o.signal.value.upper()} ({o.confidence:.0%})")
        
        return "\n".join(lines)


@dataclass
class Trade:
    """Executed or simulated trade"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    signal_id: str = ""
    symbol: str = ""
    exchange: str = ""
    side: SignalSide = SignalSide.BUY
    entry_price: float = 0.0
    quantity: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    status: str = "open"  # open, closed, cancelled
    paper: bool = True
    timestamp_opened: datetime = field(default_factory=datetime.utcnow)
    timestamp_closed: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "status": self.status,
            "paper": self.paper,
            "timestamp_opened": self.timestamp_opened.isoformat(),
            "timestamp_closed": self.timestamp_closed.isoformat() if self.timestamp_closed else None,
        }


@dataclass
class BacktestResult:
    """Backtest performance metrics"""
    symbol: str
    strategy: str
    period: str
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_return_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    calmar_ratio: float = 0.0
    expectancy: float = 0.0
    benchmark_return_pct: float = 0.0  # Buy and hold
    trades: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "period": self.period,
            "total_return_pct": round(self.total_return_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "win_rate": round(self.win_rate, 2),
            "profit_factor": round(self.profit_factor, 2),
            "total_trades": self.total_trades,
            "avg_trade_return_pct": round(self.avg_trade_return_pct, 2),
            "best_trade_pct": round(self.best_trade_pct, 2),
            "worst_trade_pct": round(self.worst_trade_pct, 2),
            "calmar_ratio": round(self.calmar_ratio, 2),
            "expectancy": round(self.expectancy, 2),
            "benchmark_return_pct": round(self.benchmark_return_pct, 2),
        }

    @property
    def grade(self) -> str:
        """A-F grade based on risk-adjusted returns"""
        if self.sharpe_ratio > 2.0 and self.max_drawdown_pct < 10:
            return "A+"
        elif self.sharpe_ratio > 1.5 and self.max_drawdown_pct < 15:
            return "A"
        elif self.sharpe_ratio > 1.0 and self.win_rate > 55:
            return "B"
        elif self.sharpe_ratio > 0.5:
            return "C"
        elif self.total_return_pct > 0:
            return "D"
        return "F"
