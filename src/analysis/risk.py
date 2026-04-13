"""
NEXUS Risk Manager Agent

Position sizing, portfolio risk limits, drawdown protection.
Kelly criterion, correlation analysis, exposure controls.
"""

import logging
from typing import Dict, Optional, TYPE_CHECKING
from ..models import AgentOpinion, SignalSide, SignalStrength, TradeSignal
from ..config import RiskConfig

if TYPE_CHECKING:
    from .correlation import CorrelationMatrix

logger = logging.getLogger(__name__)


class RiskManager:
    """Portfolio risk management"""

    def __init__(self, config: RiskConfig):
        self.config = config
        self._correlation_matrix: Optional['CorrelationMatrix'] = None

    @property
    def correlation(self) -> 'CorrelationMatrix':
        """Lazy-load correlation matrix (needs NexusConfig, not just RiskConfig)"""
        if self._correlation_matrix is None:
            raise RuntimeError("Call risk_mgr.set_correlation_matrix() first")
        return self._correlation_matrix

    def set_correlation_matrix(self, matrix: 'CorrelationMatrix') -> None:
        """Inject correlation matrix dependency"""
        self._correlation_matrix = matrix

    def check_signal(self, signal: TradeSignal, portfolio_value: float,
                     open_positions: list) -> AgentOpinion:
        """Evaluate risk of a trade signal"""
        reasons = []
        risk_score = 100  # Start at 100 (no risk), deduct for each risk factor
        
        # 1. Confidence threshold
        if signal.confidence < self.config.min_confidence:
            risk_score -= 40
            reasons.append(f"Low confidence ({signal.confidence:.0%} < {self.config.min_confidence:.0%})")
        
        # 2. Agent disagreement
        if not signal.agents_agree:
            risk_score -= 20
            reasons.append("Agents disagree on direction")
        
        # 3. Portfolio exposure check
        current_exposure = sum(
            p.get("value", 0) for p in open_positions
        )
        exposure_pct = current_exposure / portfolio_value if portfolio_value > 0 else 0
        
        if exposure_pct >= self.config.max_portfolio_risk:
            risk_score -= 50
            reasons.append(f"Max exposure reached ({exposure_pct:.0%} >= {self.config.max_portfolio_risk:.0%})")
        elif exposure_pct + self.config.max_position_pct > self.config.max_portfolio_risk:
            risk_score -= 10
            reasons.append(f"Near max exposure ({exposure_pct:.0%} + {self.config.max_position_pct:.0%})")
        
        # 4. Drawdown check (simplified — would need equity curve in production)
        # 5. Position sizing using Kelly
        if signal.side in (SignalSide.BUY, SignalSide.SELL):
            # Kelly fraction of available risk budget
            available_pct = self.config.max_portfolio_risk - exposure_pct
            kelly_size = available_pct * self.config.kelly_fraction * signal.confidence
            signal.position_size_pct = min(kelly_size, self.config.max_position_pct)
        
        # 6. Stop loss / take profit validation
        if signal.entry_price and signal.stop_loss:
            sl_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
            if sl_pct > 0.05:  # More than 5% stop loss
                risk_score -= 15
                reasons.append(f"Wide stop loss ({sl_pct:.1%})")

        # 7. Correlation check
        existing_symbols = [p.get("symbol", "") for p in open_positions if p.get("symbol")]
        if existing_symbols and self._correlation_matrix:
            approved, correlated_with, corr_value = self._correlation_matrix.check_correlation(
                signal.symbol, existing_symbols, self.config.max_correlation
            )
            if not approved:
                if self.config.correlation_override:
                    logger.warning(
                        f"⚠️ HIGH CORRELATION OVERRIDE: {signal.symbol} ↔ {correlated_with} "
                        f"(r={corr_value:.2f}) — override enabled, proceeding"
                    )
                    risk_score -= 10
                    reasons.append(f"High correlation override: {signal.symbol}↔{correlated_with} (r={corr_value:.2f})")
                else:
                    risk_score -= 40
                    reasons.append(
                        f"High correlation: {signal.symbol}↔{correlated_with} (r={corr_value:.2f} > {self.config.max_correlation})"
                    )
        
        # Determine approval
        approved = risk_score >= 50

        if approved:
            risk_signal = SignalSide.BUY
            strength = SignalStrength.BUY if signal.confidence > 0.7 else SignalStrength.NEUTRAL
        else:
            risk_signal = SignalSide.HOLD
            strength = SignalStrength.NEUTRAL

        return AgentOpinion(
            agent_name="risk",
            signal=risk_signal,
            strength=strength,
            confidence=risk_score / 100,
            reasoning=" | ".join(reasons) if reasons else "Risk parameters acceptable",
            indicators={
                "risk_score": risk_score,
                "approved": approved,
                "exposure_pct": round(exposure_pct, 3),
                "position_size_pct": round(signal.position_size_pct, 3),
            },
        )

    def calculate_position_size(self, confidence: float, portfolio_value: float,
                                entry_price: float, stop_loss: float) -> float:
        """Calculate position size using Kelly criterion"""
        # Risk per trade = portfolio * max_position_pct * kelly_fraction * confidence
        risk_amount = portfolio_value * self.config.max_position_pct * self.config.kelly_fraction * confidence
        
        # Size based on stop loss distance
        sl_distance = abs(entry_price - stop_loss)
        if sl_distance <= 0:
            return 0
        
        quantity = risk_amount / sl_distance
        return round(quantity, 8)

    def analyze(self, signal: TradeSignal, portfolio_value: float = 10000,
                open_positions: list = None) -> AgentOpinion:
        """Full risk analysis"""
        if open_positions is None:
            open_positions = []
        return self.check_signal(signal, portfolio_value, open_positions)
