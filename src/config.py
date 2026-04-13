"""
NEXUS Configuration Management

All sensitive data in .env, never hardcoded.
Paper mode ON by default.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from .models import TTLConfig, StrategyType

DEFAULT_CONFIG_PATH = Path.home() / ".nexus" / "config.json"
DEFAULT_DATA_DIR = Path.home() / ".nexus" / "data"


@dataclass
class ExchangeConfig:
    """Exchange connection settings"""
    id: str  # ccxt exchange id: 'binance', 'bybit', 'okx', etc.
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""  # OKX requires this
    testnet: bool = True  # Use testnet/sandbox by default

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_secret)


@dataclass
class RiskConfig:
    """Risk management parameters"""
    max_position_pct: float = 0.05  # Max 5% of portfolio per position
    max_portfolio_risk: float = 0.20  # Max 20% total exposure
    max_drawdown_pct: float = 0.15  # Stop all trading at 15% drawdown
    kelly_fraction: float = 0.5  # Half-Kelly sizing
    min_confidence: float = 0.65  # Minimum signal confidence to trade
    stop_loss_pct: float = 0.02  # Default 2% stop loss
    take_profit_pct: float = 0.05  # Default 5% take profit
    max_correlation: float = 0.7       # Reject if > 0.7 correlation with existing position
    correlation_override: bool = False  # Allow override (with warning log)
    correlation_window_days: int = 30   # Rolling window for correlation calculation


@dataclass
class AnalysisConfig:
    """Analysis engine settings"""
    timeframes: List[str] = field(default_factory=lambda: ["15m", "1h", "4h", "1D"])
    min_volume_usd: float = 1_000_000  # Minimum 24h volume to analyze
    reddit_subreddits: List[str] = field(default_factory=lambda: [
        "cryptocurrency", "bitcoin", "ethereum", "altcoin", "defi"
    ])
    news_limit: int = 10
    sentiment_window_hours: int = 24


@dataclass
class NexusConfig:
    """Master configuration"""
    paper_mode: bool = True
    data_dir: Path = DEFAULT_DATA_DIR
    exchanges: Dict[str, ExchangeConfig] = field(default_factory=dict)
    risk: RiskConfig = field(default_factory=RiskConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    log_level: str = "INFO"
    scan_interval_minutes: int = 30
    ttl: TTLConfig = field(default_factory=TTLConfig)

    @classmethod
    def from_env(cls) -> "NexusConfig":
        """Load config from environment variables and optional config file"""
        config = cls()

        # Paper mode from env
        config.paper_mode = os.getenv("NEXUS_PAPER", "true").lower() == "true"

        # Data directory
        config.data_dir = Path(os.getenv("NEXUS_DATA_DIR", str(DEFAULT_DATA_DIR)))
        config.data_dir.mkdir(parents=True, exist_ok=True)

        # Risk settings
        config.risk.max_position_pct = float(os.getenv("NEXUS_MAX_POSITION_PCT", "0.05"))
        config.risk.max_portfolio_risk = float(os.getenv("NEXUS_MAX_PORTFOLIO_RISK", "0.20"))
        config.risk.max_drawdown_pct = float(os.getenv("NEXUS_MAX_DRAWDOWN_PCT", "0.15"))
        config.risk.kelly_fraction = float(os.getenv("NEXUS_KELLY_FRACTION", "0.5"))
        config.risk.min_confidence = float(os.getenv("NEXUS_MIN_CONFIDENCE", "0.65"))

        # Exchanges from env (CCXT_BINANCE_KEY, CCXT_BINANCE_SECRET, etc.)
        for ex_id in ["binance", "bybit", "okx", "coinbase"]:
            key = os.getenv(f"CCXT_{ex_id.upper()}_KEY", "")
            secret = os.getenv(f"CCXT_{ex_id.upper()}_SECRET", "")
            if key or secret:
                config.exchanges[ex_id] = ExchangeConfig(
                    id=ex_id,
                    api_key=key,
                    api_secret=secret,
                    passphrase=os.getenv(f"CCXT_{ex_id.upper()}_PASSPHRASE", ""),
                    testnet=os.getenv(f"CCXT_{ex_id.upper()}_TESTNET", "true").lower() == "true",
                )

        # Telegram
        config.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        config.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

        # Analysis
        config.analysis.timeframes = os.getenv(
            "NEXUS_TIMEFRAMES", "15m,1h,4h,1D"
        ).split(",")

        config.log_level = os.getenv("NEXUS_LOG_LEVEL", "INFO")
        config.scan_interval_minutes = int(os.getenv("NEXUS_SCAN_INTERVAL", "30"))

        # TTL config
        ttl_type = os.getenv("NEXUS_STRATEGY_TYPE", "swing").lower()
        config.ttl = TTLConfig(
            strategy_type=StrategyType(ttl_type),
            timeout_seconds=float(os.getenv("NEXUS_TTL_SECONDS", "5.0")),
            custom_timeout=float(os.getenv("NEXUS_TTL_CUSTOM", "0")) or None,
        )

        # Correlation config
        config.risk.max_correlation = float(os.getenv("NEXUS_MAX_CORRELATION", "0.7"))
        config.risk.correlation_override = os.getenv("NEXUS_CORRELATION_OVERRIDE", "false").lower() == "true"
        config.risk.correlation_window_days = int(os.getenv("NEXUS_CORRELATION_WINDOW", "30"))

        return config

    @property
    def default_exchange(self) -> Optional[str]:
        """First configured exchange, or None"""
        for ex_id, ex_config in self.exchanges.items():
            if ex_config.is_configured:
                return ex_id
        return None
