"""
NEXUS Portfolio Correlation Matrix

Computes rolling 30-day Pearson correlation between all symbols in portfolio.
Rejects new positions that are >0.7 correlated with any existing position.
"""

import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from itertools import combinations

import numpy as np

from ..config import NexusConfig, RiskConfig

logger = logging.getLogger(__name__)


class CorrelationMatrix:
    """Rolling correlation matrix for portfolio symbols"""

    def __init__(self, config: NexusConfig):
        self.config = config
        self._db_path = config.data_dir / "trades.db"
        self._cache: Dict[Tuple[str, str], float] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=6)  # Re-read from DB every 6h

    def _init_db(self):
        """Create correlation_matrix table if not exists"""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS correlation_matrix (
                symbol_a TEXT NOT NULL,
                symbol_b TEXT NOT NULL,
                correlation REAL NOT NULL,
                computed_at TEXT NOT NULL,
                PRIMARY KEY (symbol_a, symbol_b)
            )
        """)
        conn.commit()
        conn.close()

    def compute_correlations(self, price_data: Dict[str, List[float]]) -> None:
        """
        Compute Pearson correlations for all symbol pairs and persist to DB.
        
        Args:
            price_data: Dict mapping symbol -> list of daily returns (30 days).
                        e.g. {"BTC/USDT": [0.01, -0.02, ...], "ETH/USDT": [...]}
        """
        symbols = list(price_data.keys())
        conn = sqlite3.connect(str(self._db_path))
        now = datetime.now(timezone.utc).isoformat()

        for sym_a, sym_b in combinations(symbols, 2):
            returns_a = np.array(price_data[sym_a])
            returns_b = np.array(price_data[sym_b])
            
            # Align lengths
            min_len = min(len(returns_a), len(returns_b))
            if min_len < 10:
                logger.warning(f"Insufficient data for {sym_a}/{sym_b}: {min_len} days")
                continue
            
            returns_a = returns_a[-min_len:]
            returns_b = returns_b[-min_len:]
            
            corr = float(np.corrcoef(returns_a, returns_b)[0, 1])
            
            # Normalize symbol order (alphabetical)
            a, b = sorted([sym_a, sym_b])
            
            conn.execute("""
                INSERT OR REPLACE INTO correlation_matrix (symbol_a, symbol_b, correlation, computed_at)
                VALUES (?, ?, ?, ?)
            """, (a, b, round(corr, 6), now))
            
            self._cache[(a, b)] = corr

        conn.commit()
        conn.close()
        self._cache_timestamp = datetime.now(timezone.utc)
        logger.info(f"Updated correlation matrix for {len(symbols)} symbols")

    def get_correlation(self, symbol_a: str, symbol_b: str) -> Optional[float]:
        """
        Get cached correlation between two symbols.
        Returns None if no data available.
        """
        a, b = sorted([symbol_a, symbol_b])
        
        # Check memory cache
        if (a, b) in self._cache:
            return self._cache[(a, b)]
        
        # Check if cache is stale
        if self._cache_timestamp and datetime.now(timezone.utc) - self._cache_timestamp > self._cache_ttl:
            self._load_cache_from_db()
            if (a, b) in self._cache:
                return self._cache[(a, b)]
        
        return None

    def check_correlation(self, symbol: str, existing_positions: List[str],
                          threshold: float = 0.7) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Check if adding `symbol` would exceed correlation threshold with any existing position.
        
        Args:
            symbol: New symbol to add
            existing_positions: List of symbols already in portfolio
            threshold: Max allowed correlation (default 0.7)
        
        Returns:
            (approved: bool, correlated_with: Optional[str], correlation: Optional[float])
        """
        if not existing_positions:
            return True, None, None

        for existing in existing_positions:
            corr = self.get_correlation(symbol, existing)
            if corr is not None and corr > threshold:
                return False, existing, corr

        return True, None, None

    def get_portfolio_heatmap(self) -> Dict[Tuple[str, str], float]:
        """Return full correlation matrix for all cached pairs"""
        if not self._cache or (
            self._cache_timestamp and 
            datetime.now(timezone.utc) - self._cache_timestamp > self._cache_ttl
        ):
            self._load_cache_from_db()
        return dict(self._cache)

    def _load_cache_from_db(self) -> None:
        """Load all correlations from DB into memory cache"""
        conn = sqlite3.connect(str(self._db_path))
        try:
            rows = conn.execute(
                "SELECT symbol_a, symbol_b, correlation FROM correlation_matrix"
            ).fetchall()
            self._cache = {(r[0], r[1]): r[2] for r in rows}
            self._cache_timestamp = datetime.now(timezone.utc)
            logger.debug(f"Loaded {len(self._cache)} correlation pairs from DB")
        finally:
            conn.close()
