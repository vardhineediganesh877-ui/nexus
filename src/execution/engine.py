"""
NEXUS Execution Engine — Trade execution via CCXT.

Paper trading by default. Real trading only with explicit API keys + paper_mode=false.
All trades logged to SQLite for knowledge accumulation.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime, timezone

import ccxt

from ..models import Trade, TradeSignal, SignalSide
from ..config import NexusConfig
from ..analysis.correlation import CorrelationMatrix

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Multi-exchange trade execution"""

    def __init__(self, config: NexusConfig):
        self.config = config
        self._exchanges: Dict[str, ccxt.Exchange] = {}
        self._db_path = config.data_dir / "trades.db"
        self._correlation = CorrelationMatrix(config)
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for trade tracking"""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                signal_id TEXT,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                exit_price REAL,
                pnl REAL,
                pnl_pct REAL,
                status TEXT DEFAULT 'open',
                paper INTEGER DEFAULT 1,
                timestamp_opened TEXT,
                timestamp_closed TEXT,
                metadata TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_value REAL,
                cash REAL,
                exposure REAL,
                open_positions INTEGER,
                unrealized_pnl REAL
            )
        """)
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
        logger.info(f"Trade DB initialized at {self._db_path}")

    def _get_exchange(self, exchange_id: str) -> ccxt.Exchange:
        """Get exchange instance"""
        if exchange_id in self._exchanges:
            return self._exchanges[exchange_id]

        exchange_class = getattr(ccxt, exchange_id, None)
        if not exchange_class:
            raise ValueError(f"Exchange {exchange_id} not supported")

        ex_cfg = self.config.exchanges.get(exchange_id)
        kwargs = {"rateLimit": 1000}

        if ex_cfg and ex_cfg.is_configured and not self.config.paper_mode:
            kwargs["apiKey"] = ex_cfg.api_key
            kwargs["secret"] = ex_cfg.api_secret
            if ex_cfg.passphrase:
                kwargs["password"] = ex_cfg.passphrase
            if ex_cfg.testnet:
                if exchange_id == "binance":
                    kwargs["urls"] = {"api": {
                        "public": "https://testnet.binance.vision",
                        "private": "https://testnet.binance.vision"
                    }}
        else:
            logger.info(f"Using paper mode for {exchange_id}")

        exchange = exchange_class(kwargs)
        self._exchanges[exchange_id] = exchange
        return exchange

    def execute(self, signal: TradeSignal, quantity: Optional[float] = None) -> Trade:
        """Execute a trade based on a signal (with correlation check)"""
        if signal.side == SignalSide.HOLD:
            logger.info(f"No trade: HOLD signal for {signal.symbol}")
            return Trade(symbol=signal.symbol, exchange=signal.exchange, status="cancelled")

        # CORRELATION CHECK
        open_positions = self.get_open_positions()
        existing_symbols = list(set(p.symbol for p in open_positions))
        
        if existing_symbols:
            approved, correlated_with, corr_value = self._correlation.check_correlation(
                signal.symbol, existing_symbols, self.config.risk.max_correlation
            )
            if not approved:
                if not self.config.risk.correlation_override:
                    logger.warning(
                        f"🚫 TRADE BLOCKED: {signal.symbol} ↔ {correlated_with} "
                        f"correlation={corr_value:.2f} > threshold={self.config.risk.max_correlation}"
                    )
                    trade = Trade(
                        symbol=signal.symbol,
                        exchange=signal.exchange,
                        status="cancelled",
                        metadata={
                            "blocked_reason": "high_correlation",
                            "correlated_with": correlated_with,
                            "correlation": corr_value,
                        },
                    )
                    self._save_trade(trade)
                    return trade
                else:
                    logger.warning(f"⚠️ High correlation override for {signal.symbol}")

        trade = Trade(
            signal_id=signal.id,
            symbol=signal.symbol,
            exchange=signal.exchange,
            side=signal.side,
            paper=self.config.paper_mode,
            timestamp_opened=datetime.now(timezone.utc),
            metadata={
                "confidence": signal.confidence,
                "strength": signal.strength.value,
                "opinions": [o.to_dict() for o in signal.opinions],
            },
        )

        try:
            exchange = self._get_exchange(signal.exchange)
            
            # Get current price
            ticker = exchange.fetch_ticker(signal.symbol)
            trade.entry_price = ticker["last"]
            
            # Calculate quantity
            if quantity is None:
                # Default: $100 position in paper mode
                if self.config.paper_mode:
                    nominal = 100.0
                else:
                    nominal = 1000.0
                trade.quantity = round(nominal / trade.entry_price, 8)
            else:
                trade.quantity = quantity

            trade.stop_loss = signal.stop_loss
            trade.take_profit = signal.take_profit

            if self.config.paper_mode:
                # Paper trade — just log it
                logger.info(
                    f"📝 PAPER TRADE: {trade.side.value} {trade.quantity} {trade.symbol} "
                    f"@ ${trade.entry_price:,.4f}"
                )
            else:
                # Real trade
                order_type = "limit" if signal.stop_loss else "market"
                price = signal.entry_price if order_type == "limit" else None
                
                order = exchange.create_order(
                    symbol=signal.symbol,
                    type=order_type,
                    side=signal.side.value,
                    amount=trade.quantity,
                    price=price,
                )
                trade.metadata["order_id"] = order.get("id")
                logger.info(
                    f"💰 LIVE TRADE: {order.get('id')} {trade.side.value} "
                    f"{trade.quantity} {trade.symbol} @ ${trade.entry_price:,.4f}"
                )

            # Save to DB
            self._save_trade(trade)
            return trade

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            trade.status = "cancelled"
            trade.metadata["error"] = str(e)
            self._save_trade(trade)
            return trade

    def close_position(self, trade: Trade, exit_price: Optional[float] = None) -> Trade:
        """Close an open trade"""
        try:
            exchange = self._get_exchange(trade.exchange)
            
            if exit_price is None:
                ticker = exchange.fetch_ticker(trade.symbol)
                exit_price = ticker["last"]

            trade.exit_price = exit_price
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity
            if trade.side == SignalSide.SELL:
                trade.pnl = -trade.pnl
            trade.pnl_pct = trade.pnl / (trade.entry_price * trade.quantity) * 100
            trade.status = "closed"
            trade.timestamp_closed = datetime.now(timezone.utc)

            if not self.config.paper_mode and trade.metadata.get("order_id"):
                exchange.cancel_order(trade.metadata["order_id"], trade.symbol)

            self._save_trade(trade)
            logger.info(
                f"{'📝' if trade.paper else '💰'} CLOSED {trade.symbol}: "
                f"PnL ${trade.pnl:.2f} ({trade.pnl_pct:+.1f}%)"
            )
            return trade

        except Exception as e:
            logger.error(f"Close position failed: {e}")
            return trade

    def get_open_positions(self) -> List[Trade]:
        """Get all open positions"""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM trades WHERE status = 'open' ORDER BY timestamp_opened DESC"
        ).fetchall()
        conn.close()
        return [self._row_to_trade(r) for r in rows]

    def get_trade_history(self, limit: int = 50) -> List[Trade]:
        """Get recent trade history"""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM trades WHERE status = 'closed' ORDER BY timestamp_closed DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [self._row_to_trade(r) for r in rows]

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio performance summary"""
        conn = sqlite3.connect(str(self._db_path))
        
        # Total trades
        total = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        
        # Open positions
        open_count = conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'open'").fetchone()[0]
        
        # Closed trades performance
        closed = conn.execute("""
            SELECT 
                COUNT(*) as count,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(pnl) as total_pnl,
                AVG(pnl_pct) as avg_pnl_pct,
                MAX(pnl) as best_trade,
                MIN(pnl) as worst_trade
            FROM trades WHERE status = 'closed' AND pnl IS NOT NULL
        """).fetchone()
        
        conn.close()
        
        wins = closed[1] or 0
        count = closed[0] or 0
        win_rate = wins / count if count > 0 else 0
        
        return {
            "total_trades": total,
            "open_positions": open_count,
            "closed_trades": count,
            "wins": wins,
            "losses": count - wins,
            "win_rate": round(win_rate, 3),
            "total_pnl": round(closed[2] or 0, 2),
            "avg_pnl_pct": round(closed[3] or 0, 2),
            "best_trade": round(closed[4] or 0, 2),
            "worst_trade": round(closed[5] or 0, 2),
            "paper_mode": self.config.paper_mode,
        }

    def _save_trade(self, trade: Trade):
        """Save trade to SQLite"""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("""
            INSERT OR REPLACE INTO trades 
            (id, signal_id, symbol, exchange, side, entry_price, quantity,
             stop_loss, take_profit, exit_price, pnl, pnl_pct, status, paper,
             timestamp_opened, timestamp_closed, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.id, trade.signal_id, trade.symbol, trade.exchange,
            trade.side.value, trade.entry_price, trade.quantity,
            trade.stop_loss, trade.take_profit, trade.exit_price,
            trade.pnl, trade.pnl_pct, trade.status, int(trade.paper),
            trade.timestamp_opened.isoformat(),
            trade.timestamp_closed.isoformat() if trade.timestamp_closed else None,
            json.dumps(trade.metadata),
        ))
        conn.commit()
        conn.close()

    @staticmethod
    def _row_to_trade(row: sqlite3.Row) -> Trade:
        """Convert DB row to Trade object"""
        return Trade(
            id=row["id"],
            signal_id=row["signal_id"],
            symbol=row["symbol"],
            exchange=row["exchange"],
            side=SignalSide(row["side"]),
            entry_price=row["entry_price"],
            quantity=row["quantity"],
            stop_loss=row["stop_loss"],
            take_profit=row["take_profit"],
            exit_price=row["exit_price"],
            pnl=row["pnl"],
            pnl_pct=row["pnl_pct"],
            status=row["status"],
            paper=bool(row["paper"]),
            timestamp_opened=datetime.fromisoformat(row["timestamp_opened"]),
            timestamp_closed=datetime.fromisoformat(row["timestamp_closed"]) if row["timestamp_closed"] else None,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )
