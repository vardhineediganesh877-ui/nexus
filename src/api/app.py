"""
NEXUS FastAPI Dashboard — REST API for the trading platform.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is on sys.path for proper imports
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.config import NexusConfig
from src.models import TradeSignal, SignalSide, SignalStrength
from src.analysis.engine import SignalEngine
from src.analysis.backtest import BacktestEngine
from src.execution.engine import ExecutionEngine

logger = logging.getLogger(__name__)

# --- Globals initialized in lifespan ---
config: NexusConfig = None
signal_engine: SignalEngine = None
execution_engine: ExecutionEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, signal_engine, execution_engine
    config = NexusConfig.from_env()
    signal_engine = SignalEngine(config)
    execution_engine = ExecutionEngine(config)
    logger.info(f"NEXUS API started — paper_mode={config.paper_mode}")
    yield


app = FastAPI(
    title="NEXUS Trading API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response models ---

class TradeRequest(BaseModel):
    symbol: str
    side: str  # "buy" or "sell"
    quantity: Optional[float] = None
    exchange: str = "binance"


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""


# --- Endpoints ---

@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "paper_mode": config.paper_mode if config else True,
        "exchanges_configured": list(config.exchanges.keys()) if config else [],
    }


@app.get("/api/v1/analyze/{symbol}")
async def analyze(
    symbol: str,
    exchange: str = Query("binance"),
    timeframe: str = Query("1h"),
):
    try:
        signal = signal_engine.analyze(symbol, exchange_id=exchange, timeframe=timeframe)
        return signal.to_dict()
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/scan")
async def scan(
    exchange: str = Query("binance"),
    top_n: int = Query(10, ge=1, le=50),
):
    try:
        signals = await signal_engine.scan_async(exchange_id=exchange, top_n=top_n)
        return {"count": len(signals), "signals": [s.to_dict() for s in signals]}
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/portfolio")
async def portfolio():
    summary = execution_engine.get_portfolio_summary()
    positions = execution_engine.get_open_positions()
    return {
        "summary": summary,
        "open_positions": [t.to_dict() for t in positions],
    }


@app.get("/api/v1/portfolio/history")
async def portfolio_history(limit: int = Query(50, ge=1, le=500)):
    trades = execution_engine.get_trade_history(limit=limit)
    return {"count": len(trades), "trades": [t.to_dict() for t in trades]}


@app.post("/api/v1/trade")
async def trade(req: TradeRequest):
    try:
        side = SignalSide(req.side.lower())
        if side == SignalSide.HOLD:
            raise HTTPException(status_code=400, detail="Cannot execute HOLD signal")

        # Build a minimal signal for execution
        signal = TradeSignal(
            symbol=req.symbol,
            exchange=req.exchange,
            side=side,
            strength=SignalStrength.BUY if side == SignalSide.BUY else SignalStrength.SELL,
            confidence=1.0,
        )
        result = execution_engine.execute(signal, quantity=req.quantity)
        return result.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trade failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/trade/{trade_id}/close")
async def close_trade(trade_id: str):
    positions = execution_engine.get_open_positions()
    trade = next((t for t in positions if t.id == trade_id), None)
    if not trade:
        raise HTTPException(status_code=404, detail=f"Open position {trade_id} not found")
    result = execution_engine.close_position(trade)
    return result.to_dict()


@app.get("/api/v1/backtest/{symbol}")
async def backtest(
    symbol: str,
    strategy: str = Query("rsi"),
    period: int = Query(365, ge=30, le=730),
    exchange: str = Query("mexc"),
    compare: bool = Query(False),
    walk_forward: bool = Query(False),
):
    try:
        bt = BacktestEngine(exchange_id=exchange)

        if compare:
            results = bt.compare_strategies(symbol, period_days=period)
            return {
                "mode": "compare",
                "results": [r.to_dict() for r in results],
                "grades": {r.strategy: r.grade for r in results},
            }

        if walk_forward:
            wf = bt.walk_forward(symbol, strategy=strategy, total_days=period * 2)
            return {"mode": "walk_forward", **wf}

        result = bt.backtest(symbol, strategy=strategy, period_days=period)
        return {"mode": "single", **result.to_dict(), "grade": result.grade}
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/exchanges")
async def exchanges():
    configured = []
    for ex_id, ex_cfg in (config.exchanges or {}).items():
        configured.append({
            "id": ex_id,
            "configured": ex_cfg.is_configured,
            "testnet": ex_cfg.testnet,
        })
    return {"exchanges": configured}


@app.get("/api/v1/market/{symbol:path}/ticker")
async def ticker(symbol: str, exchange: str = Query("binance")):
    try:
        import ccxt
        ex_class = getattr(ccxt, exchange, None)
        if not ex_class:
            raise HTTPException(status_code=400, detail=f"Exchange {exchange} not supported")
        from src.analysis.ccxt_helpers import safe_fetch_ticker
        ex = ex_class({"rateLimit": 1000})
        t = safe_fetch_ticker(ex, symbol)
        return {
            "symbol": t["symbol"],
            "last": t["last"],
            "bid": t.get("bid"),
            "ask": t.get("ask"),
            "high": t.get("high"),
            "low": t.get("low"),
            "volume": t.get("baseVolume"),
            "change_pct": t.get("percentage"),
            "timestamp": t["timestamp"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
