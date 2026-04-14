"""
NEXUS Backtesting Engine

Walk-forward backtesting to detect overfitting.
6 strategies with institutional-grade metrics.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

import ccxt

from ..models import BacktestResult, Trade, SignalSide
from .ccxt_helpers import retry_exchange, safe_fetch_ohlcv

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Strategy backtesting with anti-overfitting validation"""

    STRATEGIES = ["rsi", "bollinger", "macd", "ema_cross", "supertrend", "donchian"]

    def __init__(self, exchange_id: str = "mexc"):
        self.exchange_id = exchange_id
        self._exchange = None  # Cached exchange instance

    @retry_exchange(max_retries=2, base_delay=2.0)
    def _get_ohlcv(self, symbol: str, timeframe: str = "1d", limit: int = 500):
        """Fetch historical data (cached exchange instance)"""
        if self._exchange is None:
            self._exchange = ccxt.__dict__[self.exchange_id]({"rateLimit": 1000})
        return safe_fetch_ohlcv(self._exchange, symbol, timeframe, limit=limit)

    def _run_strategy(self, closes: List[float], strategy: str) -> List[dict]:
        """Run a strategy on closing prices, return list of trades"""
        trades = []

        if strategy == "rsi":
            trades = self._strategy_rsi(closes)
        elif strategy == "bollinger":
            trades = self._strategy_bollinger(closes)
        elif strategy == "macd":
            trades = self._strategy_macd(closes)
        elif strategy == "ema_cross":
            trades = self._strategy_ema_cross(closes)
        elif strategy == "donchian":
            trades = self._strategy_donchian(closes)
        else:
            logger.warning(f"Unknown strategy: {strategy}")

        return trades

    def _strategy_rsi(self, closes: List[float], period: int = 14) -> List[dict]:
        """RSI: Buy oversold (<30), Sell overbought (>70) — O(n) with running averages"""
        trades = []
        position = None

        if len(closes) < period + 1:
            return trades

        # Seed running averages from first period
        deltas = [closes[j] - closes[j-1] for j in range(1, period + 1)]
        avg_gain = sum(max(d, 0) for d in deltas) / period
        avg_loss = sum(abs(min(d, 0)) for d in deltas) / period

        for i in range(period + 1, len(closes)):
            # Update running averages incrementally
            delta = closes[i] - closes[i-1]
            gain = max(delta, 0)
            loss = abs(min(delta, 0))
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

            rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 100

            if position is None and rsi < 30:
                position = {"entry": closes[i], "entry_idx": i}
            elif position is not None and rsi > 70:
                pnl_pct = (closes[i] - position["entry"]) / position["entry"] * 100
                trades.append({
                    "entry": position["entry"],
                    "exit": closes[i],
                    "pnl_pct": pnl_pct,
                    "hold_bars": i - position["entry_idx"],
                })
                position = None

        # Close any open position at end
        if position:
            pnl_pct = (closes[-1] - position["entry"]) / position["entry"] * 100
            trades.append({
                "entry": position["entry"],
                "exit": closes[-1],
                "pnl_pct": pnl_pct,
                "hold_bars": len(closes) - position["entry_idx"],
            })

        return trades

    def _strategy_bollinger(self, closes: List[float], period: int = 20, std: float = 2.0) -> List[dict]:
        """Bollinger: Buy at lower band, sell at middle — O(n) with running stats"""
        trades = []
        position = None

        if len(closes) < period:
            return trades

        # Seed with first window
        window_sum = sum(closes[:period])
        window_sq_sum = sum(p * p for p in closes[:period])

        for i in range(period, len(closes)):
            # Update running sums incrementally
            if i > period:
                window_sum += closes[i-1] - closes[i-period-1]
                window_sq_sum += closes[i-1]**2 - closes[i-period-1]**2

            middle = window_sum / period
            variance = window_sq_sum / period - middle * middle
            sd = max(variance, 0) ** 0.5
            upper = middle + std * sd
            lower = middle - std * sd

            if position is None and closes[i] <= lower:
                position = {"entry": closes[i], "entry_idx": i}
            elif position is not None and closes[i] >= middle:
                pnl_pct = (closes[i] - position["entry"]) / position["entry"] * 100
                trades.append({
                    "entry": position["entry"],
                    "exit": closes[i],
                    "pnl_pct": pnl_pct,
                    "hold_bars": i - position["entry_idx"],
                })
                position = None

        if position:
            pnl_pct = (closes[-1] - position["entry"]) / position["entry"] * 100
            trades.append({"entry": position["entry"], "exit": closes[-1], "pnl_pct": pnl_pct, "hold_bars": len(closes) - position["entry_idx"]})

        return trades

    def _strategy_macd(self, closes: List[float]) -> List[dict]:
        """MACD: Buy on golden cross, sell on death cross (O(n) with incremental EMA)"""
        trades = []
        position = None
        prev_hist = None

        if len(closes) < 27:
            return trades

        # Incremental EMA computation — O(n) not O(n²)
        def _ema_series(prices, period):
            """Return full EMA series from prices"""
            if len(prices) < period:
                return []
            mult = 2 / (period + 1)
            ema = sum(prices[:period]) / period
            result = [ema]
            for p in prices[period:]:
                ema = (p - ema) * mult + ema
                result.append(ema)
            return result

        ema12_series = _ema_series(closes, 12)
        ema26_series = _ema_series(closes, 26)

        # MACD line (align series — ema26 starts later)
        offset = len(ema12_series) - len(ema26_series)
        macd_history = [ema12_series[i + offset] - ema26_series[i] for i in range(len(ema26_series))]

        # Signal line (incremental EMA of MACD)
        signal_series = _ema_series(macd_history, 9) if len(macd_history) >= 9 else []

        for i in range(len(signal_series)):
            macd_idx = i + 9 - 1  # index into macd_history
            hist = macd_history[macd_idx] - signal_series[i]

            if prev_hist is not None:
                idx = 26 + macd_idx  # 26 from slow EMA offset + macd_idx
                if idx >= len(closes):
                    prev_hist = hist
                    continue
                if prev_hist <= 0 and hist > 0 and position is None:  # Golden cross
                    position = {"entry": closes[idx], "entry_idx": idx}
                elif prev_hist >= 0 and hist < 0 and position is not None:  # Death cross
                    pnl_pct = (closes[idx] - position["entry"]) / position["entry"] * 100
                    trades.append({"entry": position["entry"], "exit": closes[idx], "pnl_pct": pnl_pct, "hold_bars": idx - position["entry_idx"]})
                    position = None

            prev_hist = hist

        if position:
            pnl_pct = (closes[-1] - position["entry"]) / position["entry"] * 100
            trades.append({"entry": position["entry"], "exit": closes[-1], "pnl_pct": pnl_pct, "hold_bars": len(closes) - position["entry_idx"]})

        return trades

    def _strategy_ema_cross(self, closes: List[float], fast: int = 20, slow: int = 50) -> List[dict]:
        """EMA Cross: Buy when fast > slow, sell on reversal (O(n) with incremental EMA)"""
        trades = []
        position = None
        prev_fast = None
        prev_slow = None

        if len(closes) < slow:
            return trades

        # Precompute EMA series incrementally
        def _ema_series(prices, period):
            if len(prices) < period:
                return []
            mult = 2 / (period + 1)
            ema = sum(prices[:period]) / period
            result = [ema]
            for p in prices[period:]:
                ema = (p - ema) * mult + ema
                result.append(ema)
            return result

        fast_series = _ema_series(closes, fast)
        slow_series = _ema_series(closes, slow)
        offset = len(fast_series) - len(slow_series)

        for i in range(len(slow_series)):
            fast_ema = fast_series[i + offset]
            slow_ema = slow_series[i]
            idx = slow - 1 + i  # actual index into closes

            if prev_fast is not None:
                if prev_fast <= prev_slow and fast_ema > slow_ema and position is None:
                    position = {"entry": closes[idx], "entry_idx": idx}
                elif prev_fast >= prev_slow and fast_ema < slow_ema and position is not None:
                    pnl_pct = (closes[idx] - position["entry"]) / position["entry"] * 100
                    trades.append({"entry": position["entry"], "exit": closes[idx], "pnl_pct": pnl_pct, "hold_bars": idx - position["entry_idx"]})
                    position = None

            prev_fast, prev_slow = fast_ema, slow_ema

        if position:
            pnl_pct = (closes[-1] - position["entry"]) / position["entry"] * 100
            trades.append({"entry": position["entry"], "exit": closes[-1], "pnl_pct": pnl_pct, "hold_bars": len(closes) - position["entry_idx"]})

        return trades

    def _strategy_donchian(self, closes: List[float], period: int = 20) -> List[dict]:
        """Donchian Channel: Buy on breakout above high, sell on breakout below low"""
        trades = []
        position = None

        for i in range(period, len(closes)):
            window = closes[i-period:i]
            high = max(window)
            low = min(window)

            if position is None and closes[i] > high:
                position = {"entry": closes[i], "entry_idx": i}
            elif position is not None and closes[i] < low:
                pnl_pct = (closes[i] - position["entry"]) / position["entry"] * 100
                trades.append({"entry": position["entry"], "exit": closes[i], "pnl_pct": pnl_pct, "hold_bars": i - position["entry_idx"]})
                position = None

        if position:
            pnl_pct = (closes[-1] - position["entry"]) / position["entry"] * 100
            trades.append({"entry": position["entry"], "exit": closes[-1], "pnl_pct": pnl_pct, "hold_bars": len(closes) - position["entry_idx"]})

        return trades

    @staticmethod
    def _ema(prices: List[float], period: int) -> float:
        if len(prices) < period:
            return sum(prices) / len(prices)
        mult = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        for p in prices[period:]:
            ema = (p - ema) * mult + ema
        return ema

    def backtest(self, symbol: str, strategy: str = "rsi",
                 timeframe: str = "1d", period_days: int = 365) -> BacktestResult:
        """Run backtest for a single strategy"""
        ohlcv = self._get_ohlcv(symbol, timeframe, limit=period_days)
        closes = [c[4] for c in ohlcv]

        trades = self._run_strategy(closes, strategy)
        if not trades:
            return BacktestResult(symbol=symbol, strategy=strategy, period=f"{period_days}d")

        # Calculate metrics
        pnls = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_return = sum(pnls)
        win_rate = len(wins) / len(pnls) if pnls else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.001
        profit_factor = avg_win * len(wins) / (avg_loss * len(losses)) if losses else float('inf')

        # Max drawdown
        equity = [100]
        for p in pnls:
            equity.append(equity[-1] * (1 + p / 100))
        peak = equity[0]
        max_dd = 0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (simplified)
        import statistics
        avg_pnl = statistics.mean(pnls) if pnls else 0
        std_pnl = statistics.stdev(pnls) if len(pnls) > 1 else 0.001
        sharpe = (avg_pnl / std_pnl) * (365 ** 0.5) if std_pnl > 0 else 0

        # Benchmark (buy and hold)
        benchmark = (closes[-1] - closes[0]) / closes[0] * 100

        # Calmar ratio
        calmar = total_return / max_dd if max_dd > 0 else 0

        # Expectancy
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

        result = BacktestResult(
            symbol=symbol,
            strategy=strategy,
            period=f"{period_days}d",
            total_return_pct=round(total_return, 2),
            sharpe_ratio=round(sharpe, 2),
            max_drawdown_pct=round(max_dd * 100, 2),
            win_rate=round(win_rate * 100, 2),
            profit_factor=round(min(profit_factor, 99.99), 2),
            total_trades=len(trades),
            avg_trade_return_pct=round(avg_pnl, 2),
            best_trade_pct=round(max(pnls), 2),
            worst_trade_pct=round(min(pnls), 2),
            calmar_ratio=round(calmar, 2),
            expectancy=round(expectancy, 2),
            benchmark_return_pct=round(benchmark, 2),
            trades=trades,
        )

        return result

    def compare_strategies(self, symbol: str, timeframe: str = "1d",
                          period_days: int = 365) -> List[BacktestResult]:
        """Run all strategies and rank them"""
        results = []
        for strategy in self.STRATEGIES:
            try:
                result = self.backtest(symbol, strategy, timeframe, period_days)
                results.append(result)
                logger.info(f"{strategy}: {result.total_return_pct:+.1f}% (Sharpe {result.sharpe_ratio:.2f}, Grade {result.grade})")
            except Exception as e:
                logger.error(f"Backtest {strategy} failed: {e}")

        results.sort(key=lambda r: r.sharpe_ratio, reverse=True)
        return results

    def walk_forward(self, symbol: str, strategy: str = "rsi",
                    timeframe: str = "1d", total_days: int = 730,
                    n_splits: int = 3, train_ratio: float = 0.7) -> Dict:
        """Walk-forward backtest to detect overfitting"""
        ohlcv = self._get_ohlcv(symbol, timeframe, limit=total_days)
        closes = [c[4] for c in ohlcv]

        fold_size = len(closes) // n_splits
        folds = []

        for i in range(n_splits):
            start = i * fold_size
            end = min(start + fold_size, len(closes))
            fold_data = closes[start:end]

            if len(fold_data) < 50:
                continue

            split_idx = int(len(fold_data) * train_ratio)
            train_data = fold_data[:split_idx]
            test_data = fold_data[split_idx:]

            train_trades = self._run_strategy(train_data, strategy)
            test_trades = self._run_strategy(test_data, strategy)

            train_return = sum(t["pnl_pct"] for t in train_trades) if train_trades else 0
            test_return = sum(t["pnl_pct"] for t in test_trades) if test_trades else 0

            folds.append({
                "fold": i + 1,
                "train_return": round(train_return, 2),
                "test_return": round(test_return, 2),
                "train_trades": len(train_trades),
                "test_trades": len(test_trades),
            })

        # Calculate robustness score
        avg_train = sum(f["train_return"] for f in folds) / len(folds) if folds else 0
        avg_test = sum(f["test_return"] for f in folds) / len(folds) if folds else 0
        robustness = avg_test / avg_train if avg_train != 0 else 0

        if robustness >= 0.8:
            verdict = "ROBUST — no overfitting detected"
        elif robustness >= 0.5:
            verdict = "MODERATE — some degradation out-of-sample"
        elif robustness >= 0.2:
            verdict = "WEAK — significant degradation, possibly overfitted"
        else:
            verdict = "OVERFITTED — fails on unseen data, do not trade live"

        return {
            "symbol": symbol,
            "strategy": strategy,
            "folds": folds,
            "avg_train_return": round(avg_train, 2),
            "avg_test_return": round(avg_test, 2),
            "robustness_score": round(robustness, 3),
            "verdict": verdict,
        }
