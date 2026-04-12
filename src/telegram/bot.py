"""
NEXUS Telegram Bot — Trade signals and portfolio via Telegram.

Integrates with OpenClaw's existing Telegram bot or runs standalone.
Commands: /analyze, /scan, /portfolio, /backtest, /status, /help
"""

import logging
import os
from typing import Optional

from ..config import NexusConfig
from ..analysis.engine import SignalEngine
from ..analysis.backtest import BacktestEngine
from ..execution.engine import ExecutionEngine

logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram bot for NEXUS trading signals"""

    COMMANDS = {
        "analyze": "Multi-agent analysis of a symbol",
        "scan": "Scan exchange for opportunities",
        "portfolio": "Show portfolio summary and positions",
        "backtest": "Backtest a strategy on a symbol",
        "status": "System status and performance",
        "help": "Show this help message",
    }

    def __init__(self, config: NexusConfig):
        self.config = config
        self.signal_engine = SignalEngine(config)
        self.backtest_engine = BacktestEngine()
        self.executor = ExecutionEngine(config)

    def process_message(self, text: str) -> str:
        """Process an incoming Telegram message and return response.
        
        This can be called by OpenClaw's Telegram handler
        or by a standalone python-telegram-bot instance.
        """
        text = text.strip().lstrip("/")
        parts = text.split()
        command = parts[0].lower() if parts else "help"
        args = parts[1:] if len(parts) > 1 else []

        handlers = {
            "analyze": self._cmd_analyze,
            "a": self._cmd_analyze,
            "scan": self._cmd_scan,
            "s": self._cmd_scan,
            "portfolio": self._cmd_portfolio,
            "p": self._cmd_portfolio,
            "backtest": self._cmd_backtest,
            "bt": self._cmd_backtest,
            "status": self._cmd_status,
            "help": self._cmd_help,
            "start": self._cmd_help,
        }

        handler = handlers.get(command, self._cmd_unknown)
        try:
            return handler(args)
        except Exception as e:
            logger.error(f"Command '{command}' failed: {e}")
            return f"❌ Error: {e}"

    def _cmd_analyze(self, args: list) -> str:
        """Analyze a symbol: /analyze BTC/USDT [exchange] [timeframe]"""
        if not args:
            return "Usage: /analyze <symbol> [exchange] [timeframe]\nExample: /analyze BTC/USDT mexc 1h"

        symbol = args[0].upper().replace("-", "/")
        if "/" not in symbol:
            symbol += "/USDT"
        exchange = args[1] if len(args) > 1 else "mexc"
        timeframe = args[2] if len(args) > 2 else "1h"

        signal = self.signal_engine.analyze(symbol, exchange, timeframe)
        return signal.to_telegram()

    def _cmd_scan(self, args: list) -> str:
        """Scan for opportunities: /scan [exchange] [top_n]"""
        exchange = args[0] if args else "mexc"
        top_n = int(args[1]) if len(args) > 1 else 5

        signals = self.signal_engine.scan(exchange, top_n=top_n)

        if not signals:
            return f"🔍 No actionable signals on {exchange.upper()} right now."

        lines = [f"🔍 Top signals on {exchange.upper()}:\n"]
        for i, sig in enumerate(signals, 1):
            emoji = "🟢" if sig.side.value == "buy" else "🔴"
            lines.append(
                f"{i}. {emoji} *{sig.symbol}*\n"
                f"   {sig.side.value.upper()} | Conf: {sig.confidence:.0%} | {sig.strength.value}\n"
            )
        return "\n".join(lines)

    def _cmd_portfolio(self, args: list) -> str:
        """Show portfolio: /portfolio"""
        summary = self.executor.get_portfolio_summary()
        positions = self.executor.get_open_positions()

        mode = "📝 PAPER" if summary["paper_mode"] else "💰 LIVE"
        lines = [
            f"{mode} *Portfolio*\n",
            f"Trades: {summary['total_trades']} | Open: {summary['open_positions']}",
            f"Win Rate: {summary['win_rate']:.1%} | PnL: ${summary['total_pnl']:.2f}",
        ]

        if positions:
            lines.append(f"\n📊 *Open Positions:*")
            for t in positions[:10]:
                emoji = "📈" if t.side.value == "buy" else "📉"
                pnl = ""
                if t.entry_price:
                    lines.append(f"  {emoji} {t.symbol}: {t.quantity:.6f} @ ${t.entry_price:,.2f}")

        if summary["closed_trades"] > 0:
            lines.append(f"\n🏆 Best: ${summary['best_trade']:.2f} | Worst: ${summary['worst_trade']:.2f}")

        return "\n".join(lines)

    def _cmd_backtest(self, args: list) -> str:
        """Backtest: /backtest BTC/USDT [strategy] [--compare]"""
        if not args:
            return "Usage: /backtest <symbol> [strategy] [--compare]\nStrategies: rsi, bollinger, macd, ema_cross, donchian"

        symbol = args[0].upper().replace("-", "/")
        if "/" not in symbol:
            symbol += "/USDT"

        if "--compare" in args or "-c" in args:
            results = self.backtest_engine.compare_strategies(symbol)
            lines = [f"📊 *Strategy Comparison: {symbol}*\n"]
            lines.append("```")
            for r in results:
                lines.append(f"  {r.strategy:<12} {r.total_return_pct:>+6.1f}%  Sharpe {r.sharpe_ratio:>5.2f}  Win {r.win_rate:>4.0f}%  [{r.grade}]")
            lines.append("```")
            return "\n".join(lines)

        strategy = args[1] if len(args) > 1 and not args[1].startswith("-") else "rsi"
        result = self.backtest_engine.backtest(symbol, strategy)

        lines = [
            f"📊 *Backtest: {strategy} × {symbol}*\n",
            f"Return: {result.total_return_pct:+.1f}%",
            f"Sharpe: {result.sharpe_ratio}",
            f"Max DD: {result.max_drawdown_pct:.1f}%",
            f"Win Rate: {result.win_rate:.0f}%",
            f"Trades: {result.total_trades}",
            f"Grade: *{result.grade}*",
            f"Benchmark: {result.benchmark_return_pct:+.1f}%",
        ]
        return "\n".join(lines)

    def _cmd_status(self, args: list) -> str:
        """System status"""
        summary = self.executor.get_portfolio_summary()
        mode = "📝 Paper" if summary["paper_mode"] else "💰 Live"

        lines = [
            "⚡ *NEXUS Status*\n",
            f"Mode: {mode}",
            f"Exchange: MEXC",
            f"Total Trades: {summary['total_trades']}",
            f"Win Rate: {summary['win_rate']:.1%}" if summary['closed_trades'] > 0 else "Win Rate: N/A",
            f"Total PnL: ${summary['total_pnl']:.2f}",
            f"\n🎯 Agents: Technical | Sentiment | Risk",
            f"📊 Strategies: RSI | Bollinger | MACD | EMA | Donchian",
        ]
        return "\n".join(lines)

    def _cmd_help(self, args: list) -> str:
        """Help message"""
        lines = ["⚡ *NEXUS — AI Trading Intelligence*\n"]
        for cmd, desc in self.COMMANDS.items():
            lines.append(f"/{cmd} — {desc}")
        lines.append("\n_Paper trading by default. Zero risk._")
        return "\n".join(lines)

    def _cmd_unknown(self, args: list) -> str:
        return f"❓ Unknown command. Type /help for available commands."
