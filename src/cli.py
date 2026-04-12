"""
NEXUS CLI — The main entry point.

pip install -e .
nexus analyze BTC/USDT
nexus scan --exchange binance
nexus portfolio
nexus start --paper
"""

import argparse
import json
import sys
import logging
from pathlib import Path

from .config import NexusConfig
from .analysis.engine import SignalEngine
from .analysis.backtest import BacktestEngine
from .execution.engine import ExecutionEngine

logger = logging.getLogger("nexus")


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_analyze(args):
    """Analyze a single symbol"""
    config = NexusConfig.from_env()
    engine = SignalEngine(config)
    
    signal = engine.analyze(args.symbol, args.exchange, args.timeframe)
    
    if args.json:
        print(json.dumps(signal.to_dict(), indent=2))
    else:
        print(signal.to_telegram())
        print(f"\n📊 Full Analysis:")
        for opinion in signal.opinions:
            print(f"  {opinion.agent_name}: {opinion.reasoning}")


def cmd_scan(args):
    """Scan exchange for opportunities"""
    config = NexusConfig.from_env()
    engine = SignalEngine(config)
    
    print(f"🔍 Scanning {args.exchange} for opportunities...")
    signals = engine.scan(args.exchange, top_n=args.top)
    
    if not signals:
        print("No actionable signals found.")
        return
    
    print(f"\n🚀 Found {len(signals)} signals:\n")
    for i, sig in enumerate(signals, 1):
        print(f"{i}. {sig.to_telegram()}")
        print()
    
    if args.json:
        print(json.dumps([s.to_dict() for s in signals], indent=2))


def cmd_portfolio(args):
    """Show portfolio summary"""
    config = NexusConfig.from_env()
    executor = ExecutionEngine(config)
    
    summary = executor.get_portfolio_summary()
    positions = executor.get_open_positions()
    history = executor.get_trade_history(limit=10)
    
    if args.json:
        output = {"summary": summary, "positions": [t.to_dict() for t in positions]}
        print(json.dumps(output, indent=2))
    else:
        mode = "📝 PAPER" if summary["paper_mode"] else "💰 LIVE"
        print(f"\n{mode} PORTFOLIO\n{'='*40}")
        print(f"Total Trades:    {summary['total_trades']}")
        print(f"Open Positions:  {summary['open_positions']}")
        print(f"Closed Trades:   {summary['closed_trades']}")
        print(f"Win Rate:        {summary['win_rate']:.1%}")
        print(f"Total PnL:       ${summary['total_pnl']:.2f}")
        print(f"Best Trade:      ${summary['best_trade']:.2f}")
        print(f"Worst Trade:     ${summary['worst_trade']:.2f}")
        
        if positions:
            print(f"\n📊 Open Positions:")
            for t in positions:
                emoji = "📈" if t.side.value == "buy" else "📉"
                print(f"  {emoji} {t.symbol}: {t.quantity} @ ${t.entry_price:,.4f}")
        
        if history:
            print(f"\n📜 Recent Trades:")
            for t in history:
                pnl_emoji = "✅" if (t.pnl or 0) > 0 else "❌"
                print(f"  {pnl_emoji} {t.symbol}: ${t.pnl:.2f} ({t.pnl_pct:+.1f}%)")


def cmd_backtest(args):
    """Backtest strategies"""
    engine = BacktestEngine(args.exchange)

    if args.compare:
        results = engine.compare_strategies(args.symbol, period_days=args.period)
        print(f"\n📊 Strategy Comparison: {args.symbol} ({args.period}d)\n")
        print(f"{'Strategy':<15} {'Return':>8} {'Sharpe':>7} {'Win%':>6} {'DD%':>6} {'Grade':>5} {'Benchmark':>10}")
        print("-" * 70)
        for r in results:
            print(f"{r.strategy:<15} {r.total_return_pct:>+7.1f}% {r.sharpe_ratio:>7.2f} {r.win_rate:>5.1f}% {r.max_drawdown_pct:>5.1f}% {r.grade:>5} {r.benchmark_return_pct:>+9.1f}%")
        if args.json:
            print("\n" + json.dumps([r.to_dict() for r in results], indent=2))

    elif args.walk_forward:
        result = engine.walk_forward(args.symbol, args.strategy, period_days=args.period)
        print(f"\n🔬 Walk-Forward: {args.symbol} {args.strategy} ({args.period}d)\n")
        for fold in result["folds"]:
            print(f"  Fold {fold['fold']}: Train {fold['train_return']:+.1f}% → Test {fold['test_return']:+.1f}%")
        print(f"\n  Robustness: {result['robustness_score']:.2f}")
        print(f"  Verdict: {result['verdict']}")
        if args.json:
            print("\n" + json.dumps(result, indent=2))

    else:
        result = engine.backtest(args.symbol, args.strategy, period_days=args.period)
        print(f"\n📊 Backtest: {result.strategy} on {result.symbol} ({result.period})")
        print(f"  Total Return: {result.total_return_pct:+.2f}%")
        print(f"  Sharpe Ratio: {result.sharpe_ratio}")
        print(f"  Max Drawdown: {result.max_drawdown_pct:.2f}%")
        print(f"  Win Rate: {result.win_rate:.1f}%")
        print(f"  Profit Factor: {result.profit_factor}")
        print(f"  Total Trades: {result.total_trades}")
        print(f"  Grade: {result.grade}")
        print(f"  Benchmark (B&H): {result.benchmark_return_pct:+.2f}%")
        if args.json:
            print("\n" + json.dumps(result.to_dict(), indent=2))


def cmd_start(args):
    """Start the NEXUS autonomous scanner"""
    import time
    from datetime import datetime
    
    config = NexusConfig.from_env()
    if args.paper:
        config.paper_mode = True
    
    engine = SignalEngine(config)
    executor = ExecutionEngine(config)
    
    mode = "📝 PAPER" if config.paper_mode else "💰 LIVE"
    print(f"\n🚀 NEXUS Starting — {mode} Mode")
    print(f"📊 Exchange: {args.exchange}")
    print(f"⏱️  Scan interval: {config.scan_interval_minutes} min")
    print(f"📋 Confidence threshold: {config.risk.min_confidence:.0%}")
    print(f"\nPress Ctrl+C to stop\n")
    
    scan_count = 0
    while True:
        try:
            scan_count += 1
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            print(f"\n{'='*50}")
            print(f"Scan #{scan_count} — {now}")
            print(f"{'='*50}")
            
            signals = engine.scan(args.exchange, top_n=args.top)
            
            actionable = [s for s in signals if s.confidence >= config.risk.min_confidence]
            
            if actionable:
                print(f"\n🎯 {len(actionable)} actionable signals:")
                for sig in actionable:
                    print(f"\n{sig.to_telegram()}")
                    
                    # Auto-execute if enabled
                    if args.auto_execute:
                        trade = executor.execute(sig)
                        print(f"  ➡️ Trade executed: {trade.id}")
            else:
                print(f"\nNo signals above {config.risk.min_confidence:.0%} confidence threshold.")
            
            time.sleep(config.scan_interval_minutes * 60)
            
        except KeyboardInterrupt:
            print(f"\n\n🛑 NEXUS stopped after {scan_count} scans.")
            break
        except Exception as e:
            logger.error(f"Scan error: {e}")
            print(f"❌ Error: {e}. Retrying in 60s...")
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(
        prog="nexus",
        description="NEXUS — AI Crypto Trading Intelligence Platform",
    )
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Multi-agent analysis of a symbol")
    p_analyze.add_argument("symbol", help="Trading pair (e.g., BTC/USDT)")
    p_analyze.add_argument("--exchange", "-e", default="mexc", help="Exchange")
    p_analyze.add_argument("--timeframe", "-tf", default="1h", help="Timeframe")
    p_analyze.set_defaults(func=cmd_analyze)

    # scan
    p_scan = subparsers.add_parser("scan", help="Scan exchange for opportunities")
    p_scan.add_argument("--exchange", "-e", default="mexc", help="Exchange")
    p_scan.add_argument("--top", "-n", type=int, default=10, help="Top N coins to scan")
    p_scan.set_defaults(func=cmd_scan)

    # portfolio
    p_portfolio = subparsers.add_parser("portfolio", help="Show portfolio summary")
    p_portfolio.set_defaults(func=cmd_portfolio)

    # start
    p_start = subparsers.add_parser("start", help="Start autonomous scanner")
    p_start.add_argument("--exchange", "-e", default="mexc", help="Exchange")
    p_start.add_argument("--paper", action="store_true", default=True, help="Paper mode (default)")
    p_start.add_argument("--live", action="store_true", help="Live trading (requires API keys)")
    p_start.add_argument("--top", "-n", type=int, default=10, help="Top N coins per scan")
    p_start.add_argument("--auto-execute", action="store_true", help="Auto-execute signals")
    p_start.set_defaults(func=cmd_start)

    # backtest
    p_bt = subparsers.add_parser("backtest", help="Backtest strategies")
    p_bt.add_argument("symbol", help="Trading pair (e.g., BTC/USDT)")
    p_bt.add_argument("--strategy", "-s", default="rsi", help="Strategy name")
    p_bt.add_argument("--compare", action="store_true", help="Compare all strategies")
    p_bt.add_argument("--walk-forward", action="store_true", help="Walk-forward validation")
    p_bt.add_argument("--exchange", "-e", default="mexc")
    p_bt.add_argument("--period", "-p", type=int, default=365, help="Days of data")
    p_bt.set_defaults(func=cmd_backtest)

    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    setup_logging("DEBUG" if args.verbose else "INFO")
    args.func(args)


if __name__ == "__main__":
    main()
