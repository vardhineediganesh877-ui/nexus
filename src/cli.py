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

    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    setup_logging("DEBUG" if args.verbose else "INFO")
    args.func(args)


if __name__ == "__main__":
    main()
