"""
NEXUS Knowledge Engine — Connects trades to GBrain knowledge graph.

Every trade and analysis enriches the brain. Over time, the system learns
which symbols, timeframes, and strategies work best.
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import Trade, TradeSignal

GBRAIN_CLI = "/root/.bun/bin/gbrain"
BRAIN_DIR = Path("/root/.openclaw/workspace/brain")
CONCEPTS_DIR = BRAIN_DIR / "concepts"
COMPANIES_DIR = BRAIN_DIR / "companies"


class KnowledgeEngine:
    """Bridges NEXUS trades/analysis with the GBrain knowledge graph."""

    def __init__(self, brain_dir: str = "/root/.openclaw/workspace/brain/"):
        self.brain_dir = Path(brain_dir)
        self.concepts_dir = self.brain_dir / "concepts"
        self.companies_dir = self.brain_dir / "companies"
        self._gbrain = GBRAIN_CLI
        # Ensure dirs exist
        self.concepts_dir.mkdir(parents=True, exist_ok=True)
        self.companies_dir.mkdir(parents=True, exist_ok=True)

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _slug(symbol: str) -> str:
        """BTC/USDT → btc-usdt"""
        return symbol.lower().replace("/", "-").replace(" ", "-")

    def _page_path(self, symbol: str) -> Path:
        return self.concepts_dir / f"{self._slug(symbol)}.md"

    def _gbrain_call(self, operation: str, payload: dict) -> Optional[dict]:
        """Call a gbrain operation via CLI."""
        try:
            result = subprocess.run(
                [self._gbrain, "call", operation, json.dumps(payload)],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout.strip())
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
            pass
        return None

    def _gbrain_search(self, query: str) -> list:
        """Search gbrain for pages matching query."""
        try:
            result = subprocess.run(
                [self._gbrain, "search", query],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                # gbrain search returns results as text lines
                return result.stdout.strip().split("\n")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return []

    def _write_or_append_page(self, path: Path, section: str, content: str) -> None:
        """Append a timeline entry to a brain page, creating it if needed."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        entry = f"\n### {timestamp}\n{content}\n"

        if path.exists():
            # Append timeline entry at end
            existing = path.read_text()
            # Find or create ## NEXUS Timeline section
            marker = "## NEXUS Timeline"
            if marker in existing:
                updated = existing + entry
            else:
                updated = existing + f"\n{marker}\n{entry}"
            path.write_text(updated)
        else:
            # Create new page
            title = path.stem.upper().replace("-", "/")
            path.write_text(
                f"# {title}\n\n> Auto-tracked by NEXUS Knowledge Engine.\n\n"
                f"## NEXUS Timeline\n{entry}\n"
            )

    # ── public API ──────────────────────────────────────────────────

    def log_trade(self, trade: Trade, signal: TradeSignal) -> None:
        """After a trade is executed, log it to GBrain."""
        symbol = trade.symbol
        page = self._page_path(symbol)
        slug = self._slug(symbol)

        # Build timeline entry
        pnl_str = ""
        if trade.pnl is not None:
            pnl_str = f" | P&L: ${trade.pnl:+.2f}"
            if trade.pnl_pct is not None:
                pnl_str += f" ({trade.pnl_pct:+.1f}%)"

        side_str = trade.side.value.upper()
        status_str = trade.status.upper()
        paper_tag = " [PAPER]" if trade.paper else ""
        strategies = [o.agent_name for o in signal.opinions]
        strategy_str = ", ".join(strategies) if strategies else "unknown"

        content = (
            f"- **{side_str} {symbol}** @ ${trade.entry_price:,.4f}{paper_tag}\n"
            f"  - Exchange: {trade.exchange}\n"
            f"  - Strategy: {strategy_str}\n"
            f"  - Signal confidence: {signal.confidence:.0%}\n"
            f"  - Status: {status_str}{pnl_str}\n"
            f"  - Timeframe: {signal.timeframe}\n"
        )
        if trade.stop_loss:
            content += f"  - Stop: ${trade.stop_loss:,.4f}\n"
        if trade.take_profit:
            content += f"  - Target: ${trade.take_profit:,.4f}\n"

        self._write_or_append_page(page, "trade", content)

        # Update KG triples
        self._gbrain_call("kg_add_entity", {"name": symbol, "type": "crypto_pair"})
        if trade.exchange:
            self._gbrain_call("kg_add_triple", {
                "subject": symbol,
                "predicate": "TRADED_ON",
                "object": trade.exchange,
            })
        self._gbrain_call("kg_add_triple", {
            "subject": symbol,
            "predicate": "TRADED_WITH",
            "object": strategy_str,
        })

        # Track win/loss as a triple attribute
        if trade.pnl is not None:
            outcome = "WIN" if trade.pnl > 0 else "LOSS"
            self._gbrain_call("kg_add_triple", {
                "subject": symbol,
                "predicate": f"TRADE_RESULT_{outcome}",
                "object": f"{trade.id}@{datetime.utcnow().strftime('%Y-%m-%d')}",
            })

    def query_pattern(self, symbol: str) -> dict:
        """Query GBrain for historical patterns on a symbol."""
        page = self._page_path(symbol)
        result: Dict[str, Any] = {
            "symbol": symbol,
            "known": False,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_pnl_pct": 0.0,
            "best_timeframe": None,
            "notes": [],
            "recent_entries": [],
        }

        if not page.exists():
            return result

        result["known"] = True
        text = page.read_text()

        # Parse timeline entries for trade stats
        pnl_values = []
        timeframe_counts: Dict[str, int] = {}
        entries = []

        for line in text.split("\n"):
            if "TRADE_RESULT_WIN" in line or "$" in line and "P&L" in line:
                if "P&L" in line:
                    # Extract pnl pct
                    try:
                        pct_str = line.split("(")[-1].split("%)")[0]
                        pnl_values.append(float(pct_str.replace("+", "")))
                    except (IndexError, ValueError):
                        pass
                if "TRADE_RESULT_WIN" in line:
                    result["wins"] += 1
                    result["trades"] += 1
                elif "TRADE_RESULT_LOSS" in line:
                    result["losses"] += 1
                    result["trades"] += 1
            if "Timeframe:" in line:
                tf = line.split("Timeframe:")[1].strip()
                timeframe_counts[tf] = timeframe_counts.get(tf, 0) + 1

        # Calculate stats
        total = result["wins"] + result["losses"]
        if total > 0:
            result["win_rate"] = round(result["wins"] / total, 3)
        if pnl_values:
            result["avg_pnl_pct"] = round(sum(pnl_values) / len(pnl_values), 2)
        if timeframe_counts:
            result["best_timeframe"] = max(timeframe_counts, key=timeframe_counts.get)

        # Recent entries (last 5 timeline blocks)
        blocks = text.split("### ")
        result["recent_entries"] = blocks[-5:] if len(blocks) > 1 else []

        # Also try KG query for additional data
        kg_result = self._gbrain_call("kg_query", {"subject": symbol})
        if kg_result:
            result["kg_data"] = kg_result

        return result

    def market_intelligence(self) -> list:
        """Get overall market intelligence from GBrain."""
        intelligence = []

        # Scan all concept pages for NEXUS-tracked symbols
        if not self.concepts_dir.exists():
            return intelligence

        for page_file in self.concepts_dir.glob("*.md"):
            text = page_file.read_text()
            if "NEXUS Timeline" not in text:
                continue

            symbol = page_file.stem.upper().replace("-", "/")
            wins = text.count("TRADE_RESULT_WIN")
            losses = text.count("TRADE_RESULT_LOSS")
            # Also count inline P&L markers
            total_trades = wins + losses
            if total_trades == 0:
                # Try counting from timeline entries
                total_trades = text.count("BUY") + text.count("SELL") - text.count("BUY/SELL")

            if total_trades > 0:
                win_rate = wins / total_trades if total_trades else 0
                intelligence.append({
                    "symbol": symbol,
                    "trades": total_trades,
                    "wins": wins,
                    "win_rate": round(win_rate, 3),
                    "page": str(page_file),
                })

        # Sort by win rate desc
        intelligence.sort(key=lambda x: x["win_rate"], reverse=True)
        return intelligence

    def log_analysis(self, signal: TradeSignal) -> None:
        """Log analysis results (even without a trade) to build knowledge."""
        symbol = signal.symbol
        page = self._page_path(symbol)

        # Build analysis entry
        opinions_text = []
        for o in signal.opinions:
            opinions_text.append(
                f"  - {o.agent_name}: {o.signal.value} ({o.confidence:.0%}) — {o.reasoning[:100]}"
            )

        consensus = f"{signal.side.value.upper()} (conf: {signal.confidence:.0%}, strength: {signal.strength.value})"
        content = (
            f"- **Analysis**: {consensus}\n"
            f"  - Timeframe: {signal.timeframe}\n"
            f"  - Agents agree: {'Yes' if signal.agents_agree else 'No'}\n"
        )
        if opinions_text:
            content += "  - Opinions:\n" + "\n".join(opinions_text) + "\n"
        if signal.entry_price:
            content += f"  - Entry zone: ${signal.entry_price:,.4f}\n"

        self._write_or_append_page(page, "analysis", content)
