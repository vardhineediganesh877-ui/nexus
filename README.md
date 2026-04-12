<p align="center">
  <img src="docs/logo.svg" alt="NEXUS" width="200" />
</p>

<h1 align="center">NEXUS — AI Crypto Trading Intelligence Platform</h1>

<p align="center">
  <strong>Multi-agent analysis × Multi-exchange execution × Knowledge accumulation</strong>
</p>

<p align="center">
  <a href="https://github.com/vardhineediganesh877-ui/nexus/stargazers"><img src="https://img.shields.io/github/stars/vardhineediganesh877-ui/nexus?style=social" alt="Stars"></a>
  <a href="https://github.com/vardhineediganesh877-ui/nexus/releases"><img src="https://img.shields.io/github/v/release/vardhineediganesh877-ui/nexus" alt="Release"></a>
  <a href="https://github.com/vardhineediganesh877-ui/nexus/actions"><img src="https://img.shields.io/github/actions/workflow/status/vardhineediganesh877-ui/nexus/test.yml" alt="CI"></a>
  <a href="https://pypi.org/project/nexus-trading/"><img src="https://img.shields.io/pypi/v/nexus-trading" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
</p>

---

## What is NEXUS?

NEXUS is an autonomous AI crypto trading platform that combines multi-agent market intelligence with multi-exchange execution and persistent knowledge accumulation. Unlike traditional bots that rely on single-strategy indicators, NEXUS uses a **council of AI agents** — Technical Analyst, Sentiment Analyst, Risk Manager, and Fundamental Analyst — that debate every trade decision before a dollar is risked.

Think of it as having a hedge fund research team that never sleeps, backed by institutional-grade tools (30+ TradingView indicators, Reddit sentiment, financial news), executing across 100+ exchanges via CCXT, and learning from every trade to get smarter over time.

## ⚡ What Makes NEXUS Different

| Feature | NEXUS | Freqtrade | OctoBot | Others |
|---------|-------|-----------|---------|--------|
| **Multi-Agent Analysis** | ✅ 4 agents debate | ❌ Single strategy | ❌ Single strategy | ❌ |
| **Sentiment Analysis** | ✅ Reddit + News | ❌ | ❌ | Rare |
| **Knowledge Accumulation** | ✅ Learns from trades | ❌ | ❌ | ❌ |
| **Walk-Forward Backtest** | ✅ Anti-overfitting | ✅ Basic | ✅ Basic | Rare |
| **Multi-Exchange** | ✅ 100+ via CCXT | ✅ ~20 | ✅ ~15 | Varies |
| **Zero-Config Start** | ✅ Paper mode default | ⚠️ Needs config | ⚠️ Needs config | ⚠️ |
| **MCP Integration** | ✅ TradingView MCP | ❌ | ❌ | ❌ |
| **Telegram Dashboard** | ✅ Real-time | ✅ Basic | ✅ Basic | Varies |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    NEXUS CORE                        │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐ │
│  │Technical │  │Sentiment │  │Fundamental│ │ Risk │ │
│  │ Analyst  │  │ Analyst  │  │ Analyst  │  │Mgr   │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──┬───┘ │
│       │             │             │            │      │
│       └─────────────┴─────────────┴────────────┘      │
│                         │                             │
│                  ┌──────▼──────┐                      │
│                  │   Signal    │                      │
│                  │   Engine    │                      │
│                  └──────┬──────┘                      │
│                         │                             │
│       ┌─────────────────┼─────────────────┐          │
│       │                 │                 │          │
│  ┌────▼─────┐    ┌─────▼──────┐   ┌──────▼─────┐   │
│  │ CCXT     │    │ Knowledge  │   │ Dashboard  │   │
│  │ Executor │    │ Engine     │   │ (Telegram) │   │
│  └──────────┘    │ (GBrain)   │   └────────────┘   │
│                  └────────────┘                      │
└─────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

```bash
git clone https://github.com/vardhineediganesh877-ui/nexus.git
cd nexus
pip install -e .

# Launch with paper trading (zero risk)
nexus start --paper

# Analyze a coin
nexus analyze BTC/USDT

# Run full autonomous scan
nexus scan --exchange binance
```

## 📱 Telegram Bot Commands

```
/analyze <symbol>     — Multi-agent analysis
/scan                 — Find opportunities across exchanges
/portfolio            — Current positions and P&L
/signal <symbol>      — Get trade signal with confidence
/backtest <strategy>  — Backtest with walk-forward validation
/status               — System health and performance
/paper on|off         — Toggle paper trading
```

## 🔬 Analysis Engine

Each analysis runs through 4 specialized agents:

### 1. Technical Analyst
- 30+ TradingView indicators (RSI, MACD, Bollinger, EMA, Supertrend, etc.)
- Multi-timeframe alignment (1m → Weekly)
- Volume confirmation analysis
- Candle pattern detection

### 2. Sentiment Analyst
- Reddit sentiment (r/cryptocurrency, r/bitcoin, etc.)
- Financial news aggregation (Reuters, CoinDesk)
- Fear & Greed index tracking
- Social signal detection

### 3. Fundamental Analyst
- On-chain metrics (whale movements, exchange flows)
- Market cap analysis
- Tokenomics evaluation
- Sector rotation detection

### 4. Risk Manager
- Position sizing (Kelly criterion)
- Correlation analysis
- Drawdown limits
- Maximum exposure controls

## 📊 Backtesting

```bash
# Walk-forward backtest (anti-overfitting)
nexus backtest BTC/USDT --strategy rsi --period 1y --walk-forward

# Compare all strategies
nexus backtest BTC/USDT --compare --period 2y
```

Strategies: RSI, Bollinger Bands, MACD, EMA Cross, Supertrend, Donchian Channel

## 🔒 Security

- Paper trading by default — no API keys needed to start
- API keys stored in `.env` (never in code or config files)
- All exchange communication over HTTPS
- No external data sharing
- Rate limiting built-in

## 🛠️ Tech Stack

- **Python 3.10+** — Core engine
- **FastAPI** — REST API + WebSocket
- **CCXT** — Multi-exchange trading
- **TradingView TA** — Technical analysis (30+ indicators)
- **SQLite** — Trade history and portfolio
- **PostgreSQL + pgvector** — Knowledge graph (optional)
- **Matplotlib** — Chart generation
- **Telegram Bot API** — Dashboard

## 📄 License

MIT — Use it, modify it, deploy it. Just don't blame us for trades.

---

> ⚠️ **Disclaimer**: Trading involves risk. NEXUS is for educational purposes. Always start with paper trading. Past performance doesn't guarantee future results.
