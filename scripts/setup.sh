#!/bin/bash
# NEXUS One-Command Setup
# Run: bash scripts/setup.sh

set -e

echo "⚡ NEXUS Setup"
echo "================"

# Check Python version
python3 --version | grep -q "3.1[0-9]" || { echo "❌ Python 3.10+ required"; exit 1; }

# Install dependencies
echo "📦 Installing dependencies..."
pip install -e . --break-system-packages 2>&1 | tail -3

# Create data directory
mkdir -p ~/.nexus/data
echo "📁 Data directory: ~/.nexus/data"

# Test with a quick analysis
echo ""
echo "🧪 Running quick test..."
python3 -m src.cli analyze BTC/USDT --exchange mexc 2>&1 | head -5

echo ""
echo "✅ NEXUS is ready!"
echo ""
echo "Quick start:"
echo "  nexus analyze ETH/USDT     # Analyze a coin"
echo "  nexus scan                  # Find opportunities"
echo "  nexus portfolio             # Check positions"
echo "  nexus backtest BTC/USDT --compare  # Compare strategies"
echo "  nexus start --paper         # Autonomous mode"
echo ""
echo "API server:"
echo "  make run  # Start at http://localhost:8000"
