.PHONY: install test run analyze scan portfolio backtest docker clean

install:
	pip install -e . --break-system-packages

test:
	python3 -m pytest tests/ -v

run:
	python3 -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

analyze:
	python3 -m src.cli analyze BTC/USDT

scan:
	python3 -m src.cli scan --exchange mexc --top 5

portfolio:
	python3 -m src.cli portfolio

backtest:
	python3 -m src.cli backtest BTC/USDT --compare

start:
	python3 -m src.cli start --paper --exchange mexc

telegram:
	python3 -m src.cli telegram

docker:
	docker compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
	find . -type f -name "*.pyc" -delete 2>/dev/null
	rm -rf dist/ build/ *.egg-info .nexus/
