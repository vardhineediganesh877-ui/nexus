"""Tests for StrategyEvolver — population generation, mutation, crossover, tournament selection."""
import copy
from unittest.mock import MagicMock, patch

import pytest

from src.analysis.evolve import (
    PARAM_SPACES,
    EvolvedStrategy,
    StrategyEvolver,
)


@pytest.fixture
def evolver():
    """Create a StrategyEvolver with a mocked BacktestEngine."""
    with patch("src.analysis.evolve.BacktestEngine"):
        return StrategyEvolver(exchange_id="mexc")


# ── EvolvedStrategy dataclass ────────────────────────────────────────

class TestEvolvedStrategy:
    def test_to_dict_roundtrip(self):
        es = EvolvedStrategy(
            strategy="rsi", params={"period": 14}, generation=1,
            fitness=0.85, sharpe=1.5, total_return=12.3,
            max_drawdown=8.1, win_rate=62.0, profit_factor=1.8, total_trades=10,
        )
        d = es.to_dict()
        assert d["strategy"] == "rsi"
        assert d["fitness"] == 0.85
        assert d["total_trades"] == 10


# ── _generate_population ─────────────────────────────────────────────

class TestGeneratePopulation:
    def test_correct_size(self, evolver):
        pop = evolver._generate_population(20)
        assert len(pop) == 20

    def test_every_entry_has_strategy(self, evolver):
        pop = evolver._generate_population(50)
        for indiv in pop:
            assert "_strategy" in indiv
            assert indiv["_strategy"] in PARAM_SPACES

    def test_params_within_bounds(self, evolver):
        pop = evolver._generate_population(100)
        for indiv in pop:
            strategy = indiv["_strategy"]
            space = PARAM_SPACES[strategy]
            for key, (lo, hi) in space.items():
                assert lo <= indiv[key] <= hi, f"{key}={indiv[key]} out of [{lo},{hi}]"


# ── _mutate ──────────────────────────────────────────────────────────

class TestMutate:
    def test_preserves_strategy(self, evolver):
        params = {"_strategy": "rsi", "period": 14, "oversold": 30, "overbought": 70}
        child = evolver._mutate(params)
        assert child["_strategy"] == "rsi"

    def test_does_not_mutate_original(self, evolver):
        params = {"_strategy": "rsi", "period": 14, "oversold": 30, "overbought": 70}
        original = copy.deepcopy(params)
        evolver._mutate(params)
        assert params == original

    def test_stays_within_bounds(self, evolver):
        for _ in range(50):
            params = {"_strategy": "macd", "fast": 12, "slow": 26, "signal": 9}
            child = evolver._mutate(params)
            space = PARAM_SPACES["macd"]
            for key, (lo, hi) in space.items():
                assert lo <= child[key] <= hi


# ── _crossover ───────────────────────────────────────────────────────

class TestCrossover:
    def test_same_strategy_blends(self, evolver):
        p1 = {"_strategy": "rsi", "period": 14, "oversold": 30, "overbought": 70}
        p2 = {"_strategy": "rsi", "period": 10, "oversold": 25, "overbought": 75}
        child = evolver._crossover(p1, p2)
        assert child["_strategy"] == "rsi"
        # Child should have params from both parents
        all_params = set()
        for _ in range(20):
            c = evolver._crossover(p1, p2)
            all_params.update(c.items())
        # Over 20 runs, uniform crossover should pull from both parents
        assert len(all_params) > 3  # not just identical to one parent

    def test_different_strategy_picks_one(self, evolver):
        p1 = {"_strategy": "rsi", "period": 14}
        p2 = {"_strategy": "macd", "fast": 12, "slow": 26, "signal": 9}
        child = evolver._crossover(p1, p2)
        assert child["_strategy"] in ("rsi", "macd")


# ── _select (tournament) ────────────────────────────────────────────

class TestTournamentSelect:
    def test_selects_best_from_sample(self, evolver):
        population = [
            EvolvedStrategy(strategy="a", params={}, generation=1, fitness=0.1),
            EvolvedStrategy(strategy="b", params={}, generation=1, fitness=0.9),
            EvolvedStrategy(strategy="c", params={}, generation=1, fitness=0.5),
        ]
        # With tournament_size=3 (all), should always pick fitness=0.9
        winner = evolver._select(population, 3)
        assert winner.fitness == 0.9

    def test_never_picks_worst_when_all_sampled(self, evolver):
        population = [
            EvolvedStrategy(strategy="a", params={}, generation=1, fitness=0.2),
            EvolvedStrategy(strategy="b", params={}, generation=1, fitness=0.8),
            EvolvedStrategy(strategy="c", params={}, generation=1, fitness=0.5),
        ]
        for _ in range(20):
            winner = evolver._select(population, 3)
            assert winner.fitness > 0.2


# ── _evaluate_fitness ────────────────────────────────────────────────

class TestEvaluateFitness:
    def test_returns_evolved_strategy_on_success(self, evolver):
        mock_result = MagicMock()
        mock_result.sharpe_ratio = 1.5
        mock_result.total_return_pct = 20.0
        mock_result.profit_factor = 2.0
        mock_result.win_rate = 65.0
        mock_result.max_drawdown_pct = 10.0
        mock_result.total_trades = 10

        with patch.object(evolver, "_backtest_with_params", return_value=mock_result):
            result = evolver._evaluate_fitness(
                {"_strategy": "rsi", "period": 14}, "BTC/USDT"
            )
        assert result is not None
        assert result.fitness > 0
        assert result.strategy == "rsi"
        assert result.total_trades == 10

    def test_returns_none_on_too_few_trades(self, evolver):
        mock_result = MagicMock()
        mock_result.total_trades = 1  # Below threshold of 2

        with patch.object(evolver, "_backtest_with_params", return_value=mock_result):
            result = evolver._evaluate_fitness(
                {"_strategy": "rsi", "period": 14}, "BTC/USDT"
            )
        assert result is None

    def test_returns_none_on_exception(self, evolver):
        with patch.object(evolver, "_backtest_with_params", side_effect=Exception("boom")):
            result = evolver._evaluate_fitness(
                {"_strategy": "rsi", "period": 14}, "BTC/USDT"
            )
        assert result is None


# ── evolve (integration) ─────────────────────────────────────────────

class TestEvolveIntegration:
    def test_evolve_returns_hall_of_fame(self, evolver):
        mock_result = MagicMock()
        mock_result.sharpe_ratio = 1.0
        mock_result.total_return_pct = 10.0
        mock_result.profit_factor = 1.5
        mock_result.win_rate = 55.0
        mock_result.max_drawdown_pct = 15.0
        mock_result.total_trades = 5

        with patch.object(evolver, "_backtest_with_params", return_value=mock_result):
            results = evolver.evolve("BTC/USDT", population_size=6, generations=2)

        assert len(results) > 0
        # Hall of fame should have entries from each generation
        assert all(isinstance(r, EvolvedStrategy) for r in results)
        # Sorted by fitness descending
        for i in range(len(results) - 1):
            assert results[i].fitness >= results[i + 1].fitness
