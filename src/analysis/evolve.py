"""
NEXUS Strategy Evolution Engine

Inspired by DeepMind's AlphaEvolve (via paperswithbacktest/pwb-alphaevolve):
- Generate initial population of strategy parameter combinations
- Backtest each using NEXUS's existing BacktestEngine
- Evaluate fitness (Sharpe, win rate, max DD, profit factor)
- Select best performers via tournament selection
- Mutate/crossover to create next generation
- Repeat for N generations, return ranked strategies

Key patterns stolen from AlphaEvolve:
- Elite selection ratio (exploit best) + exploration (random sampling)
- Multi-metric fitness scoring (not just return)
- Hall-of-fame tracking across generations
"""

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .backtest import BacktestEngine

logger = logging.getLogger(__name__)


# Parameter search spaces per strategy
PARAM_SPACES = {
    "rsi": {
        "period": (7, 21),
        "oversold": (20, 40),
        "overbought": (60, 80),
    },
    "bollinger": {
        "period": (10, 30),
        "std": (1.5, 3.0),
    },
    "macd": {
        "fast": (8, 15),
        "slow": (20, 35),
        "signal": (7, 12),
    },
    "ema_cross": {
        "fast": (5, 25),
        "slow": (30, 70),
    },
    "donchian": {
        "period": (10, 40),
    },
}


@dataclass
class EvolvedStrategy:
    """A strategy with its parameters and fitness metrics."""
    strategy: str
    params: Dict
    generation: int
    fitness: float = 0.0
    sharpe: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0

    def to_dict(self) -> Dict:
        return {
            "strategy": self.strategy,
            "params": self.params,
            "generation": self.generation,
            "fitness": round(self.fitness, 4),
            "sharpe": round(self.sharpe, 2),
            "total_return": round(self.total_return, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "win_rate": round(self.win_rate, 2),
            "profit_factor": round(self.profit_factor, 2),
            "total_trades": self.total_trades,
        }


class StrategyEvolver:
    """Evolves trading strategy parameters using genetic programming.

    Works with NEXUS's existing BacktestEngine. Evolves PARAMETERS of
    existing strategies (rsi, bollinger, macd, ema_cross, donchian),
    not new strategy types.
    """

    def __init__(self, exchange_id: str = "mexc"):
        self.engine = BacktestEngine(exchange_id)
        self.hall_of_fame: List[EvolvedStrategy] = []
        self._best_fitness_history: List[float] = []

    def evolve(
        self,
        symbol: str,
        population_size: int = 20,
        generations: int = 5,
        elite_ratio: float = 0.2,
        mutation_rate: float = 0.3,
        tournament_size: int = 3,
    ) -> List[EvolvedStrategy]:
        """Run evolution loop. Returns ranked strategies (best first)."""
        # 1. Generate initial population
        population = self._generate_population(population_size)
        logger.info(f"Generated initial population of {len(population)} strategies")

        for gen in range(1, generations + 1):
            # 2. Evaluate fitness
            fitness_results = []
            for indiv in population:
                result = self._evaluate_fitness(indiv, symbol)
                if result is not None:
                    fitness_results.append(result)

            if not fitness_results:
                logger.warning(f"Gen {gen}: No viable strategies found")
                continue

            # 3. Sort by fitness
            fitness_results.sort(key=lambda x: x.fitness, reverse=True)

            # Track best
            best = fitness_results[0]
            self._best_fitness_history.append(best.fitness)
            self.hall_of_fame.append(copy.deepcopy(best))

            logger.info(
                f"Gen {gen}/{generations}: Best fitness={best.fitness:.4f} "
                f"({best.strategy} sharpe={best.sharpe:.2f} ret={best.total_return:+.1f}%)"
            )

            if gen == generations:
                break

            # 4. Select parents (elitism + tournament)
            elite_count = max(2, int(elite_ratio * len(fitness_results)))
            elites = fitness_results[:elite_count]

            # 5. Create next generation
            next_pop = [{"_strategy": e.strategy, **e.params} for e in elites]  # Elites survive

            while len(next_pop) < population_size:
                r = random.random()
                if r < mutation_rate:
                    parent = self._select(fitness_results, tournament_size)
                    child = self._mutate({"_strategy": parent.strategy, **parent.params})
                    next_pop.append(child)
                else:
                    p1 = self._select(fitness_results, tournament_size)
                    p2 = self._select(fitness_results, tournament_size)
                    child = self._crossover(
                        {"_strategy": p1.strategy, **p1.params},
                        {"_strategy": p2.strategy, **p2.params},
                    )
                    next_pop.append(child)

            population = next_pop

        # Final ranking
        self.hall_of_fame.sort(key=lambda x: x.fitness, reverse=True)
        return self.hall_of_fame

    def _generate_population(self, size: int) -> List[Dict]:
        """Generate initial population with random parameter combinations."""
        population = []
        strategies = list(PARAM_SPACES.keys())

        for _ in range(size):
            strategy = random.choice(strategies)
            params = {"_strategy": strategy}

            for param_name, (lo, hi) in PARAM_SPACES[strategy].items():
                if isinstance(lo, int) and isinstance(hi, int):
                    params[param_name] = random.randint(lo, hi)
                else:
                    params[param_name] = round(random.uniform(lo, hi), 2)

            population.append(params)

        return population

    def _evaluate_fitness(self, params: Dict, symbol: str) -> Optional[EvolvedStrategy]:
        """Backtest a strategy with given params and compute fitness score."""
        strategy = params["_strategy"]
        try:
            # Use the existing backtest engine but with custom params
            # We need to monkey-patch the strategy method temporarily
            result = self._backtest_with_params(symbol, strategy, params)
            if result is None or result.total_trades < 2:
                return None

            # Multi-metric fitness (inspired by AlphaEvolve's multi-branch evaluation)
            # Weight: Sharpe (40%), Return (25%), Profit Factor (20%), Win Rate (15%)
            sharpe_score = max(0, result.sharpe_ratio) * 0.4
            return_score = max(0, result.total_return_pct) * 0.0025
            pf_score = min(result.profit_factor, 5) * 4 * 0.20
            wr_score = result.win_rate * 0.15 / 100

            # Penalty for high drawdown
            dd_penalty = max(0, result.max_drawdown_pct - 30) * 0.01

            fitness = sharpe_score + return_score + pf_score + wr_score - dd_penalty

            return EvolvedStrategy(
                strategy=strategy,
                params={k: v for k, v in params.items() if k != "_strategy"},
                generation=len(self._best_fitness_history) + 1,
                fitness=fitness,
                sharpe=result.sharpe_ratio,
                total_return=result.total_return_pct,
                max_drawdown=result.max_drawdown_pct,
                win_rate=result.win_rate,
                profit_factor=result.profit_factor,
                total_trades=result.total_trades,
            )
        except Exception as e:
            logger.debug(f"Eval failed for {strategy} {params}: {e}")
            return None

    def _backtest_with_params(self, symbol: str, strategy: str, params: Dict):
        """Run backtest with custom parameters by temporarily modifying strategy."""
        # Save original method
        original = getattr(self.engine, f"_strategy_{strategy}")

        # Create parameterized version
        def make_parametrized(orig_method, p):
            def wrapper(closes):
                # Filter out _strategy key, pass remaining as kwargs
                kwargs = {k: v for k, v in p.items() if k != "_strategy"}
                # Map param names to what the strategy methods expect
                # The existing methods accept limited params, so we inject them
                return orig_method(closes, **kwargs)
            return wrapper

        try:
            setattr(self.engine, f"_strategy_{strategy}", make_parametrized(original, params))
            return self.engine.backtest(symbol, strategy)
        finally:
            setattr(self.engine, f"_strategy_{strategy}", original)

    def _select(self, population: List[EvolvedStrategy], tournament_size: int) -> EvolvedStrategy:
        """Tournament selection — pick tournament_size random, return best."""
        contenders = random.sample(population, min(tournament_size, len(population)))
        return max(contenders, key=lambda x: x.fitness)

    def _mutate(self, params: Dict) -> Dict:
        """Random mutation of 1-2 strategy parameters."""
        strategy = params.get("_strategy", "")
        space = PARAM_SPACES.get(strategy, {})
        child = copy.deepcopy(params)

        # Mutate 1-2 params
        keys_to_mutate = random.sample(
            list(space.keys()), min(random.randint(1, 2), len(space))
        )
        for key in keys_to_mutate:
            if key not in space:
                continue
            lo, hi = space[key]
            if isinstance(lo, int):
                # Small perturbation (±20% of range)
                delta = max(1, int((hi - lo) * 0.2))
                child[key] = max(lo, min(hi, child[key] + random.randint(-delta, delta)))
            else:
                delta = (hi - lo) * 0.2
                child[key] = round(max(lo, min(hi, child[key] + random.uniform(-delta, delta))), 2)

        return child

    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Combine two parent strategies (same strategy type, blended params)."""
        # Prefer same-strategy crossover
        if parent1.get("_strategy") == parent2.get("_strategy"):
            child = copy.deepcopy(parent1)
            strategy = child["_strategy"]
            for key in PARAM_SPACES.get(strategy, {}):
                # Uniform crossover per param
                if random.random() < 0.5:
                    child[key] = parent2[key]
            return child
        else:
            # Different strategies — just pick one parent (no cross-species breeding)
            return copy.deepcopy(random.choice([parent1, parent2]))
