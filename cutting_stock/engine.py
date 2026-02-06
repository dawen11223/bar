"""Industrial-grade 1D cutting stock optimizer for rebar processing."""

from __future__ import annotations

import dataclasses
import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple


@dataclasses.dataclass(frozen=True)
class Stock:
    stock_id: str
    length: int
    cost: float
    quantity: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class Demand:
    item_id: str
    length: int
    quantity: int


@dataclasses.dataclass(frozen=True)
class Pattern:
    stock_id: str
    counts: Dict[str, int]
    waste: int
    cost: float


@dataclasses.dataclass
class OptimizationConfig:
    kerf: int = 0
    endtrim: int = 0
    max_iterations: int = 200
    min_reduced_cost: float = -1e-6
    stabilize_weight: float = 0.7
    random_seed: int = 7
    use_remnants_first: bool = True


@dataclasses.dataclass
class OptimizationResult:
    patterns: List[Pattern]
    pattern_counts: Dict[int, int]
    unmet_demand: Dict[str, int]
    total_cost: float
    total_waste: int


class CuttingStockOptimizer:
    """Adaptive cutting stock optimizer with column generation and heuristics."""

    def __init__(
        self,
        stocks: Sequence[Stock],
        demands: Sequence[Demand],
        config: Optional[OptimizationConfig] = None,
    ) -> None:
        self.stocks = list(stocks)
        self.demands = list(demands)
        self.config = config or OptimizationConfig()
        self._rng = random.Random(self.config.random_seed)

    def solve(self) -> OptimizationResult:
        """Solve the cutting stock problem end-to-end."""
        if not self.demands or not self.stocks:
            return OptimizationResult([], {}, {}, 0.0, 0)

        strategy = self._select_strategy()
        if strategy == "EXACT_ILP":
            result = self._solve_exact()
            if result is not None:
                return result

        return self._solve_column_generation()

    def _select_strategy(self) -> str:
        unique_items = len(self.demands)
        total_qty = sum(d.quantity for d in self.demands)
        if unique_items <= 8 and total_qty <= 60:
            return "EXACT_ILP"
        return "COLUMN_GENERATION"

    def _solve_exact(self) -> Optional[OptimizationResult]:
        """Attempt an exact ILP solve using available solvers."""
        import importlib.util

        if importlib.util.find_spec("pulp") is None:
            return None
        import pulp  # type: ignore

        patterns = self._seed_patterns()
        problem = pulp.LpProblem("cutting_stock", pulp.LpMinimize)
        x_vars = [pulp.LpVariable(f"x_{i}", lowBound=0, cat="Integer") for i in range(len(patterns))]
        problem += pulp.lpSum(x_vars[i] * patterns[i].cost for i in range(len(patterns)))
        for demand in self.demands:
            problem += (
                pulp.lpSum(x_vars[i] * patterns[i].counts.get(demand.item_id, 0) for i in range(len(patterns)))
                >= demand.quantity
            )
        status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
        if status != pulp.LpStatusOptimal:
            return None
        pattern_counts = {i: int(round(x_vars[i].value())) for i in range(len(patterns)) if x_vars[i].value()}
        return self._build_result(patterns, pattern_counts)

    def _solve_column_generation(self) -> OptimizationResult:
        patterns = self._seed_patterns()
        duals = {d.item_id: 0.0 for d in self.demands}
        stabilized_duals = duals.copy()

        for _ in range(self.config.max_iterations):
            lp_solution = _solve_lp_master(patterns, self.demands)
            if lp_solution is None:
                break
            primal, duals = lp_solution
            stabilized_duals = {
                k: self.config.stabilize_weight * stabilized_duals[k]
                + (1 - self.config.stabilize_weight) * duals[k]
                for k in duals
            }
            new_patterns = self._price_patterns(stabilized_duals)
            if not new_patterns:
                pattern_counts = self._round_down(primal, patterns)
                return self._build_result(patterns, pattern_counts)
            patterns.extend(new_patterns)

        pattern_counts = self._heuristic_pack(patterns)
        return self._build_result(patterns, pattern_counts)

    def _seed_patterns(self) -> List[Pattern]:
        patterns: List[Pattern] = []
        for stock in self.stocks:
            usable = stock.length - self.config.endtrim
            if usable <= 0:
                continue
            for demand in self.demands:
                piece_len = demand.length + self.config.kerf
                if piece_len <= 0:
                    continue
                max_count = usable // piece_len
                if max_count > 0:
                    counts = {demand.item_id: max_count}
                    waste = usable - max_count * piece_len
                    patterns.append(Pattern(stock.stock_id, counts, waste, stock.cost))
        return patterns

    def _price_patterns(self, duals: Dict[str, float]) -> List[Pattern]:
        new_patterns: List[Pattern] = []
        for stock in self.stocks:
            usable = stock.length - self.config.endtrim
            if usable <= 0:
                continue
            pattern = _solve_knapsack(self.demands, duals, usable, self.config.kerf)
            if not pattern:
                continue
            reduced_cost = stock.cost - sum(duals[k] * v for k, v in pattern.items())
            if reduced_cost < self.config.min_reduced_cost:
                total_len = sum(
                    (self._demand_by_id(item_id).length + self.config.kerf) * count
                    for item_id, count in pattern.items()
                )
                waste = usable - total_len
                new_patterns.append(Pattern(stock.stock_id, pattern, waste, stock.cost))
        return new_patterns

    def _round_down(self, primal: Dict[int, float], patterns: List[Pattern]) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for idx, value in primal.items():
            if value > 0:
                counts[idx] = int(math.floor(value))
        return self._fill_remaining(counts, patterns)

    def _fill_remaining(self, counts: Dict[int, int], patterns: List[Pattern]) -> Dict[int, int]:
        remaining = {d.item_id: d.quantity for d in self.demands}
        for idx, qty in counts.items():
            for item_id, count in patterns[idx].counts.items():
                remaining[item_id] = max(0, remaining[item_id] - count * qty)

        scored = sorted(
            range(len(patterns)),
            key=lambda i: _pattern_efficiency(patterns[i], remaining),
            reverse=True,
        )
        for idx in scored:
            if all(v == 0 for v in remaining.values()):
                break
            pattern = patterns[idx]
            max_add = math.inf
            for item_id, count in pattern.counts.items():
                if count > 0:
                    max_add = min(max_add, remaining[item_id] // count)
            if max_add == math.inf or max_add == 0:
                continue
            counts[idx] = counts.get(idx, 0) + max_add
            for item_id, count in pattern.counts.items():
                remaining[item_id] = max(0, remaining[item_id] - count * max_add)
        return counts

    def _heuristic_pack(self, patterns: List[Pattern]) -> Dict[int, int]:
        remaining = {d.item_id: d.quantity for d in self.demands}
        counts: Dict[int, int] = {}
        scored = sorted(
            range(len(patterns)),
            key=lambda i: _pattern_efficiency(patterns[i], remaining),
            reverse=True,
        )
        for idx in scored:
            pattern = patterns[idx]
            max_add = math.inf
            for item_id, count in pattern.counts.items():
                if count > 0:
                    max_add = min(max_add, remaining[item_id] // count)
            if max_add == math.inf or max_add == 0:
                continue
            counts[idx] = max_add
            for item_id, count in pattern.counts.items():
                remaining[item_id] = max(0, remaining[item_id] - count * max_add)
        return counts

    def _build_result(self, patterns: List[Pattern], counts: Dict[int, int]) -> OptimizationResult:
        remaining = {d.item_id: d.quantity for d in self.demands}
        total_cost = 0.0
        total_waste = 0
        for idx, qty in counts.items():
            pattern = patterns[idx]
            total_cost += qty * pattern.cost
            total_waste += qty * pattern.waste
            for item_id, count in pattern.counts.items():
                remaining[item_id] = max(0, remaining[item_id] - count * qty)

        return OptimizationResult(patterns, counts, remaining, total_cost, total_waste)

    def _demand_by_id(self, item_id: str) -> Demand:
        for demand in self.demands:
            if demand.item_id == item_id:
                return demand
        raise KeyError(item_id)


def _pattern_efficiency(pattern: Pattern, remaining: Dict[str, int]) -> float:
    produced = sum(min(remaining.get(item_id, 0), count) for item_id, count in pattern.counts.items())
    if produced == 0:
        return 0.0
    return produced / max(1, pattern.waste + 1)


def _solve_knapsack(
    demands: Sequence[Demand],
    duals: Dict[str, float],
    capacity: int,
    kerf: int,
) -> Dict[str, int]:
    """Solve the pricing subproblem with a bounded knapsack DP."""
    items = [d for d in demands if d.length + kerf <= capacity]
    if not items:
        return {}

    dp = [0.0] * (capacity + 1)
    keep: List[Optional[int]] = [None] * (capacity + 1)

    for idx, demand in enumerate(items):
        weight = demand.length + kerf
        value = duals.get(demand.item_id, 0.0)
        for c in range(weight, capacity + 1):
            candidate = dp[c - weight] + value
            if candidate > dp[c]:
                dp[c] = candidate
                keep[c] = idx

    counts: Dict[str, int] = defaultdict(int)
    c = max(range(capacity + 1), key=lambda x: dp[x])
    while c > 0 and keep[c] is not None:
        idx = keep[c]
        demand = items[idx]
        weight = demand.length + kerf
        counts[demand.item_id] += 1
        c -= weight

    return dict(counts)


def _solve_lp_master(
    patterns: Sequence[Pattern],
    demands: Sequence[Demand],
) -> Optional[Tuple[Dict[int, float], Dict[str, float]]]:
    import importlib.util

    if importlib.util.find_spec("scipy") is None:
        return None
    from scipy.optimize import linprog  # type: ignore

    cost = [p.cost for p in patterns]
    A = []
    b = []
    for demand in demands:
        row = [p.counts.get(demand.item_id, 0) for p in patterns]
        A.append([-value for value in row])
        b.append(-demand.quantity)

    result = linprog(cost, A_ub=A, b_ub=b, bounds=[(0, None)] * len(patterns), method="highs")
    if not result.success:
        return None

    primal = {i: value for i, value in enumerate(result.x) if value > 1e-9}
    duals = {d.item_id: -result.ineqlin.marginals[i] for i, d in enumerate(demands)}
    return primal, duals
