# Cutting Stock Optimizer (1D Rebar)

This package provides an adaptive, industrial-grade 1D cutting stock optimizer for rebar processing. It combines:

- **Exact ILP** for small instances (when `pulp` is available).
- **Column generation** with knapsack pricing for medium/large instances (using `scipy` if available).
- **Heuristic rounding** and pattern efficiency scoring to convert fractional solutions into integer plans.

## Installation

Optional dependencies improve solution quality:

```bash
pip install scipy pulp
```

## Usage

```python
from cutting_stock import CuttingStockOptimizer, Demand, OptimizationConfig, Stock

stocks = [
    Stock(stock_id="S1", length=12000, cost=100.0),
    Stock(stock_id="S2", length=9000, cost=80.0),
]

demands = [
    Demand(item_id="D1", length=3000, quantity=10),
    Demand(item_id="D2", length=2500, quantity=15),
    Demand(item_id="D3", length=1500, quantity=12),
]

config = OptimizationConfig(kerf=5, endtrim=10)
optimizer = CuttingStockOptimizer(stocks, demands, config)
result = optimizer.solve()

print("Total cost:", result.total_cost)
print("Total waste:", result.total_waste)
print("Unmet demand:", result.unmet_demand)
```

## Notes

- The optimizer accounts for kerf and end trim losses.
- When an LP solver is unavailable, it falls back to pattern heuristics.
- For production, ensure a reliable LP/ILP solver is installed for best results.
