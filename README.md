# AutoGMDH

Advanced Two-Stage GMDH + Neural Architecture Search (NAS) with weighted hybrid
and stacked aggregators for tabular regression.

This package wraps a self-organizing hybrid model that combines:
- Polynomial GMDH-style neurons
- Per-pair Neural Architecture Search
- Weighted poly/NN hybrids
- A stacked final aggregator (NN + ElasticNet + Gradient Boosting)

## Installation

```bash
pip install autogmdh
