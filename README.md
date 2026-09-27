# TrillionStack Blog Code

Runnable examples that accompany the technical articles on [TrillionStack](https://trillionver2.tistory.com/).

This repository intentionally contains only reader-facing example code and small verification tests. Blog publishing automation, private data, OCR artifacts, credentials, and unrelated TrillionStack services are excluded.

## Graph Neural Networks

| Article | Example |
|---|---|
| [NetworkX and PyTorch Geometric](https://trillionver2.tistory.com/151) | [`networkx_pyg_consistency.py`](examples/graph-neural-networks/networkx_pyg_consistency.py) |
| [Graph Embedding](https://trillionver2.tistory.com/312) | [`anonymous_walk_example.py`](examples/graph-neural-networks/anonymous_walk_example.py) |
| [Core Concepts of GNNs](https://trillionver2.tistory.com/315) | [`permutation_equivariance.py`](examples/graph-neural-networks/permutation_equivariance.py) |
| [Graph Convolutional Networks](https://trillionver2.tistory.com/316) | [`gcn_normalization.py`](examples/graph-neural-networks/gcn_normalization.py) |

The English version of each article is available at the same URL with `?lang=en`.

## Reinforcement Learning: Cliff Walking

| Article | Runnable code |
|---|---|
| [① MC, SARSA, Q-learning](https://trillionver2.tistory.com/336) | [`cliff_walking.py`](examples/reinforcement-learning/cliff-walking/cliff_walking.py), [`algorithms.py`](examples/reinforcement-learning/cliff-walking/algorithms.py) |
| [② REINFORCE, Actor-Critic](https://trillionver2.tistory.com/492) | [`algorithms.py`](examples/reinforcement-learning/cliff-walking/algorithms.py) |
| [③ Off-policy MC, Actor-Critic](https://trillionver2.tistory.com/493) | [`algorithms.py`](examples/reinforcement-learning/cliff-walking/algorithms.py) |

The three posts share a 3×5 environment, seven teaching implementations, and one evaluation script. These are tabular demonstrations, not full reproductions of research algorithms. Quick run:

```bash
python examples/reinforcement-learning/cliff-walking/run_experiment.py --episodes 100 --seeds 1 --eval-episodes 20
```

To reproduce the article's result tables, use the defaults (5,000 episodes × 5 seeds per algorithm). That run takes longer.

## Time Series

| Article | Example |
|---|---|
| [Difference equations and impulse responses](https://trillionver2.tistory.com/137) | [`ar2_impulse_response.py`](examples/time-series/ar2_impulse_response.py) |
| [ARMA forecast error](https://trillionver2.tistory.com/154) | [`arma11_forecast_error.py`](examples/time-series/arma11_forecast_error.py) |

## Statistics

| Article | Example |
|---|---|
| [OLS linear regression](https://trillionver2.tistory.com/62) | [`ols_lstsq.py`](examples/statistics/ols_lstsq.py) |
| [Entropy, cross-entropy, and KL](https://trillionver2.tistory.com/64) | [`information_theory.py`](examples/statistics/information_theory.py) |

## Run the lightweight examples

Python 3.11 or newer is recommended. This command runs tests for the NumPy examples and the reinforcement learning code.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The NetworkX/PyG example has separate dependencies because PyTorch Geometric installation varies by platform:

```bash
python -m pip install -r requirements-pyg.txt
python examples/graph-neural-networks/networkx_pyg_consistency.py
```

## License

MIT. See [`LICENSE`](LICENSE).
