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

The English article links will be added after their Tistory URLs have been created and verified.

## Run the lightweight examples

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
