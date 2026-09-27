# TrillionStack 블로그 코드

[TrillionStack](https://trillionver2.tistory.com/) 기술 글에서 사용하는 실행 가능한 예제를 모은 저장소입니다.

독자가 직접 실행할 예제와 작은 검증 테스트만 공개합니다. 블로그 발행 자동화, 개인 데이터, OCR 산출물, 인증 정보와 블로그에 관계없는 TrillionStack 서비스는 포함하지 않습니다.

## Graph Neural Networks

| 포스팅 | 예제 |
|---|---|
| [NetworkX와 PyTorch Geometric](https://trillionver2.tistory.com/151) | [`networkx_pyg_consistency.py`](examples/graph-neural-networks/networkx_pyg_consistency.py) |
| [Graph Embedding](https://trillionver2.tistory.com/312) | [`anonymous_walk_example.py`](examples/graph-neural-networks/anonymous_walk_example.py) |
| [GNN의 핵심 개념](https://trillionver2.tistory.com/315) | [`permutation_equivariance.py`](examples/graph-neural-networks/permutation_equivariance.py) |
| [Graph Convolutional Networks](https://trillionver2.tistory.com/316) | [`gcn_normalization.py`](examples/graph-neural-networks/gcn_normalization.py) |

영문 내용은 각 글의 같은 주소에 `?lang=en`을 붙여 볼 수 있습니다.

## 강화학습: Cliff Walking

| 포스팅 | 실행 코드 |
|---|---|
| [① MC·SARSA·Q-learning](https://trillionver2.tistory.com/336) | [`cliff_walking.py`](examples/reinforcement-learning/cliff-walking/cliff_walking.py), [`algorithms.py`](examples/reinforcement-learning/cliff-walking/algorithms.py) |
| [② REINFORCE·Actor-Critic](https://trillionver2.tistory.com/492) | [`algorithms.py`](examples/reinforcement-learning/cliff-walking/algorithms.py) |
| [③ Off-policy MC·Actor-Critic](https://trillionver2.tistory.com/493) | [`algorithms.py`](examples/reinforcement-learning/cliff-walking/algorithms.py) |

세 글의 3×5 환경, 일곱 알고리즘과 평가 코드는 같은 폴더에 있습니다. 논문 전체 구현이 아니라 글에서 설명한 교육용 tabular 실험입니다. 빠른 실행:

```bash
python examples/reinforcement-learning/cliff-walking/run_experiment.py --episodes 100 --seeds 1 --eval-episodes 20
```

글의 결과표를 재현하려면 기본값(알고리즘별 5,000 episodes × 5 seeds)을 사용하세요. 실행 시간이 더 걸립니다.

## 시계열

| 포스팅 | 실행 코드 |
|---|---|
| [차분방정식과 충격반응](https://trillionver2.tistory.com/137) | [`ar2_impulse_response.py`](examples/time-series/ar2_impulse_response.py) |
| [ARMA 예측오차](https://trillionver2.tistory.com/154) | [`arma11_forecast_error.py`](examples/time-series/arma11_forecast_error.py) |

## 통계 기초

| 포스팅 | 실행 코드 |
|---|---|
| [OLS 선형 회귀](https://trillionver2.tistory.com/62) | [`ols_lstsq.py`](examples/statistics/ols_lstsq.py) |
| [Entropy·Cross-Entropy·KL](https://trillionver2.tistory.com/64) | [`information_theory.py`](examples/statistics/information_theory.py) |

## 가벼운 예제 실행

Python 3.11 이상을 권장합니다. 아래 명령은 NumPy 기반 예제와 강화학습 테스트를 실행합니다.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

PyTorch Geometric 설치 방법은 환경에 따라 달라질 수 있어 NetworkX/PyG 예제의 의존성을 분리했습니다.

```bash
python -m pip install -r requirements-pyg.txt
python examples/graph-neural-networks/networkx_pyg_consistency.py
```

## 라이선스

MIT 라이선스를 적용합니다. 자세한 내용은 [`LICENSE`](LICENSE)를 참고하세요.
