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

영문 포스팅 링크는 Tistory 영문 URL이 생성되고 검증된 뒤 추가합니다.

## 가벼운 예제 실행

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
