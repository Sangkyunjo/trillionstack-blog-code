import numpy as np


def aggregate(adjacency, features):
    return adjacency @ features


if __name__ == "__main__":
    adjacency = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    features = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
    permutation = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    original = aggregate(adjacency, features)
    relabelled = aggregate(
        permutation @ adjacency @ permutation.T,
        permutation @ features,
    )

    np.testing.assert_allclose(relabelled, permutation @ original)
    print(relabelled)
