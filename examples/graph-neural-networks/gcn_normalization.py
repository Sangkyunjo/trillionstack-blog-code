import numpy as np


def normalized_adjacency(adjacency):
    """Return D^(-1/2) (A + I) D^(-1/2)."""
    adjacency_with_loops = adjacency + np.eye(adjacency.shape[0])
    degree = adjacency_with_loops.sum(axis=1)
    degree_inverse_sqrt = np.diag(degree**-0.5)
    return degree_inverse_sqrt @ adjacency_with_loops @ degree_inverse_sqrt


if __name__ == "__main__":
    adjacency = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    features = np.array([[1.0], [2.0], [4.0]])
    adjacency_with_loops = adjacency + np.eye(adjacency.shape[0])
    degree = adjacency_with_loops.sum(axis=1)

    matrix_result = normalized_adjacency(adjacency) @ features
    node_result = np.zeros_like(features)
    for receiving_node in range(adjacency.shape[0]):
        for sending_node in range(adjacency.shape[0]):
            if adjacency_with_loops[receiving_node, sending_node] != 0:
                node_result[receiving_node] += features[sending_node] / np.sqrt(
                    degree[receiving_node] * degree[sending_node]
                )

    np.testing.assert_allclose(matrix_result, node_result)
    print(matrix_result)
