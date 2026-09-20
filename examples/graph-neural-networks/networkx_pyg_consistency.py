import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree, from_networkx, to_undirected


NUM_NODES = 5
UNDIRECTED_EDGES = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 4)]
EXPECTED_DEGREES = [4, 3, 2, 1, 2]
EXPECTED_DENSITY = 0.6


graph = nx.Graph()
graph.add_nodes_from(range(NUM_NODES))
graph.add_edges_from(UNDIRECTED_EDGES)

networkx_degrees = [value for _, value in graph.degree()]
networkx_density = nx.density(graph)

edge_index = torch.tensor(UNDIRECTED_EDGES, dtype=torch.long).t().contiguous()
edge_index = to_undirected(edge_index, num_nodes=NUM_NODES)
features = torch.tensor(
    [[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]], dtype=torch.float
)
data = Data(x=features, edge_index=edge_index)
data.validate(raise_on_error=True)

pyg_degrees = degree(data.edge_index[0], num_nodes=data.num_nodes).to(torch.long)
pyg_density = data.num_edges / (data.num_nodes * (data.num_nodes - 1))

for node, feature in enumerate(features.tolist()):
    graph.nodes[node]["x"] = feature
converted = from_networkx(graph, group_node_attrs=["x"])
converted.validate(raise_on_error=True)

assert networkx_degrees == EXPECTED_DEGREES
assert pyg_degrees.tolist() == EXPECTED_DEGREES
assert abs(networkx_density - EXPECTED_DENSITY) < 1e-12
assert abs(pyg_density - EXPECTED_DENSITY) < 1e-12
assert converted.x.shape == (NUM_NODES, 2)
assert converted.edge_index.shape == (2, 2 * len(UNDIRECTED_EDGES))

print(f"NetworkX degrees: {networkx_degrees}")
print(f"PyG degrees: {pyg_degrees.tolist()}")
print(f"density: {networkx_density:.1f} / {pyg_density:.1f}")
print(f"converted: {converted}")
