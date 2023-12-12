import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data


class InitialEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(InitialEmbedding, self).__init__()
        self.embedding_mlp = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, x):
        x = self.embedding_mlp(x)
        x = x * torch.sigmoid(x)  # Swish activation
        x = nn.LayerNorm(x.size()[1:])(x)  # Layer normalization
        return x


class GNNLayer(MessagePassing):
    def __init__(self, node_dim):
        super(GNNLayer, self).__init__(aggr='mean')  # 'mean' aggregation.
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, 2 * node_dim),
            nn.ReLU(),
            nn.Linear(2 * node_dim, node_dim)
        )

    def forward(self, x, edge_index):
        x = self.mlp(x)
        x = x * torch.sigmoid(x)  # Swish activation
        x = nn.LayerNorm(x.size()[1:])(x)  # Layer normalization
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        # x_j denotes the features of source nodes
        return x_j

    def update(self, aggr_out):
        # aggr_out denotes the aggregated features from neighbors
        return aggr_out


class MeshGNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, node_dim, num_layers):
        super(MeshGNN, self).__init__()
        self.initial_embedding = InitialEmbedding(input_dim, embedding_dim)
        self.layers = nn.ModuleList([GNNLayer(node_dim) for _ in range(num_layers)])

    def forward(self, x, edge_index):
        x = self.initial_embedding(x)
        for layer in self.layers:
            x = layer(x, edge_index) + x  # Residual connection
        return x


def normalize_features(features):
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    return (features - mean) / std


def test_mesh_gnn():
    # Example data - replace with your actual data
    num_nodes = 10000  # Number of nodes
    num_node_features = 20  # Number of features per node
    num_edges = 20000  # Number of edges

    # Randomly generated data for demonstration
    nodes = torch.rand(num_nodes, num_node_features)
    nodes = normalize_features(nodes)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # todo:
    #  in our real case scenario, nodes would be every skeleton voxel and its features, the edges would be every
    #  adjacent voxel, but then also connections to every 2, 4, 8, etc neighbors away (the multi-mesh), and the
    #  edge_index would be a list of all these edges

    # Create the model
    model = MeshGNN(input_dim=num_node_features, embedding_dim=512, node_dim=512, num_layers=3)

    # Forward pass
    node_embeddings = model(nodes, edge_index)

    return node_embeddings


if __name__ == '__main__':
    test_output_embeddings = test_mesh_gnn()
    print(test_output_embeddings.shape)
