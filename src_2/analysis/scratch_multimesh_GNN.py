import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch.utils.data import DataLoader


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
    def __init__(self, in_channels, out_channels):
        super(GNNLayer, self).__init__(aggr='mean')  # 'mean' aggregation.
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index):
        x = self.mlp(x)
        x = x * torch.sigmoid(x)  # Swish activation
        x = nn.LayerNorm(x.size()[1:])(x)  # Layer normalization
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class GNNEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers):
        super(GNNEncoder, self).__init__()
        self.initial_embedding = InitialEmbedding(input_dim, hidden_dim)

        # Setup GNN layers
        self.layers = nn.ModuleList()
        current_dim = hidden_dim

        # Intermediate layers
        for _ in range(num_layers - 1):
            self.layers.append(GNNLayer(current_dim, hidden_dim))

        # Final layer to produce embeddings
        self.layers.append(GNNLayer(current_dim, embedding_dim))

    def forward(self, x, edge_index):
        x = self.initial_embedding(x)
        for layer in self.layers:
            x = layer(x, edge_index) + x  # Residual connection
        return x


class GNNDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers):
        super(GNNDecoder, self).__init__()
        self.layers = nn.ModuleList()

        # Start with the embedding dimension
        current_dim = embedding_dim

        # Intermediate layers
        for _ in range(num_layers - 1):
            self.layers.append(GNNLayer(current_dim, hidden_dim))
            current_dim = hidden_dim

        # Final layer to output original feature size
        self.layers.append(GNNLayer(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index) + x  # Residual connection
        return x


class GNNAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers):
        super(GNNAutoencoder, self).__init__()
        self.encoder = GNNEncoder(input_dim, embedding_dim, hidden_dim, num_layers)
        self.decoder = GNNDecoder(embedding_dim, hidden_dim, input_dim, num_layers)

    def forward(self, x, edge_index):
        embeddings = self.encoder(x, edge_index)
        reconstructed = self.decoder(embeddings, edge_index)
        return reconstructed


def normalize_features(features):
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    return (features - mean) / std

def test_mesh_gnn():
    # Randomly generated dataset
    num_nodes = 100  # Number of nodes in the graph
    num_node_features = 16  # Number of features per node
    num_edges = 300  # Number of edges in the graph

    # Random node features
    nodes = torch.rand((num_nodes, num_node_features))

    # Random edge indices (assuming undirected graph)
    edge_index = torch.randint(num_nodes, (2, num_edges))

    # Create a simple dataset and dataloader
    graph_data = Data(x=nodes, edge_index=edge_index)
    # dataloader = DataLoader([graph_data], batch_size=1, shuffle=True)

    # Initialize the autoencoder
    embedding_dim = 512  # Dimensionality of the node embeddings
    hidden_dim = 512  # Hidden dimension size
    num_layers = 3  # Number of GNN layers
    autoencoder = GNNAutoencoder(num_node_features, embedding_dim, hidden_dim, num_layers)

    num_epochs = 20
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        for data in [graph_data]:  # Assuming graph_data is a list of Data objects
            optimizer.zero_grad()
            reconstructed = autoencoder(data.x, data.edge_index)
            loss = F.mse_loss(reconstructed, data.x)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
    # for epoch in range(20):  # Number of training epochs
    #     for batch in dataloader:
    #         optimizer.zero_grad()
    #         reconstructed = autoencoder(batch.x, batch.edge_index)
    #         loss = F.mse_loss(reconstructed, batch.x)
    #         loss.backward()
    #         optimizer.step()
    #     print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

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
