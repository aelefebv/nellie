import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

from src.feature_extraction.graph_frame import GraphBuilder
from src.im_info.im_info import ImInfo

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


class InitialEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(InitialEmbedding, self).__init__()
        self.embedding_mlp = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.SiLU(),
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.embedding_mlp(x)
        x = self.layer_norm(x)  # Layer normalization
        return x


class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super(GATLayer, self).__init__()
        self.gat_conv = GATv2Conv(in_channels, out_channels // heads, heads=heads, dropout=0.2)
        self.layer_norm = nn.LayerNorm(out_channels)
        # todo need to figure out this attention head issue when non-divisible

    def forward(self, x, edge_index):
        x = self.gat_conv(x, edge_index)
        x = F.silu(x)  # Using SiLU activation function
        x = self.layer_norm(x)
        return x


class GNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GNNLayer, self).__init__(aggr='mean')  # 'mean' aggregation.
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.SiLU(),
        )
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index):
        x = self.mlp(x)
        x = self.layer_norm(x)  # Layer normalization
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class GNNEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers):
        super(GNNEncoder, self).__init__()
        self.initial_embedding = InitialEmbedding(input_dim, hidden_dim)
        self.layers = nn.ModuleList()

        # Intermediate layers
        for _ in range(num_layers - 1):
            self.layers.append(GNNLayer(hidden_dim, hidden_dim))
            # self.layers.append(GATLayer(hidden_dim, hidden_dim))

    def forward(self, x, edge_index):
        x = self.initial_embedding(x)
        for layer in self.layers:
            x = layer(x, edge_index) + x  # Residual connection
        return x


class GNNDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers):
        super(GNNDecoder, self).__init__()
        self.layers = nn.ModuleList()

        # Intermediate layers
        for _ in range(num_layers - 1):
            # self.layers.append(GNNLayer(hidden_dim, hidden_dim))
            self.layers.append(GATLayer(hidden_dim, hidden_dim))

        # Final layer to output original feature size
        # self.layers.append(GNNLayer(hidden_dim, output_dim))
        self.layers.append(GATLayer(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index) + x  # Residual connection
        x = self.layers[-1](x, edge_index)
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

def import_data(im_path):
    # im_path = r"D:\test_files\nelly_tests\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif"
    im_info = ImInfo(im_path)
    graph_builder = GraphBuilder(im_info)
    graph_builder.run()
    dataset = Data(x=graph_builder.node_features, edge_index=graph_builder.edges)
    dataset_normalized = dataset.clone()
    dataset_normalized.x = normalize_features(dataset.x)
    return dataset, dataset_normalized

def get_final_reconstruction(dataset, model):
    dataset = dataset.to(device)
    with torch.no_grad():
        reconstructed = model(dataset.x, dataset.edge_index).cpu().numpy()
    return reconstructed

def train_model():
    dataset_0, dataset_0_norm = import_data(r"D:\test_files\nelly_tests\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif")
    datasets = [dataset_0_norm,]
    dataloader = DataLoader(datasets, batch_size=32, shuffle=True)

    num_node_features = 25

    # Initialize the autoencoder
    embedding_dim = 512  # Dimensionality of the node embeddings
    hidden_dim = 512  # Hidden dimension size
    num_layers = 16  # Number of GNN layers
    autoencoder = GNNAutoencoder(num_node_features, embedding_dim, hidden_dim, num_layers).to(device)

    num_epochs = 5000
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)
    loss_values = []
    for epoch in range(num_epochs):
        total_loss = 0
        for n, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            reconstructed = autoencoder(data.x, data.edge_index)
            loss = F.mse_loss(reconstructed, data.x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        loss_values.append(avg_loss)
        print(f'Epoch {epoch + 1}, Avg Loss: {avg_loss}')

    plt.plot(loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()


    def get_embeddings(dataset):
        dataset = dataset.to(device)
        with torch.no_grad():
            embeddings = autoencoder.encoder(dataset.x, dataset.edge_index).cpu().numpy()
        return embeddings

    reconstructed_final = get_final_reconstruction(datasets[0])
    # transform back to unnormalized
    reconstructed_final = reconstructed_final * dataset_0.x.std(dim=0, keepdim=True).cpu().numpy() + dataset_0.x.mean(dim=0, keepdim=True).cpu().numpy()
    original = dataset_0.x.cpu().numpy()

    # Get embeddings for each dataset
    embeddings = get_embeddings(datasets[0])[::20]
    np.save(r"D:\test_files\nelly_tests\embeddings.npy", embeddings)

    # save the model
    torch.save(autoencoder.state_dict(), r"D:\test_files\nelly_tests\autoencoder.pt")
    # load the model
    new_autoencoder = GNNAutoencoder(num_node_features, embedding_dim, hidden_dim, num_layers).to(device)
    new_autoencoder.load_state_dict(torch.load(r"D:\test_files\nelly_tests\autoencoder.pt"))
    test_reconstructed = get_final_reconstruction(datasets[0], new_autoencoder)
    test_reconstructed = test_reconstructed * dataset_0.x.std(dim=0, keepdim=True).cpu().numpy() + dataset_0.x.mean(dim=0, keepdim=True).cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])

    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.title('t-SNE Visualization of All Dataset Embeddings')
    plt.legend()
    plt.show()
    print('hi')

def test_and_train():
    def create_dataset(num_nodes, num_features, feature_range=(0, 1)):
        nodes = np.random.uniform(low=feature_range[0], high=feature_range[1], size=(num_nodes, num_features))
        edge_index = np.random.randint(num_nodes, size=(2, num_edges))
        return Data(x=torch.tensor(nodes, dtype=torch.float), edge_index=torch.tensor(edge_index, dtype=torch.long))

    # Parameters
    num_nodes = 100
    num_node_features = 16
    num_edges = 300

    # Create datasets
    similar_dataset1 = create_dataset(num_nodes, num_node_features, feature_range=(0, 0.5))
    similar_dataset2 = create_dataset(num_nodes, num_node_features, feature_range=(0, 0.5))
    # replace the first 90 items with the same values
    similar_dataset2.x[:90] = similar_dataset1.x[:90]
    similar_dataset2.edge_index[:290] = similar_dataset1.edge_index[:290]
    #randomly remove 20 edges

    different_dataset = create_dataset(num_nodes, num_node_features, feature_range=(20, 30))
    # different_dataset2 = create_dataset(num_nodes, num_node_features, feature_range=(20, 30))
    # different_dataset2.x[:90] = different_dataset.x[:90]
    # different_dataset2.edge_index[:290] = different_dataset.edge_index[:290]

    # Normalize features
    similar_dataset1.x = normalize_features(similar_dataset1.x)
    similar_dataset2.x = normalize_features(similar_dataset2.x)
    different_dataset.x = normalize_features(different_dataset.x)

    # Combine the similar datasets for training
    train_datasets = [similar_dataset1, different_dataset]

    dataloader = DataLoader(train_datasets, batch_size=2, shuffle=True)

    # Initialize the autoencoder
    embedding_dim = 512  # Dimensionality of the node embeddings
    hidden_dim = 512  # Hidden dimension size
    num_layers = 16  # Number of GNN layers
    autoencoder = GNNAutoencoder(num_node_features, embedding_dim, hidden_dim, num_layers).to(device)

    num_epochs = 5000
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)
    loss_values = []
    for epoch in range(num_epochs):
        total_loss = 0
        for n, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            reconstructed = autoencoder(data.x, data.edge_index)
            loss = F.mse_loss(reconstructed, data.x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        loss_values.append(avg_loss)
        print(f'Epoch {epoch + 1}, Avg Loss: {avg_loss}')

    plt.plot(loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()

    def get_embeddings(dataset):
        dataset = dataset.to(device)
        with torch.no_grad():
            embeddings = autoencoder.encoder(dataset.x, dataset.edge_index).cpu().numpy()
        return embeddings

    # Get embeddings for each dataset
    embeddings_similar1 = get_embeddings(similar_dataset1)
    embeddings_similar2 = get_embeddings(similar_dataset2)
    embeddings_different = get_embeddings(different_dataset)

    all_embeddings = np.vstack([embeddings_similar1, embeddings_similar2, embeddings_different])
    labels = np.array(['Similar1']*len(embeddings_similar1) + ['Similar2']*len(embeddings_similar2) + ['Different']*len(embeddings_different))

    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(all_embeddings)

    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        indices = labels == label
        plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=label)

    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.title('t-SNE Visualization of All Dataset Embeddings')
    plt.legend()
    plt.show()

    from torch.nn.functional import cosine_similarity

    # Convert embeddings to PyTorch tensors
    embeddings_similar1 = torch.tensor(embeddings_similar1)
    embeddings_similar2 = torch.tensor(embeddings_similar2)
    embeddings_different = torch.tensor(embeddings_different)

    # # Calculate the mean embedding for each dataset
    # mean_embedding_similar1 = embeddings_similar1.mean(dim=0)
    # mean_embedding_similar2 = embeddings_similar2.mean(dim=0)
    # mean_embedding_different = embeddings_different.mean(dim=0)

    similarity_sim1_sim2 = cosine_similarity(embeddings_similar1, embeddings_similar2, dim=1).mean().item()
    similarity_sim1_diff = cosine_similarity(embeddings_similar1, embeddings_different, dim=1).mean().item()
    similarity_sim2_diff = cosine_similarity(embeddings_similar2, embeddings_different, dim=1).mean().item()

    # Compare the mean embeddings
    # similarity_sim1_sim2 = cosine_similarity(mean_embedding_similar1.unsqueeze(0),
    #                                          mean_embedding_similar2.unsqueeze(0)).item()
    # similarity_sim1_diff = cosine_similarity(mean_embedding_similar1.unsqueeze(0),
    #                                          mean_embedding_different.unsqueeze(0)).item()

    print(f"Mean Cosine Similarity between Similar1 and Similar2: {similarity_sim1_sim2}")
    print(f"Mean Cosine Similarity between Similar1 and Different: {similarity_sim1_diff}")
    print(f"Mean Cosine Similarity between Similar2 and Different: {similarity_sim2_diff}")

def run_model(model_path, dataset):
    model = GNNAutoencoder(25, 512, 512, 16).to(device)
    model.load_state_dict(torch.load(model_path))
    reconstruction = get_final_reconstruction(dataset, model)
    return reconstruction



if __name__ == '__main__':
    model_path = r"D:\test_files\nelly_tests\autoencoder.pt"
    dataset_0, dataset_0_norm = import_data(r"D:\test_files\nelly_tests\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif")
    datasets = [dataset_0_norm, ]
    reconstruction = run_model(model_path, datasets[0])
    # transform back to unnormalized
    reconstruction = reconstruction * dataset_0.x.std(dim=0, keepdim=True).cpu().numpy() + dataset_0.x.mean(dim=0, keepdim=True).cpu().numpy()

    pass
