import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cosine_similarity
from torch_geometric.nn import MessagePassing, GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import datetime
import os

from src.feature_extraction.graph_frame import GraphBuilder
from src.im_info.im_info import ImInfo

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# YYYYMMDD_HHMMSS
current_dt = datetime.datetime.now()
current_dt_str = current_dt.strftime("%Y%m%d_%H%M%S")


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

def get_final_reconstruction(dataset, model):
    dataset = dataset.to(device)
    with torch.no_grad():
        reconstructed = model(dataset.x, dataset.edge_index).cpu().numpy()
    return reconstructed

def train_model(training, validation, savedir):
    num_node_features = training.dataset[0].num_features

    # Initialize the autoencoder
    embedding_dim = 512  # Dimensionality of the node embeddings
    hidden_dim = 512  # Hidden dimension size
    num_layers = 16  # Number of GNN layers
    autoencoder = GNNAutoencoder(num_node_features, embedding_dim, hidden_dim, num_layers).to(device)

    num_epochs = 1e5
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)
    train_loss_values = []
    val_loss_values = []
    lowest_validation_loss = np.inf

    # Early stopping parameters
    patience = 10  # Number of epochs to wait for improvement
    min_delta = 0.001  # Minimum change to qualify as an improvement
    best_epoch = 0
    early_stopping_counter = 0
    for epoch in range(int(num_epochs)):
        total_loss = 0
        for n, data in enumerate(training):
            data = data.to(device)
            optimizer.zero_grad()
            reconstructed = autoencoder(data.x, data.edge_index)
            loss = F.mse_loss(reconstructed, data.x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(training)
        train_loss_values.append(avg_train_loss)
        print(f'Epoch {epoch + 1}, Avg Loss: {avg_train_loss}')

        # also check validation loss
        with torch.no_grad():
            for n, data in enumerate(validation):
                data = data.to(device)
                reconstructed = autoencoder(data.x, data.edge_index)
                loss = F.mse_loss(reconstructed, data.x)
                total_loss += loss.item()
            avg_val_loss = total_loss / len(validation)
            val_loss_values.append(avg_val_loss)
            print(f'Epoch {epoch + 1}, Avg Validation Loss: {avg_val_loss}')

        # Check if validation loss improved
        if avg_val_loss < (lowest_validation_loss - min_delta):
            print(
                f'Validation loss decreased ({lowest_validation_loss:.6f} --> {avg_val_loss:.6f}). Saving model...')
            lowest_validation_loss = avg_val_loss
            best_epoch = epoch
            early_stopping_counter = 0
            torch.save(autoencoder.state_dict(), os.path.join(savedir, f"{current_dt_str}-autoencoder.pt"))
        else:
            early_stopping_counter += 1
            print(f'EarlyStopping counter: {early_stopping_counter} out of {patience}')
            if early_stopping_counter >= patience:
                print(
                    f'Early stopping triggered. Stopping at epoch {epoch + 1}. Best epoch was {best_epoch + 1} with loss {lowest_validation_loss:.6f}.')
                break

        # plot every 20 epochs
        if epoch % 10 == 0:
            print('Plotting')
            plt.plot(train_loss_values, label='Training Loss')
            plt.plot(val_loss_values, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Average Loss')
            plt.title('Loss Over Epochs')
            plt.legend()
            # save the plot
            plt.savefig(os.path.join(savedir, f"{current_dt_str}-loss_plot.png"))
            plt.close()

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


def run_decoder_from_embeddings(model_path, original_dataset, embeddings):
    model = GNNAutoencoder(original_dataset.num_features, 512, 512, 16).to(device)
    model.load_state_dict(torch.load(model_path))
    embeddings_torch = torch.tensor(embeddings, dtype=torch.float).to(device)
    original_dataset_edge_index = original_dataset.edge_index.to(device)
    with torch.no_grad():
        out = model.decoder(embeddings_torch, original_dataset_edge_index).cpu().numpy()
    return out


def run_model(model_path, dataset, reconstruction=False):
    model = GNNAutoencoder(dataset.num_features, 512, 512, 16).to(device)
    model.load_state_dict(torch.load(model_path))
    dataset = dataset.to(device)
    with torch.no_grad():
        if reconstruction:
            out = model(dataset.x, dataset.edge_index).cpu().numpy()
        else:
            out = model.encoder(dataset.x, dataset.edge_index).cpu().numpy()
    return out

def import_data(im_path, ch=0):
    # im_path = r"D:\test_files\nelly_tests\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif"
    im_info = ImInfo(im_path, ch=ch)
    #load graph_features as pd.DataFrame
    graph_features = pd.read_csv(im_info.pipeline_paths['graph_features'])
    # load graph_edges as pd.DataFrame
    graph_edges = pd.read_csv(im_info.pipeline_paths['graph_edges'])
    # group by column 't'
    graph_features_grouped = graph_features.groupby('t')
    graph_edges_grouped = graph_edges.groupby('t')
    # create a list of numpy arrays, each array is a frame
    graph_features_list = [group.to_numpy() for _, group in graph_features_grouped]
    graph_edges_list = [group.to_numpy() for _, group in graph_edges_grouped]
    # drop the 't' column
    graph_features_list = [frame[:, 1:] for frame in graph_features_list]
    graph_edges_list = [frame[:, 1:] for frame in graph_edges_list]
    # convert to torch tensors
    graph_features_list = [torch.tensor(frame, dtype=torch.float) for frame in graph_features_list]
    graph_edges_list = [torch.tensor(frame, dtype=torch.long) for frame in graph_edges_list]
    # transpose edge list
    graph_edges_list = [edge_list.t() for edge_list in graph_edges_list]
    # create a list of Data objects
    datasets = [Data(x=graph_features, edge_index=graph_edges) for graph_features, graph_edges in zip(graph_features_list, graph_edges_list)]

    return datasets

if __name__ == '__main__':
    model_path = r"D:\test_files\nelly_tests\20231214_190130-autoencoder - Copy.pt"
    datasets = import_data(r"D:\test_files\nelly_tests\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif")
    normalized_datasets = [Data(x=normalize_features(dataset.x), edge_index=dataset.edge_index) for dataset in datasets]

    training_datasets = normalized_datasets[:5]
    validation_datasets = normalized_datasets[5:]

    training_dataloader = DataLoader(training_datasets, batch_size=1, shuffle=True)
    validation_dataloader = DataLoader(validation_datasets, batch_size=1, shuffle=True)

    train_model(training_dataloader, validation_dataloader)

    # datasets = [dataset_0_norm, ]
    # test_dataset_num = len(datasets) - 1
    # reconstruction = run_model(model_path, normalized_datasets[test_dataset_num])
    # reconstruction = (reconstruction * datasets[test_dataset_num].x.std(dim=0, keepdim=True).cpu().numpy() +
    #                   datasets[test_dataset_num].x.mean(dim=0, keepdim=True).cpu().numpy())
    # real_features = datasets[test_dataset_num].x.cpu().numpy()

    # get embeddings for each dataset
    embeddings_og = [run_model(model_path, dataset) for dataset in normalized_datasets]
    # add labels
    labels = []
    for i, dataset in enumerate(embeddings_og):
        labels += [i] * len(dataset)
    embeddings = np.vstack(embeddings_og)

    # label by dataset
    # dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=500)
    reduced_embeddings = tsne.fit_transform(embeddings[::50])

    # plot
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels[::50], cmap='tab10')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.title('t-SNE Visualization of All Dataset Embeddings')
    plt.legend()
    plt.show()

    new_dataset_paths = [
        r"D:\test_files\nelly_smorgasbord\deskewed-iono_post.ome.tif",
        r"D:\test_files\nelly_smorgasbord\deskewed-iono_pre.ome.tif",
        r"D:\test_files\nelly_smorgasbord\deskewed-mt_ends.ome.tif",
        r"D:\test_files\nelly_smorgasbord\deskewed-peroxisome.ome.tif",
    ]
    new_datasets = [import_data(path)[0] for path in new_dataset_paths]
    new_normalized_datasets = [Data(x=normalize_features(dataset_new.x), edge_index=dataset_new.edge_index) for dataset_new in new_datasets]
    new_embeddings_list = [run_model(model_path, dataset_new) for dataset_new in new_normalized_datasets]
    new_labels = []
    for i, dataset in enumerate(new_embeddings_list):
        new_labels += [i] * len(dataset)
    new_embeddings = np.vstack(new_embeddings_list)

    skip_size = 1
    new_reduced_embeddings = tsne.fit_transform(new_embeddings_list[1][::skip_size])

    plt.figure(figsize=(8, 6))
    plt.scatter(new_reduced_embeddings[:, 0], new_reduced_embeddings[:, 1], alpha=0.75, s=5,)# c=new_labels[::skip_size], cmap='tab10', )
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.title('t-SNE Visualization of All Dataset Embeddings')
    plt.legend()
    plt.show()

    embeds_to_use = embeddings_og

    from scipy.spatial.distance import cdist
    similarity_matrix = np.zeros((len(embeds_to_use), len(embeds_to_use)))
    # Compute cosine similarity (Note: 'cdist' returns the distance, so you need to subtract it from 1)
    for i in range(len(embeds_to_use)):
        for j in range(i + 1, len(embeds_to_use)):
            cosine_similarity = 1 - cdist(embeds_to_use[i], embeds_to_use[j], metric='cosine')
            mean_cosine_similarity = cosine_similarity.mean()
            similarity_matrix[i, j] = mean_cosine_similarity
            similarity_matrix[j, i] = mean_cosine_similarity
        similarity_matrix[i, i] = 1

    #plot heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(similarity_matrix, cmap='viridis')
    plt.show()

    all_embeddings = embeddings_og + new_embeddings_list
    mean_embeddings = [np.mean(embed, axis=0) for embed in all_embeddings]
    # compute cosine similarity
    similarity_matrix = np.zeros((len(mean_embeddings), len(mean_embeddings)))
    for i in range(len(mean_embeddings)):
        for j in range(i + 1, len(mean_embeddings)):
            cosine_similarity = 1 - cdist(mean_embeddings[i].reshape(1, -1), mean_embeddings[j].reshape(1, -1), metric='cosine')
            mean_cosine_similarity = cosine_similarity.mean()
            similarity_matrix[i, j] = mean_cosine_similarity
            similarity_matrix[j, i] = mean_cosine_similarity
        similarity_matrix[i, i] = 1

    # normalize between 0 and 1
    similarity_matrix = (similarity_matrix - similarity_matrix.min()) / (similarity_matrix.max() - similarity_matrix.min())

    #plot heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(similarity_matrix, cmap='viridis')

    # plt.show()

    mean_embeddings_all = np.vstack(mean_embeddings)
    mean_embeddings_labels = range(len(mean_embeddings_all))

    tsne = TSNE(n_components=2, random_state=42, perplexity=2)
    reduced_mean_embeddings = tsne.fit_transform(mean_embeddings_all)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_mean_embeddings[:, 0], reduced_mean_embeddings[:, 1], alpha=0.75, s=5, c=mean_embeddings_labels, cmap='turbo', )
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.title('t-SNE Visualization of All Dataset Embeddings')
    plt.legend()
    plt.show()

