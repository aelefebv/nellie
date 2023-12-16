import numpy as np

from src.analysis.multimesh_GNN.scratch_multimesh_GNN import import_data, normalize_features, run_model
from torch_geometric.data import Data
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


model_path = r"D:\test_files\nelly_tests\20231214_190130-autoencoder - Copy.pt"

dataset_paths = [
    r"D:\test_files\nelly_iono\deskewed-pre_0-19.ome.tif",
    r"D:\test_files\nelly_iono\deskewed-post_1_0-19.ome.tif",
]
datasets = []
for dataset_path in dataset_paths:
    print(dataset_path)
    datasets.extend(import_data(dataset_path))

normalized_datasets = [Data(x=normalize_features(dataset.x), edge_index=dataset.edge_index) for dataset in datasets]
embeddings = [run_model(model_path, normalized_dataset) for normalized_dataset in normalized_datasets]
mean_embeddings = [np.mean(embed, axis=0) for embed in embeddings]


all_embeddings = np.vstack(embeddings)
# all embeddings from the first 18 files are the first group, the last 18 are the second group. There should be as many labels as there are all_embeddings
all_embeddings_labels = []
for i in range(len(embeddings)):
    if i < len(embeddings) // 2:
        all_embeddings_labels.extend([0] * len(embeddings[i]))
    else:
        all_embeddings_labels.extend([1] * len(embeddings[i]))
all_embeddings_labels = np.array(all_embeddings_labels)

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
plt.imshow(similarity_matrix, cmap='turbo')
plt.colorbar()
plt.show()

# similarity_to_0
similarity_to_first = similarity_matrix[0, 1:]
similarity_to_last = similarity_matrix[-1, :-1]

# color by filenum
plt.figure(figsize=(8, 6))
plt.scatter(similarity_to_first, similarity_to_last, label='Similarity to First File', c=np.arange(len(similarity_to_first)), cmap='turbo')
plt.xlabel('Similarity to First File')
plt.ylabel('Similarity to Last File')
plt.title('Similarity of Embeddings to First and Last Files')
plt.legend()
plt.show()


mean_embeddings_all = np.vstack(mean_embeddings)
# every 18 files is a group
mean_embeddings_labels = np.repeat(np.arange(len(mean_embeddings) // 18), 18)


tsne = TSNE(n_components=2, random_state=42, perplexity=20)
reduced_mean_embeddings = tsne.fit_transform(mean_embeddings_all)
# reduced_mean_embeddings = tsne.fit_transform(all_embeddings[::50])

# color is categorical by filename
plt.figure(figsize=(8, 6))
plt.scatter(reduced_mean_embeddings[:, 0], reduced_mean_embeddings[:, 1], c=mean_embeddings_labels, cmap='Paired')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.title('t-SNE Visualization of All Dataset Embeddings')
plt.legend
plt.show()
