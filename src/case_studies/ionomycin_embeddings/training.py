import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.case_studies.ionomycin_embeddings.multimesh_GNN import import_data, normalize_features, train_model

dataset_paths = [
    r"D:\test_files\nelly_iono\full_2\deskewed-pre_1.ome.tif",
    r"D:\test_files\nelly_iono\full_2\deskewed-full_post_1.ome.tif",
    r"D:\test_files\nelly_iono\full_2\deskewed-pre_2.ome.tif",
    r"D:\test_files\nelly_iono\full_2\deskewed-full_post_2.ome.tif",
    r"D:\test_files\nelly_iono\full_pre\deskewed-full_pre.ome.tif"
]

datasets = []
training_datasets = []
testing_datasets = []
for dataset_path in dataset_paths:
    print(dataset_path)
    new_datasets = import_data(dataset_path)
    num_new_datasets = len(new_datasets)
    testing_size = int(num_new_datasets * 0.3)
    datasets.extend(new_datasets)
    # get a random 30% for testing, the rest for testing
    random_list = np.random.choice(len(new_datasets), testing_size, replace=False)
    for i in range(len(new_datasets)):
        if i in random_list:
            testing_datasets.append(new_datasets[i])
        else:
            training_datasets.append(new_datasets[i])

normalized_training_datasets = [Data(x=normalize_features(dataset.x), edge_index=dataset.edge_index) for dataset in training_datasets]
normalized_testing_datasets = [Data(x=normalize_features(dataset.x), edge_index=dataset.edge_index) for dataset in testing_datasets]

training_dataloader = DataLoader(normalized_training_datasets, batch_size=1, shuffle=True)
testing_dataloader = DataLoader(normalized_testing_datasets, batch_size=1, shuffle=True)

train_model(training_dataloader, testing_dataloader, r"D:\test_files\nelly_iono")
