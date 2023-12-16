import numpy as np
from torch_geometric.data import Data, DataLoader

from src.analysis.multimesh_GNN.scratch_multimesh_GNN import import_data, normalize_features, train_model

dataset_paths = [
    r"D:\test_files\nelly_iono\deskewed-pre_0-19.ome.tif",
    r"D:\test_files\nelly_iono\deskewed-post_1_0-19.ome.tif",
]
datasets = []
training_datasets = []
testing_datasets = []
for dataset_path in dataset_paths:
    print(dataset_path)
    new_datasets = import_data(dataset_path)
    datasets.extend(new_datasets)
    # get a random 5 for testing, the rest for testing
    random_list = np.random.choice(len(new_datasets), 5, replace=False)
    for i in range(len(datasets)):
        if i in random_list:
            testing_datasets.append(datasets[i])
        else:
            training_datasets.append(datasets[i])

normalized_training_datasets = [Data(x=normalize_features(dataset.x), edge_index=dataset.edge_index) for dataset in training_datasets]
normalized_testing_datasets = [Data(x=normalize_features(dataset.x), edge_index=dataset.edge_index) for dataset in testing_datasets]

training_dataloader = DataLoader(normalized_training_datasets, batch_size=1, shuffle=True)
training_dataloader = DataLoader(normalized_testing_datasets, batch_size=1, shuffle=True)

train_model(training_dataloader, training_dataloader)
