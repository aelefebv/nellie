import os

import numpy as np

from src.analysis.multimesh_GNN.scratch_multimesh_GNN import import_data, normalize_features, run_model
from torch_geometric.data import Data
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import datetime
from scipy.optimize import curve_fit

# YYYYMMDD_HHMMSS
current_dt = datetime.datetime.now()
current_dt_str = current_dt.strftime("%Y%m%d_%H%M%S")


def import_datasets(dataset_paths):
    datasets = []
    labels = []
    for i, dataset_path in enumerate(dataset_paths):
        print(dataset_path)
        new_imports = import_data(dataset_path)
        datasets.extend(new_imports)
        labels.extend([i] * len(new_imports))
    return datasets, labels


def get_embeddings(datasets, model_path):
    normalized_datasets = [Data(x=normalize_features(dataset.x), edge_index=dataset.edge_index) for dataset in datasets]
    embeddings = [run_model(model_path, normalized_dataset) for normalized_dataset in normalized_datasets]
    return embeddings


def get_similarity_matrix(mean_embeddings, normalize=True):
    similarity_matrix = np.zeros((len(mean_embeddings), len(mean_embeddings)))
    for i in range(len(mean_embeddings)):
        for j in range(i + 1, len(mean_embeddings)):
            cosine_similarity = 1 - cdist(mean_embeddings[i].reshape(1, -1), mean_embeddings[j].reshape(1, -1), metric='cosine')
            mean_cosine_similarity = cosine_similarity.mean()
            similarity_matrix[i, j] = mean_cosine_similarity
            similarity_matrix[j, i] = mean_cosine_similarity
        similarity_matrix[i, i] = 1
    if normalize:
        similarity_matrix = (similarity_matrix - similarity_matrix.min()) / (similarity_matrix.max() - similarity_matrix.min())
    return similarity_matrix

def get_peak_difference_time(similarity_matrix):
    similarity_to_first = similarity_matrix[0, 1:]
    peak_dissimilarity = np.argmin(similarity_to_first)
    return peak_dissimilarity

def exponential_decay(x, a, b):
    return a * np.exp(-b * x)

def get_recovery(x, y):
    fit_params = curve_fit(exponential_decay, x, y)

    # calculate goodness of fit via R^2
    residuals = y - exponential_decay(x, *fit_params[0])
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared, fit_params

def plot_embedding_changes(dissimilarity_to_control, save_plots=False, save_path=None):
    # plot similarity to control
    plt.figure(figsize=(8, 6))
    plt.scatter(np.arange(len(dissimilarity_to_control)), dissimilarity_to_control, label='Similarity to Control')
    plt.xlabel('File Number')
    plt.ylabel('Dissimilarity to Control')
    plt.title('Dissimilarity of Embeddings to Control')
    plt.legend()
    if save_plots:
        plt.savefig(os.path.join(save_path, f'{current_dt_str}-dissimilarity_to_control.png'))
        plt.close()
    else:
        plt.show()

def plot_recovery(xdata, recovery_array, fit_params, r_squared, save_plots=False, save_path=None):
    # plot decay
    plt.figure(figsize=(8, 6))
    plt.scatter(xdata, recovery_array, label='Dissimilarity to Control')
    plt.plot(xdata, exponential_decay(xdata, *fit_params[0]), 'r-', label='Exponential fit')
    plt.xlabel('File Number')
    plt.ylabel('Dissimilarity to Control')
    plt.title('Recovery post-treatment\n'
              f'R2 = {r_squared:.2f}\n'
              f'Decay Time = {1/fit_params[0][1]:.2f} frames')
    plt.legend()
    if save_plots:
        plt.savefig(os.path.join(save_path, f'{current_dt_str}-recovery.png'))
        plt.close()
    else:
        plt.show()

def plot_tsne(reduced_mean_embeddings, labels, save_plots=False, save_path=None):
    # color is categorical by filename
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_mean_embeddings[:, 0], reduced_mean_embeddings[:, 1], c=labels, cmap='turbo')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.title('t-SNE Visualization of All Dataset Embeddings')
    if save_plots:
        plt.savefig(os.path.join(save_path, f'{current_dt_str}-tsne_all.png'))
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    model_path = r"D:\test_files\nelly_tests\20231215_145237-autoencoder - Copy.pt"

    dataset_paths = [
        r"D:\test_files\nelly_iono\deskewed-pre_0-19.ome.tif",
        r"D:\test_files\nelly_iono\full_2\deskewed-full_post_1.ome.tif",
    ]

    datasets, labels = import_datasets(dataset_paths)
    embeddings = get_embeddings(datasets, model_path)
    avg_embeddings = [np.median(embed, axis=0) for embed in embeddings]

    control_timepoints = 18
    mean_control_embeddings = np.mean(avg_embeddings[:control_timepoints], axis=0).tolist()

    # mean_control and all treated
    new_embeddings = np.array([mean_control_embeddings] + avg_embeddings[control_timepoints:])
    new_labels = np.array([0] + labels[control_timepoints:])
    similarity_matrix = get_similarity_matrix(new_embeddings)

    peak_dissimilarity = get_peak_difference_time(similarity_matrix)

    dissimilarity_to_control = 1-similarity_matrix[0, 1:]
    plot_embedding_changes(dissimilarity_to_control)

    recovery_array = dissimilarity_to_control[peak_dissimilarity:]
    xdata = np.arange(len(recovery_array))
    r_squared, fit_params = get_recovery(xdata, recovery_array)
    plot_recovery(xdata, recovery_array, fit_params, r_squared)

    stacked_embeddings = np.vstack(avg_embeddings)

    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(stacked_embeddings)
    frame_array = np.arange(len(stacked_embeddings))
    plot_tsne(reduced_embeddings, frame_array)



save_plots = False
save_path = r"D:\test_files\nelly_tests"

ratio = similarity_to_least_similar / similarity_to_first
ratio_test = ratio[18:] #ignore pre-treated
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Sample rate and duration
fs = 1  # in Hz
duration = 97  # in seconds

# Time array
t = np.arange(0, duration, 1/fs)

# Generate a signal, for example, a sine wave
frequency = 50  # in Hz
signal = ratio_test_filtered

signal = signal - np.mean(signal)

# Apply FFT
signal_fft = fft(signal)

# Compute the frequency axis
freq = np.fft.fftfreq(len(t), 1/fs)

# Compute the magnitude of the FFT (two-sided spectrum)
magnitude = np.abs(signal_fft)

# Find the first peak frequency
magnitude[0] = 0  # ignore DC component
first_peak_index = np.argmax(magnitude)
first_peak_freq = freq[first_peak_index]

# Zero out the first peak
magnitude[first_peak_index] = 0

# Find the second peak frequency
second_peak_index = np.argmax(magnitude)
second_peak_freq = freq[second_peak_index]


# Plotting
plt.plot(freq, magnitude)
plt.title('FFT of the signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim([0, fs/2])  # Limit x-axis to half the sampling rate
plt.show()

print(f"The 2nd strongest oscillating frequency is: {second_peak_freq:.2f} Hz")



# ratio = similarity_to_first
# low pass filter using scipy
from scipy.signal import butter, filtfilt

b, a = butter(2, 0.3)
ratio_test_filtered = filtfilt(b, a, ratio_test)

