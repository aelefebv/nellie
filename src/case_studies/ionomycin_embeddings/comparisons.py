import os

import numpy as np

from src.case_studies.ionomycin_embeddings.multimesh_GNN import import_data, normalize_features, run_model
from torch_geometric.data import Data
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import datetime
from scipy.optimize import curve_fit
from scipy.fft import fft


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


def get_reconstruction_from_embeddings(embeddings, model_path):
    normalized_embeddings = [normalize_features(embedding) for embedding in embeddings]
    reconstructions = [run_model(model_path, normalized_embedding, decode=True) for normalized_embedding in normalized_embeddings]
    return reconstructions


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

def plot_similarity_matrix(similarity_matrix, save_plots=False, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(similarity_matrix, cmap='turbo')
    plt.xlabel('File Number')
    plt.ylabel('File Number')
    plt.title('Similarity Matrix of All Dataset Embeddings')
    if save_plots:
        plt.savefig(os.path.join(save_path, f'{current_dt_str}-similarity_matrix.png'))
        plt.close()
    else:
        plt.show()

def plot_embedding_changes(dissimilarity_to_control, save_plots=False, save_path=None, line=False):
    # plot similarity to control
    plt.figure(figsize=(8, 6))
    if line:
        plt.plot(np.arange(len(dissimilarity_to_control)), dissimilarity_to_control, 'r-', label='Similarity to Control')
    plt.scatter(np.arange(len(dissimilarity_to_control)), dissimilarity_to_control)
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
    fit_y = exponential_decay(xdata, *fit_params[0])
    plt.scatter(xdata, recovery_array, label='Dissimilarity to Control')
    plt.plot(xdata, fit_y, 'r-', label='Exponential fit')
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

    return fit_y

def plot_tsne(reduced_mean_embeddings, labels, alpha=1, size=10, cmap='turbo', save_plots=False, save_path=None):
    # color is categorical by filename
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_mean_embeddings[:, 0], reduced_mean_embeddings[:, 1], c=labels, cmap=cmap,
                alpha=alpha, s=size)
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.title('t-SNE Visualization of All Dataset Embeddings')
    if save_plots:
        plt.savefig(os.path.join(save_path, f'{current_dt_str}-tsne_all.png'))
        plt.close()
    else:
        plt.show()

def get_frequency_stats(vec):
    fs = 1  # in Hz
    duration = len(vec)  # in seconds
    t = np.arange(0, duration, 1 / fs)

    signal = vec - np.mean(vec)  # remove DC component

    signal_fft = fft(signal)
    freq = np.fft.fftfreq(len(t), 1 / fs)

    # Compute the magnitude of the FFT (two-sided spectrum)
    magnitude = np.abs(signal_fft)
    # get all peaks, ignore negative peaks
    magnitude = magnitude[:len(magnitude) // 2]
    freq = freq[:len(freq) // 2]
    # find all peaks
    peak_indices = np.argsort(magnitude)[-5:][::-1]
    peak_freqs = freq[peak_indices]
    peak_magnitudes = magnitude[peak_indices]
    print(f"The 5 strongest oscillating wavelengths are: {1/peak_freqs}")
    print(f"Their magnitudes are: {peak_magnitudes}")
    return np.array([1/peak_freqs, peak_magnitudes]).T


if __name__ == '__main__':
    # model_path = r"D:\test_files\nelly_tests\20231215_145237-autoencoder - Copy.pt"
    model_path = r"D:\test_files\nelly_tests\20231219_122843-autoencoder - Copy.pt"
    file_set_num = 1

    dataset_paths = [
        # r"D:\test_files\nelly_iono\full_2\deskewed-pre_2.ome.tif",
        # r"D:\test_files\nelly_iono\full_2\deskewed-full_post_2.ome.tif",
        rf"D:\test_files\nelly_iono\full_2\deskewed-pre_{file_set_num}.ome.tif",
        rf"D:\test_files\nelly_iono\full_2\deskewed-full_post_{file_set_num}.ome.tif",
        # r"D:\test_files\nelly_iono\full_pre\deskewed-full_pre.ome.tif",
        # r"D:\test_files\nelly_iono\full_2\deskewed-full_pre_2.ome.tif"
        # r"D:\test_files\nelly_iono\full\deskewed-full_v2.ome.tif"
    ]
    save_path = r"D:\test_files\nelly_iono\full_2"

    # Import datasets and get some embeddings
    datasets, labels = import_datasets(dataset_paths)
    #testing
    datasets = datasets[50:]
    labels = labels[50:]

    embeddings = get_embeddings(datasets, model_path)
    median_embeddings = [np.median(embed, axis=0) for embed in embeddings]
    mean_embeddings = [np.mean(embed, axis=0) for embed in embeddings]

    # All of our controls are from frames 0 to 18, so lets get the mean of those to compare to.
    control_timepoints = 18
    mean_control_embeddings = np.mean(median_embeddings[:control_timepoints], axis=0).tolist()

    # Let's have our controls as one embedding, and compare the rest to that
    new_embeddings = np.array([mean_control_embeddings] + median_embeddings[control_timepoints:])
    new_labels = np.array([0] + labels[control_timepoints:])
    similarity_matrix = get_similarity_matrix(new_embeddings, normalize=False)
    # save as csv
    np.savetxt(os.path.join(save_path, f'{current_dt_str}-similarity_matrix_mean_control-{file_set_num}.txt'), similarity_matrix)

    # Let's analyze how things change over time
    peak_dissimilarity = get_peak_difference_time(similarity_matrix)
    dissimilarity_to_control = 1-similarity_matrix[0, 1:]
    plot_embedding_changes(dissimilarity_to_control)
    print(f"Peak dissimilarity is at frame {peak_dissimilarity}")
    np.savetxt(os.path.join(save_path, f'{current_dt_str}-difference_to_first-{file_set_num}.txt'), dissimilarity_to_control)

    similarity_to_first = similarity_matrix[0, 1:]
    similarity_to_peak_dissimilarity = similarity_matrix[peak_dissimilarity, 1:]
    both_arrays = np.array([similarity_to_first, similarity_to_peak_dissimilarity]).T
    np.savetxt(os.path.join(save_path, f'{current_dt_str}-similarity_to_first_vs_peak-{file_set_num}.txt'), both_arrays)

    # Let's fit an exponential decay to the recovery curve to analyze recovery time
    recovery_array = dissimilarity_to_control[peak_dissimilarity+1:]
    xdata = np.arange(len(recovery_array))
    r_squared, fit_params = get_recovery(xdata, recovery_array)
    fit_y = plot_recovery(xdata, recovery_array, fit_params, r_squared)
    print(f"Recovery time is {1/fit_params[0][1]:.2f} frames")
    output_recovery_array = np.array([xdata, recovery_array, fit_y]).T
    output_recovery_stats = np.array([r_squared, 1/fit_params[0][1]])
    np.savetxt(os.path.join(save_path, f'{current_dt_str}-recovery-{file_set_num}.txt'), output_recovery_array)
    np.savetxt(os.path.join(save_path, f'{current_dt_str}-recovery_stats-{file_set_num}.txt'), output_recovery_stats)

    # Let's plot all of our embeddings in 2D space to visualize any other trends
    stacked_embeddings = np.vstack(mean_embeddings)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_embeddings = tsne.fit_transform(stacked_embeddings)
    frame_array = np.arange(len(stacked_embeddings))
    plot_tsne(reduced_embeddings, frame_array)

    # Are there any other smaller patterns happening?
    similarity_matrix = get_similarity_matrix(mean_embeddings, normalize=False)
    # similarity_matrix = get_similarity_matrix(mean_embeddings[36:])
    plot_similarity_matrix(similarity_matrix)
    np.savetxt(os.path.join(save_path, f'{current_dt_str}-similarity_matrix_all-{file_set_num}.txt'), similarity_matrix)

    # Let's look at the frequency of the whole treatment curve
    similarity_1 = similarity_matrix[0, control_timepoints:]
    np.savetxt(os.path.join(save_path, f'{current_dt_str}-similarity_to_first-{file_set_num}.txt'), similarity_1)
    # low_pass filter
    from scipy.signal import butter, filtfilt

    plot_embedding_changes(similarity_1, line=False)
    # Let's get a filter for every 5 frames
    b, a = butter(2, 0.2)
    vec_filtered = filtfilt(b, a, similarity_1)
    np.savetxt(os.path.join(save_path, f'{current_dt_str}-similarity_to_first_5f_filtered-{file_set_num}.txt'), vec_filtered)
    plot_embedding_changes(vec_filtered, line=True)
    # Let's get a filter for every 20 frames to get a good average
    b2, a2 = butter(2, 0.05)
    vec_bigger_filter = filtfilt(b2, a2, similarity_1)
    np.savetxt(os.path.join(save_path, f'{current_dt_str}-similarity_to_first_20f_filtered-{file_set_num}.txt'), vec_bigger_filter)
    plot_embedding_changes(vec_bigger_filter, line=True)

    # Let's remove the low frequency component (remove the average)
    low_freq_removed_and_filtered = vec_filtered - vec_bigger_filter
    np.savetxt(os.path.join(save_path, f'{current_dt_str}-similarity_to_first_5f_filtered_low_freq_removed-{file_set_num}.txt'), low_freq_removed_and_filtered)
    plot_embedding_changes(low_freq_removed_and_filtered, line=True)
    # Let's see what's happening after that.
    freq_stats = get_frequency_stats(low_freq_removed_and_filtered)
    np.savetxt(os.path.join(save_path, f'{current_dt_str}-similarity_to_first_5f_filtered_low_freq_removed_freq_stats-{file_set_num}.txt'), freq_stats)


    # # now let's find the individual nodes that are most dissimilar to the control
    # # eg for max timepoint after treatment:
    # max_timepoint = peak_dissimilarity + control_timepoints
    # node_diffs = np.abs(embeddings[max_timepoint] - mean_control_embeddings)
    # tsne = TSNE(n_components=2, random_state=42, perplexity=200)
    # max_timepoint_embeddings = embeddings[max_timepoint][::10]
    # last_timepoint_embeddings = embeddings[-1][::10]
    # # combined
    # reduced_embeddings = tsne.fit_transform(np.vstack((max_timepoint_embeddings, last_timepoint_embeddings)))
    # labels = np.array([0] * len(max_timepoint_embeddings) + [1] * len(last_timepoint_embeddings))
    # # color array by mean node difference value
    # mean_node_diffs = np.mean(node_diffs, axis=1)
    # # plot_tsne(reduced_embeddings, mean_node_diffs[::10], alpha=0.5, size=2)
    # plot_tsne(reduced_embeddings, labels, alpha=0.75, size=5, cmap='bwr')

