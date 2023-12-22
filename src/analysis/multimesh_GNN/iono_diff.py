import os
import napari

import numpy as np
import torch

from src.analysis.multimesh_GNN.compare_output_to_original import Reconstructor, create_sphere, generate_array
from src.analysis.multimesh_GNN.scratch_multimesh_GNN import import_data, normalize_features, run_model, \
    run_decoder_from_embeddings, GNNAutoencoder
from torch_geometric.data import Data
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import datetime
from scipy.optimize import curve_fit
from scipy.fft import fft

from src.im_info.im_info import ImInfo

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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
    # todo try to get the "iono" vector, then add it to the control and see if it's similar to iono
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
    # datasets = datasets[50:]
    # labels = labels[50:]

    embeddings = get_embeddings(datasets, model_path)
    # median_embeddings = [np.median(embed, axis=0) for embed in embeddings]
    mean_embeddings = [np.mean(embed, axis=0) for embed in embeddings]


    # All of our controls are from frames 0 to 18, so lets get the mean of those to compare to.
    control_timepoints = 18
    mean_control_embeddings = np.mean(mean_embeddings[:control_timepoints], axis=0).tolist()

    # Let's have our controls as one embedding, and compare the rest to that
    new_embeddings = np.array([mean_control_embeddings] + mean_embeddings[control_timepoints:])
    new_labels = np.array([0] + labels[control_timepoints:])
    similarity_matrix = get_similarity_matrix(new_embeddings, normalize=False)

    # Let's analyze how things change over time
    peak_dissimilarity = get_peak_difference_time(similarity_matrix)
    dissimilarity_to_control = 1-similarity_matrix[0, 1:]
    plot_embedding_changes(dissimilarity_to_control)
    real_frame_num = peak_dissimilarity + control_timepoints
    print(f"Peak dissimilarity is at frame {peak_dissimilarity}")

    peak_dissimilarity_embedding = embeddings[real_frame_num]
    diff_embedding = peak_dissimilarity_embedding - mean_control_embeddings
    shift_to_control_embedding = embeddings[real_frame_num] - mean_embeddings[real_frame_num] + mean_control_embeddings

    control_timepoint = 0
    im_info_control = ImInfo(dataset_paths[0])
    reconstructor = Reconstructor(im_info_control, t=control_timepoint+1)
    reconstructor.run()
    skel_idxs = np.argwhere(reconstructor.pixel_class > 0)
    # every 10 frames
    frame_range = range(0, len(embeddings))
    num_frames = len(frame_range)
    new_im = np.zeros((num_frames, *reconstructor.im_memmap.shape), dtype=np.uint16)
    for i, frame_after_control in enumerate(frame_range):
        print(f'Processing frame {frame_after_control} of {len(embeddings)}')
        shift_control_to_treated = embeddings[control_timepoint] - mean_embeddings[control_timepoint] + \
                                   mean_embeddings[frame_after_control]
        mean_shift_control_to_treated = np.mean(shift_control_to_treated, axis=0)

        # reconstructed_diff = run_decoder_from_embeddings(model_path, datasets[real_frame_num], diff_embedding)
        # # get mean of reconstructed diff columns
        # mean_reconstructed_diff = np.mean(reconstructed_diff, axis=0)

        reconstruct_control = run_decoder_from_embeddings(model_path, datasets[control_timepoint], shift_control_to_treated)

        treated_dataset = datasets[frame_after_control]
        reconstruction_original = (reconstruct_control * treated_dataset.x.std(dim=0, keepdim=True).cpu().numpy() +
                                   treated_dataset.x.mean(dim=0, keepdim=True).cpu().numpy())

        assert len(skel_idxs) == len(reconstruction_original)


        means = reconstruction_original[:, 1]
        maxs = reconstruction_original[:, 2]
        mins = reconstruction_original[:, 3]
        covs = reconstruction_original[:, 4]

        radii = reconstruction_original[:, 0] / 2
        # generate a sphere of radius radii for each point
        spheres = []
        for idx, skel_idx in enumerate(skel_idxs):
            int_radius = int(np.round(radii[idx] + 1))
            # sphere = viewer.add_points([skel_idx], size=radii[idx]*2, face_color='red', edge_color='red')
            # spheres.append(sphere)
            sphere = create_sphere(int_radius).astype(np.float32)
            true_locs = np.argwhere(sphere)
            fill_array = generate_array(means[idx], mins[idx], maxs[idx], covs[idx], len(true_locs))
            sphere[true_locs[:, 0], true_locs[:, 1], true_locs[:, 2]] = fill_array
            sphere = sphere.astype(new_im.dtype)
            spheres.append(sphere)
            min_idx = skel_idx - int_radius
            max_idx = skel_idx + int_radius

            min_x = max(0, min_idx[0])
            min_y = max(0, min_idx[1])
            min_z = max(0, min_idx[2])
            max_x = min(new_im[i].shape[0], max_idx[0])
            max_y = min(new_im[i].shape[1], max_idx[1])
            max_z = min(new_im[i].shape[2], max_idx[2])

            x_len = max_x - min_x
            y_len = max_y - min_y
            z_len = max_z - min_z

            # the new_im coords at that location should be the max of the existing value and the new value
            new_im[i][min_x:max_x, min_y:max_y, min_z:max_z][new_im[i][min_x:max_x, min_y:max_y, min_z:max_z]==0] = sphere[:x_len, :y_len, :z_len][new_im[i][min_x:max_x, min_y:max_y, min_z:max_z]==0]
            new_im[i][min_x:max_x, min_y:max_y, min_z:max_z] = np.mean([new_im[i][min_x:max_x, min_y:max_y, min_z:max_z],
                                                                     sphere[:x_len, :y_len, :z_len]],axis=0)

        # todo use the frangi filtered reconstructured sphere as a probability distribution for the intensities?



    viewer = napari.Viewer()
    viewer.add_image(reconstructor.im_memmap * (reconstructor.label_memmap > 0))
    # todo actually in reality, I should reconstruct it with the original data too, and compare those to each other.
    # viewer.add_image(reconstructor.pixel_class)
    viewer.add_image(new_im)

    im_info_peak = ImInfo(dataset_paths[1])
    reconstructor_control = Reconstructor(im_info_peak, t=peak_dissimilarity+control_timepoints+1)
    reconstructor_control.run()
    viewer.add_image(reconstructor_control.im_memmap)
