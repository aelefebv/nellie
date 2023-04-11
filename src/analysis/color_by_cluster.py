import os
import pandas as pd
import napari
import tifffile
import re
from sklearn.preprocessing import LabelEncoder
from src.analysis.preprocessing.clustering import ClusteringAnalysis
from src.analysis.preprocessing.csv_preprocessing import fill_na_in_csv
from src.io.pickle_jar import unpickle_object
from src.utils.general import get_reshaped_image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from src import logger
import logging
logger.basicConfig(level=logging.INFO)

def plot_umap_from_df(df, color_by=None, highlight_value=None):
    # Define your list of colors
    # colors = ['#46231a', '#317485', '#46507d', '#3e1968',
    #           '#2d2e5a', '#5d7e54', '#4f6064', '#4b5358',
    #           '#88832b', '#863030', '#12352e', '#426240',]

    colors = ['#a34129', '#298ba3', '#293fa3', '#6228a4',
              '#292ba3', '#8cd99d', '#8f9fa3', '#75828a',
              '#a39d29', '#eb4747', '#145233', '#53b34d',
              '#749c63', '#4b3e8e', '#5e804d', '#4b96b4',]

    if color_by is not None and color_by in df.columns:
        categories = df[color_by].unique()
        colors = colors * (len(categories) // len(colors) + 1)
        if -1 in categories:
            categories += 2
        elif 0 in categories:
            categories += 1
        color_dict = {category: i for i, category in enumerate(categories)}
        color_data = df[color_by].replace(color_dict)
        cmap = ListedColormap(colors[:len(categories)])
    else:
        color_data = None
        cmap = None

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    plt.title(f'Clusters in UMAP space (colored by {color_by})')
    # Manually create the legend by iterating over the unique cluster labels
    handles = []
    labels = []
    if highlight_value is not None:
        df_highlight = df[df[color_by] == highlight_value]
        df_rest = df[df[color_by] != highlight_value]
        plt.scatter(df_rest['umap_1'], df_rest['umap_2'], c='#cccccc', s=0.5)
        plt.scatter(df_highlight['umap_1'], df_highlight['umap_2'], c='red', s=0.5, alpha=0.5)
        handles.append(plt.scatter([], [], color='red', s=10))
        labels.append(str(highlight_value))
    else:
        plt.scatter(df['umap_1'], df['umap_2'],
                              cmap=cmap, c=color_data, s=0.5, alpha=0.5)
        for i, category in enumerate(categories):
            # print(color_num, len(categories))
            handles.append(plt.scatter([], [], color=colors[i], s=10))
            labels.append(str(category))
    legend1 = ax.legend(handles, labels, title="Clusters", bbox_to_anchor=(1.01, 1),
                        loc=2, borderaxespad=0., handlelength=2.5)
    ax.add_artist(legend1)
    plt.subplots_adjust(right=0.80)
    plt.show()


def get_treatment_info(file_name):
    match = re.search(r'-(\d+)-(\d+)h.ome', file_name)
    if match:
        concentration = int(match.group(1))
        time = int(match.group(2))
        return concentration, time
    else:
        raise ValueError(f"Unable to extract concentration and time information from file_name: {file_name}")


def preprocess_data(top_output_dir, im_name, ch):
    csv_dir = os.path.join(top_output_dir, 'csv')
    file_path = f'summary_stats_regions-{im_name}-ch{ch}.csv'
    df_preprocessed = fill_na_in_csv(csv_dir, file_path)
    concentration, time = get_treatment_info(file_path)
    df_preprocessed['concentration'] = concentration
    df_preprocessed['time'] = time
    return df_preprocessed, file_path


def perform_clustering_analysis(df_preprocessed, top_output_dir, file_path,
                                subsample = 5000, umap_params = None):
    analysis_output_dir = os.path.join(top_output_dir, 'analysis')
    if not os.path.exists(analysis_output_dir):
        os.makedirs(analysis_output_dir)
    ignore_cols = ['group', 'filename', 'frame_number',
                   'branch_ids', 'node_id', 'region_id', 'branch_id', 'node_ids']
    remove_cols = None
    clustering_analysis = ClusteringAnalysis(df_preprocessed, analysis_output_dir, file_path,
                                             corr_threshold=0.8, frames_to_keep=1,
                                             ignore_cols=ignore_cols,
                                             remove_cols=remove_cols, sample_size=subsample, umap_params=umap_params)
    df_out = clustering_analysis.run()
    return df_out, clustering_analysis


def load_region_props(top_output_dir, im_name):
    pkl_dir = os.path.join(top_output_dir, 'pickles')
    pkl_file = f'ch1-obj-{im_name}.pkl'
    full_pkl_file = os.path.join(pkl_dir, pkl_file)
    region_props = unpickle_object(full_pkl_file)
    return region_props


def process_images(region_props, df_out):
    mask_im = tifffile.memmap(region_props.im_info.path_im_mask, mode='r')
    mask_im = get_reshaped_image(mask_im, 2, region_props.im_info)
    new_im = np.zeros(mask_im.shape, dtype=np.uint16)

    frame_to_color = 1
    frame_regions = region_props.organelles[frame_to_color]
    # get a view of the dataframe with the correct filename
    df_out = df_out[df_out['filename'] == region_props.im_info.filename]

    for i, row in df_out.iterrows():
        region = frame_regions[row['region_id']-1]
        for coord in region.coords:
            new_im[frame_to_color, coord[0], coord[1], coord[2]] = row['cluster_umap']+2

    return new_im


def display_images(region_props, new_im):
    viewer = napari.Viewer()
    im_og = tifffile.memmap(region_props.im_info.im_path, mode='r')
    im_og = im_og[:2, 1, ...]
    scaling = [region_props.im_info.dim_sizes['Z'], region_props.im_info.dim_sizes['Y'], region_props.im_info.dim_sizes['X']]
    viewer.add_image(im_og, name='original image', scale=scaling)
    viewer.add_labels(new_im, name='tip_labels', scale=scaling)
    napari.run()


def run_analysis(top_output_dir, im_name, ch):
    df_preprocessed, file_path = preprocess_data(top_output_dir, im_name, ch)
    df_out, clustering_analysis = perform_clustering_analysis(df_preprocessed, top_output_dir, file_path)
    region_props = load_region_props(top_output_dir, im_name)
    new_im = process_images(region_props, df_out)
    display_images(region_props, new_im)


def process_multiple_files(top_output_dir, file_list, ch):
    all_data = pd.DataFrame()
    for file_name in file_list:
        df_preprocessed, _ = preprocess_data(top_output_dir, file_name, ch)
        # add the filename to the df
        df_preprocessed['filename'] = file_name
        all_data = pd.concat([all_data, df_preprocessed], ignore_index=True)
    return all_data


if __name__ == "__main__":
    top_dir = r'D:\test_files\nelly\20230406-AELxKL-dmr_lipid_droplets_mtDR'
    top_output_dir = os.path.join(top_dir, 'output')
    im_name = 'deskewed-2023-04-06_17-01-43_000_AELxKL-dmr_PERK-lipid_droplets_mtDR-5000-4h.ome'
    ch = 1
    # find all files in the top_dir that ends with .ome.tif
    file_list = [f[:-4] for f in os.listdir(top_dir) if f.endswith('.ome.tif')]
    # remove the .tif from the filenames
    all_data = process_multiple_files(top_output_dir, file_list, ch)
    umap_params = {'n_neighbors': 100, 'min_dist': 0.05, 'n_components': 2, 'metric': 'euclidean', 'spread': 1.0, 'random_state': 42}
    df_out, clustering_analysis = perform_clustering_analysis(all_data, top_output_dir, 'combined_files',
                                         subsample=None, umap_params=umap_params)
    # save df_out to a csv in the analysis folder
    df_out.to_csv(os.path.join(top_output_dir, 'analysis', 'combined_files.csv'))
    # run_analysis(top_output_dir, im_name, ch)
    # for testing:
    df_out = pd.read_csv(os.path.join(top_output_dir, 'analysis', 'combined_files.csv'))
    # Plot UMAP colored by concentration
    plot_umap_from_df(df_out, color_by='concentration')
    plot_umap_from_df(df_out, color_by='concentration', highlight_value=5000)
    plot_umap_from_df(df_out, color_by='cluster_umap')
    plot_umap_from_df(df_out, color_by='cluster_pca')

    # Plot UMAP colored by time
    plot_umap_from_df(df_out, color_by='time')

    # Plot UMAP colored by filename

    im_name = file_list[1]
    plot_umap_from_df(df_out, color_by='filename', highlight_value=im_name)

    region_props = load_region_props(top_output_dir, im_name)
    new_im = process_images(region_props, df_out)
    display_images(region_props, new_im)
