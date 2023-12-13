import os
import pandas as pd
import napari
import tifffile
import re
from src.analysis.old.preprocessing.clustering import ClusteringAnalysis
from src.analysis.old.preprocessing.csv_preprocessing import fill_na_in_csv
from src import unpickle_object
from src import get_reshaped_image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from src import logger
import logging
logger.basicConfig(level=logging.INFO)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def plot_umap_from_df(df, color_by=None, highlight_value=None, full_save_path=None, s=None, vmin=None, vmax=None):
    s = s or 0.5
    # Define your list of colors
    # colors = ['#46231a', '#317485', '#46507d', '#3e1968',
    #           '#2d2e5a', '#5d7e54', '#4f6064', '#4b5358',
    #           '#88832b', '#863030', '#12352e', '#426240',]

    colors = ['#a34129', '#298ba3', '#293fa3', '#6228a4',
              '#292ba3', '#8cd99d', '#8f9fa3', '#75828a',
              '#a39d29', '#eb4747', '#145233', '#53b34d',
              '#749c63', '#4b3e8e', '#5e804d', '#4b96b4',]
    increase = 0
    if color_by is not None and color_by in df.columns:
        categories = df[color_by].unique()
        if len(categories) < len(colors):
            colors = colors * (len(categories) // len(colors) + 1)
            print(categories)
            print(colors)
            color_dict = {category: i + increase for i, category in enumerate(categories)}
            color_data = df[color_by].replace(color_dict)
            cmap = ListedColormap(colors[:len(categories)])
            print(color_data)
        else:
            cmap = 'viridis'
            color_data = df[color_by]
            categories = []
    else:
        color_data = None
        cmap = None
        categories = []

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
        plt.scatter(df_rest['umap_1'], df_rest['umap_2'], c='#cccccc', s=s)
        plt.scatter(df_highlight['umap_1'], df_highlight['umap_2'], c='red', s=s, alpha=0.5)
        handles.append(plt.scatter([], [], color='red', s=10))
        labels.append(str(highlight_value))
    else:
        if vmin is not None and vmax is not None:
            plt.scatter(df['umap_1'], df['umap_2'],
                        cmap=cmap, c=color_data, s=s, alpha=0.5, vmin=vmin, vmax=vmax)
        else:
            plt.scatter(df['umap_1'], df['umap_2'],
                        cmap=cmap, c=color_data, s=s, alpha=0.5)
        for i, category in enumerate(categories):
            # print(color_num, len(categories))
            handles.append(plt.scatter([], [], color=colors[i], s=10))
            labels.append(str(category))
    legend1 = ax.legend(handles, labels, title="Clusters", bbox_to_anchor=(1.01, 1),
                        loc=2, borderaxespad=0., handlelength=2.5)
    ax.add_artist(legend1)
    plt.subplots_adjust(right=0.80)
    if full_save_path is not None:
        plt.savefig(full_save_path, dpi=500, bbox_inches='tight')
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
    remove_cols = ['direction', 'orientation', 'fission', 'fusion', 'concentration', 'time', 'filename']
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
    # get the min of the cluster_umap column
    min_cluster = df_out['cluster_umap'].min()
    # subtract the min from the cluster_umap column
    df_out['cluster_umap'] = df_out['cluster_umap'] - min_cluster + 1

    for i, row in df_out.iterrows():
        region = frame_regions[row['region_id']-1]
        for coord in region.coords:
            new_im[frame_to_color, coord[0], coord[1], coord[2]] = row['cluster_umap']

    return new_im


def display_images(region_props, new_im):
    viewer = napari.Viewer()
    im_og = tifffile.memmap(region_props.im_info.im_path, mode='r')
    im_og = im_og[:2, 1, ...]
    scaling = [region_props.im_info.dim_sizes['Z'], region_props.im_info.dim_sizes['Y'], region_props.im_info.dim_sizes['X']]
    viewer.add_image(im_og, name='original image', scale=scaling)
    viewer.add_labels(new_im, name='tip_labels', scale=scaling)
    return viewer


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


def train_test_split_data(x, y, test_size=0.3, random_state=42):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def remove_highly_correlated_features(x_train, x_test, corr_threshold=0.8):
    corr_matrix = x_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    x_train = x_train.drop(to_drop, axis=1)
    x_test = x_test.drop(to_drop, axis=1)
    return x_train, x_test


def perform_feature_selection(x_train, y_train, x_test, k_best=None):
    if k_best is None:
        k_best = 'all'
    selector = SelectKBest(f_classif, k=k_best)
    selector.fit(x_train, y_train)
    x_train_selected = selector.transform(x_train)
    x_test_selected = selector.transform(x_test)
    return x_train_selected, x_test_selected


def evaluate_models(models, x_train, y_train, x_test, y_test, x, y):
    results = {}
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, x, y, cv=5)

        results[model_name] = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'cross_val_scores': cv_scores,
            'mean_cross_val_score': np.mean(cv_scores)
        }
    return results


def display_evaluation_results(results, save_path, date_time, feature, save=True):
    for model_name, model_results in results.items():
        print(f"Model: {model_name}")
        print(f"Accuracy: {model_results['accuracy']}")
        print(f"Classification Report: \n{model_results['classification_report']}")
        print(f"Cross Validation Scores: {model_results['cross_val_scores']}")
        print(f"Mean Cross Validation Score: {model_results['mean_cross_val_score']}")
        print("----------------------------------------------------------")
    if save:
        with open(os.path.join(save_path, f"{date_time}_{feature}_evaluation_results.txt"), 'w') as f:
            for model_name, model_results in results.items():
                f.write(f"Model: {model_name}\n")
                f.write(f"Accuracy: {model_results['accuracy']}\n")
                f.write(f"Classification Report: \n{model_results['classification_report']}\n")
                f.write(f"Cross Validation Scores: {model_results['cross_val_scores']}\n")
                f.write(f"Mean Cross Validation Score: {model_results['mean_cross_val_score']}\n")
                f.write("----------------------------------------------------------\n")


def feature_ranking(x_train, y_train, random_forest_model, feature_columns, feature_name, save_path, date_time, save=False):
    # Train the random forest classifier on the selected features
    random_forest_model.fit(x_train, y_train)

    # Get the feature importances
    importances = random_forest_model.feature_importances_

    # Get the indices of the features sorted by importance
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for feature in range(x_train.shape[1]):
        print(f"{feature + 1}, {feature_columns[indices[feature]]}, {importances[indices[feature]]}")
    if save:
        with open(os.path.join(save_path, f"{date_time}_{feature_name}_feature_rankings.txt"), 'w') as f:
            f.write("feature_rank, feature, importance")
            for feature in range(x_train.shape[1]):
                f.write(f"\n{feature + 1}, {feature_columns[indices[feature]]}, {importances[indices[feature]]}")

    return importances, indices


def model_worth(normalized_df, full_df, feature, now, k_best_features=10, save_path=None):
    x = normalized_df
    y = full_df[feature]

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split_data(x, y)

    # Remove highly correlated features
    # x_train, x_test = remove_highly_correlated_features(x_train, x_test)

    # Perform feature selection
    x_train_selected, x_test_selected = perform_feature_selection(x_train, y_train, x_test, k_best=k_best_features)

    # Define models to test
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Support Vector Machine': SVC(),
    }

    # Evaluate models
    results = evaluate_models(models, x_train_selected, y_train, x_test_selected, y_test, x, y)

    if save_path is not None:
        # Display evaluation results
        display_evaluation_results(results, save_path, now, feature, save=True)

    # Feature ranking
    random_forest_model = models['Random Forest']
    feature_columns = x_train.columns
    feature_ranking(x_train_selected, y_train, random_forest_model, feature_columns, feature, save_path, now, save=True)


if __name__ == "__main__":
    top_dir = r'D:\test_files\nelly\20230406-AELxKL-dmr_lipid_droplets_mtDR'
    top_output_dir = os.path.join(top_dir, 'output')
    im_name = 'deskewed-2023-04-06_17-06-22_000_AELxKL-dmr_PERK-lipid_droplets_mtDR-5000-4h.ome'
    ch = 1
    # find all files in the top_dir that ends with .ome.tif
    file_list = [f[:-4] for f in os.listdir(top_dir) if f.endswith('.ome.tif')]
    # remove the .tif from the filenames
    # all_data = process_multiple_files(top_output_dir, file_list, ch)
    #
    df_in = pd.read_csv(os.path.join(top_output_dir, 'analysis', 'combined_files.csv'))
    # only get the rows from one file
    df_in = df_in.filter(regex='^(?!Unnamed:)')
    df_out = df_in[df_in['filename'] == im_name]
    df_out = df_in
    umap_params = {'n_neighbors': 15, 'min_dist': 0.2, 'n_components': 2, 'metric': 'euclidean', 'spread': 1.0, 'random_state': 42}
    df_out, clustering_analysis = perform_clustering_analysis(df_out, top_output_dir, 'combined_files', subsample=None, umap_params=umap_params)
    # umap_params = {'n_neighbors': 200, 'min_dist': 0.2, 'n_components': 2, 'metric': 'euclidean', 'spread': 1.0, 'random_state': 42}
    # find the min cluster number
    min_cluster = df_out['cluster_umap'].min()
    # make the min cluster number 1
    df_out['cluster_umap'] = df_out['cluster_umap'] - min_cluster + 1

    # df_out['cluster_umap'] = df_out['cluster_umap'] + 1
    # df_out.to_csv(os.path.join(top_output_dir, 'analysis', 'combined_files.csv'))

    # df_in = pd.read_csv(os.path.join(top_output_dir, 'analysis', 'combined_files.csv'))
    # # only get the rows from one file
    # df_out = df_in[df_in['filename'] == im_name]
    plot_umap_from_df(df_out, color_by='cluster_umap')

    # umap_params = {'n_neighbors': 20, 'min_dist': 0.1, 'n_components': 2, 'metric': 'euclidean', 'spread': 1.0, 'random_state': 42}
    # pt_2, cluster_2 = perform_clustering_analysis(df_out[df_out['cluster_umap']==3], top_output_dir, 'combined_files',
    #                                      subsample=None, umap_params=umap_params)
    # plot_umap_from_df(pt_2, color_by='r_intensity_coords_ch1_mean')
    # plot_umap_from_df(pt_2, color_by='concentration', highlight_value=5000)
    # save df_out to a csv in the analysis folder
    # run_analysis(top_output_dir, im_name, ch)
    # for testing:
    # # Plot UMAP colored by concentration
    # plot_umap_from_df(df_out, color_by='concentration')
    # plot_umap_from_df(df_out, color_by='rn_speed_mean')
    # plot_umap_from_df(df_out, color_by='concentration', highlight_value=5000)
    # plot_umap_from_df(df_out, color_by='cluster_pca')
    #
    # # Plot UMAP colored by time
    # plot_umap_from_df(df_out, color_by='time')
    # plot_umap_from_df(df_out, color_by='time', highlight_value=1)

    # Plot UMAP colored by filename

    # im_name = file_list[-1]
    # df_out['filename'] = df_in[df_in.columns[0]]
    test_savepath = os.path.join(top_output_dir, 'analysis', 'test.png')
    save_path = os.path.join(clustering_analysis.output_dir, f"{clustering_analysis.date_time}_{im_name}_umap.png")
    plot_umap_from_df(df_out, color_by='filename', highlight_value=im_name, full_save_path=save_path, s=1, vmin=0, vmax=4)
    # image = imread(test_savepath)
    region_props = load_region_props(top_output_dir, im_name)
    new_im = process_images(region_props, df_out)
    viewer = display_images(region_props, new_im)
    scaled_data = pd.DataFrame(clustering_analysis.scaled_data)
    scaled_data.columns = clustering_analysis.df.columns
    # viewer.add_image(image)
    model_worth(scaled_data, df_out, 'cluster_umap', clustering_analysis.date_time, k_best_features=10, save_path=clustering_analysis.output_dir)
