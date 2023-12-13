import os
import pandas as pd

def fill_na_in_csv(output_dir, file_path, save=False):
    df = pd.read_csv(os.path.join(output_dir, file_path))
    df.loc[:, df.columns.str.contains('aspect_ratio')] = df.loc[:, df.columns.str.contains('aspect_ratio')].fillna(1)
    df.loc[:, df.columns.str.contains('circularity')] = df.loc[:, df.columns.str.contains('circularity')].fillna(1)
    rb_dist = df.filter(like='rb_distance_from').columns
    rn_dist = df.filter(like='rn_distance_from').columns
    for i in range(len(rn_dist)):
        df[rb_dist[i]] = df[rb_dist[i]].fillna(df[rn_dist[i]])
        df[rb_dist[i+len(rn_dist)]] = df[rb_dist[i + len(rn_dist)]].fillna(df[rn_dist[i]])
    rb_int = df.filter(like='rb_intensity').columns
    rn_int = df.filter(like='rn_intensity').columns
    for i in range(len(rn_int)):
        df[rb_int[i]] = df[rb_int[i]].fillna(df[rn_int[i]])
    rb_len = df.filter(like='rb_length').columns
    rb_width = df.filter(like='rb_width').columns
    rn_width = df.filter(like='rn_node_width').columns
    for i in range(len(rn_int)):
        df[rb_len[i]] = df[rb_len[i]].fillna(df[rn_width[i]])
        df[rb_width[i]] = df[rb_width[i]].fillna(df[rn_width[i]])
        df[rb_width[i+len(rn_width)]] = df[rb_width[i + len(rn_width)]].fillna(df[rn_width[i]])
    df.loc[:, df.columns.str.contains('std')] = df.loc[:, df.columns.str.contains('std')].fillna(0)
    df = df.fillna(0)
    if save:
        new_file_path = file_path.replace('.csv', '_filled.csv')
        full_save_path = os.path.join(output_dir, new_file_path)
        df.to_csv(full_save_path, index=False)
    return df


if __name__ == "__main__":
    output_dir = r'D:\test_files\nelly\20230406-AELxKL-dmr_lipid_droplets_mtDR\output\csv'
    file_path = 'summary_stats_regions-deskewed-2023-04-06_17-01-43_000_AELxKL-dmr_PERK-lipid_droplets_mtDR-5000-4h.ome-ch1.csv'
    path_out = fill_na_in_csv(output_dir, file_path)
