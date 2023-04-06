import pandas as pd
import os

top_dir = r'D:\test_files\nelly\20230330-AELxZL-A549-TMRE_mtG\output\csv'
ch0_filename = 'summary_stats_regions-deskewed-2023-03-30_15-28-45_000_20230330-AELxZL-A549-TMRE_mtG-ctrl.ome-ch0.csv'
ch0_intensity = pd.read_csv(os.path.join(top_dir, ch0_filename))

ch1_filename = 'summary_stats_regions-deskewed-2023-03-30_15-28-45_000_20230330-AELxZL-A549-TMRE_mtG-ctrl.ome.csv'
ch1_intensity = pd.read_csv(os.path.join(top_dir, ch1_filename))

intensity_ratio = ch0_intensity['r_intensity_coords_ch1_mean'] / ch1_intensity['r_intensity_coords_mean']

test = 'summary_stats_regions-deskewed-2023-03-30_15-28-45_000_20230330-AELxZL-A549-TMRE_mtG-ctrl.ome-ch0.csv'
test_pd = pd.read_csv(os.path.join(top_dir, ch0_filename))

test_pd == ch0_intensity

a = test_pd.iloc[0]
b = ch0_intensity.iloc[0]