import pandas as pd
import os

top_dir = r'D:\test_files\nelly\20230330-AELxZL-A549-TMRE_mtG\output\csv'
files = os.listdir(top_dir)
csv_files = [f for f in files if f.endswith('.csv') and 'summary_stats_regions' in f]

import numpy as np

intensity_ratios = {}
for csv_file in csv_files:
    frame_ratios = []
    if '-ch0.csv' in csv_file:
        ch0_intensity = pd.read_csv(os.path.join(top_dir, csv_file))
    else:
        ch1_intensity = pd.read_csv(os.path.join(top_dir, csv_file))
        num_frames = ch1_intensity['frame_number'].max()
        for frame in range(num_frames):
            ch0_frame = ch0_intensity[ch0_intensity['frame_number'] == frame]
            ch1_frame = ch1_intensity[ch1_intensity['frame_number'] == frame]
            frame_ratio = ch0_frame['r_intensity_coords_ch1_mean'].values / ch1_frame['r_intensity_coords_mean'].values
            # append the mean of each frame to the list of frame ratios
            frame_ratios.append(np.median(frame_ratio))
        intensity_ratios[csv_file] = frame_ratios
        print(f"Intensity ratio for {csv_file}: {frame_ratios}")

# for each intensity ratio, get the mean of the frame ratios
mean_intensity_ratios = {}
for key, value in intensity_ratios.items():
    mean_intensity_ratios[key] = np.mean(value)
    print(f"Mean intensity ratio for {key}: {np.mean(value)}")

# group each intensity ratio by if it has OE, KO, or Ctrl in the title, and whether or not it has FCCP in the title
OE = []
KO = []
Ctrl = []
OE_FCCP = []
KO_FCCP = []
Ctrl_FCCP = []
for key, value in mean_intensity_ratios.items():
    if 'OE' in key:
        if 'FCCP' in key:
            OE_FCCP.append(value)
        else:
            OE.append(value)
    elif 'KO' in key:
        if 'FCCP' in key:
            KO_FCCP.append(value)
        else:
            KO.append(value)
    elif 'Ctrl' in key or 'ctrl' in key:
        if 'FCCP' in key:
            Ctrl_FCCP.append(value)
        else:
            Ctrl.append(value)

# make matplotlib plots to compare each group
import matplotlib.pyplot as plt


plt.boxplot([OE, KO, Ctrl, OE_FCCP, KO_FCCP, Ctrl_FCCP])
plt.xticks([1, 2, 3, 4, 5, 6], ['OE', 'KO', 'Ctrl', 'OE_FCCP', 'KO_FCCP', 'Ctrl_FCCP'])
plt.ylabel('Intensity Ratio')
plt.title('Intensity Ratio Comparison')
plt.show()

# export the mean_intensity_ratios to a csv file
import csv

import re
# create a dataframe from the mean intensity ratios
ratio_list = []
for key, value in mean_intensity_ratios.items():
    ratio_list.append([key, value])
# make ratio_list into a df
df = pd.DataFrame(ratio_list, columns=['filename', 'mean_intensity_ratio'])

# df.index.name = 'filename'

# add a column based on the group's name
groups = df['filename'].str.extract(r'(OE|KO|Ctrl|ctrl)-?(FCCP)?', flags=re.IGNORECASE)
test = groups[0] + '_' + groups[1].fillna('')
for i in range(len(test)):
    test[i] = test[i].lower()
df['group'] = test

# make matplotlib plots to compare each group
fig, ax = plt.subplots()
boxplot = df.boxplot('mean_intensity_ratio', by='group', ax=ax)
ax.set_xlabel('Group')
ax.set_ylabel('Intensity Ratio')
ax.set_title('Intensity Ratio Comparison')
plt.show()

# export the mean_intensity_ratios to a csv file
df.to_csv(os.path.join(top_dir, 'mean_intensity_ratios_all.csv'))
# reshape the dataframe to long format
melted_df = df.reset_index().melt(id_vars=['filename', 'group'], value_vars=['mean_intensity_ratio'], var_name='measurement')

# reorder the columns
melted_df = melted_df[['group', 'measurement', 'filename', 'value']]

# export the dataframe to a csv file
melted_df.to_csv('mean_intensity_ratios_long.csv', index=False)
df_pivot = df.pivot_table(index='group', values='mean_intensity_ratio', aggfunc='mean')

# save the pivoted dataframe to a CSV file
df_pivot.to_csv(os.path.join(top_dir, 'mean_intensity_ratios_prism.csv'), header=['Intensity Ratio'])
