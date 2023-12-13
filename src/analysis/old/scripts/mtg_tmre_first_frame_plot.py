import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

top_dir = r"D:\test_files\20230713-AELxZL-coated_DENSPM_wt_ko_A549"
csv_name = r"20230713-AELxZL-coated_DENSPM_wt_ko_A549_first_frame.csv"
csv_path = os.path.join(top_dir, csv_name)

df = pd.read_csv(csv_path)

# Group by concentration, condition, and file_num to get mean values for each group
file_means = df.groupby(['concentration', 'condition', 'file_num']).mean().reset_index()

# Group the file_means by concentration and condition to get the mean and std of the means across file_nums
grouped_means = file_means.groupby(['concentration', 'condition']).agg(
    tmrm_mtg_mean_of_means=('tmrm_mtg_mean_ratio', 'mean'),
    tmrm_mtg_std_of_means=('tmrm_mtg_mean_ratio', 'std')
).reset_index()

# Extracting values
concentrations = grouped_means['concentration'].unique()
wt_means = grouped_means[grouped_means['condition'] == 'wt']['tmrm_mtg_mean_of_means'].values
ko_means = grouped_means[grouped_means['condition'] == 'ko']['tmrm_mtg_mean_of_means'].values
wt_err = grouped_means[grouped_means['condition'] == 'wt']['tmrm_mtg_std_of_means'].values
ko_err = grouped_means[grouped_means['condition'] == 'ko']['tmrm_mtg_std_of_means'].values

# Plotting
plt.figure(figsize=(12, 7))

# Bar widths and offsets
width = 0.35
ind = np.arange(len(concentrations))

# Bars
plt.bar(ind - width/2, wt_means, yerr=wt_err, width=width, label='wt', alpha=0.7)
plt.bar(ind + width/2, ko_means, yerr=ko_err, width=width, label='ko', alpha=0.7)

# Overlay individual file_num means
sns.stripplot(
    x='concentration',
    y='tmrm_mtg_mean_ratio',
    hue='condition',
    data=file_means,
    dodge=True,
    jitter=True,
    marker='o',
    alpha=0.6,
    color='black',  # Setting color to black for visibility
    edgecolor='white',
    linewidth=0.5,
    hue_order=['wt', 'ko']
)

# Adjusting x-ticks
plt.xticks(ind, concentrations)
plt.xlabel('Concentration')
plt.ylabel('Mean of Means tmrm_mtg_mean_ratio')
plt.title('Means of Means TMRM MTG Mean Ratio between KO and WT across Concentrations with File Means Overlay')
plt.legend(title='Condition')

plt.tight_layout()
plt.savefig(os.path.join(top_dir, 'means_of_means_tmrmtmgmeanratio_wt_ko.png'), dpi=300)
plt.close()

file_means.to_csv(os.path.join(top_dir, 'means_of_means_tmrmtmgmeanratio_wt_ko.csv'))

# Pivot the DataFrame
pivot_df = file_means.pivot_table(index='file_num',
                                 columns=['condition', 'concentration'],
                                 values='tmrm_mtg_mean_ratio')

# Rename the columns for clarity
pivot_df.columns = ['_'.join(map(str, col)).strip() for col in pivot_df.columns.values]

# Save to CSV
pivot_df.to_csv(os.path.join(top_dir, 'means_of_means_tmrmtmgmeanratio_wt_ko_pivot.csv'))