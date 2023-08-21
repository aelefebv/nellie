import os
import pandas as pd


top_dir = r"D:\test_files\20230713-AELxZL-coated_DENSPM_wt_ko_A549"
csv_name = "20230803-105514-ch_0-track_stats.csv"
csv_path = os.path.join(top_dir, csv_name)

df = pd.read_csv(csv_path)
filenames = df['file_name']
unique_names = list(set(filenames))

# find either wt or ko in filename
mutant = []
concentration = []
roi_num = []
for filename in filenames:
    if 'wt' in filename.lower():
        mutant.append('wt')
        concentration.append(filename.split('wt_')[1].split('_')[0])
    elif 'ko' in filename.lower():
        mutant.append('ko')
        concentration.append(filename.split('ko_')[1].split('_')[0])
    else:
        print(f'Could not find wt or ko in {filename}')
        mutant.append('unknown')
        concentration.append('unknown')
    # append the roi number based on unique name idx
    roi_num.append(unique_names.index(filename))

# add the columns to the dataframe
df['mutant'] = mutant
df['concentration'] = concentration
df['roi_num'] = roi_num

# save the dataframe
new_csv_name = csv_name.split('.')[0] + '_mutant_concentration_roi.csv'
new_csv_path = os.path.join(top_dir, new_csv_name)
df.to_csv(new_csv_path, index=False)
