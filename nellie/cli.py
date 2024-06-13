import argparse
import os
from nellie.run import run


def process_files(files, ch, num_t, output_dir):
    for file_num, tif_file in enumerate(files):
        print(f'Processing file {file_num + 1} of {len(files)}, channel {ch + 1} of 1')
        try:
            _ = run(tif_file, remove_edges=False, ch=ch, num_t=num_t, output_dirpath=output_dir)
        except:
            print(f'Failed to run {tif_file}')
            continue


def process_directory(directory, substring, output_dir, ch, num_t):
    all_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if substring in f and f.endswith('.tiff')])
    process_files(all_files, ch, num_t, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process TIF images within subdirectories containing a specific substring.')
    parser.add_argument('--directory', required=True, help='Subdirectory with TIF files')
    parser.add_argument('--substring', required=True, help='Substring to look for in filenames')
    parser.add_argument('--output_directory', default=None, help='Output directory. Default is parent + nellie_output.')
    parser.add_argument('--ch', type=int, default=0, help='Channel number to process')
    parser.add_argument('--num_t', type=int, default=1, help='Number of time points')
    args = parser.parse_args()

    process_directory(args.directory, args.substring, args.output_directory, args.ch, args.num_t)
