import os

from nellie.run import run_all_directories_parallel

import datetime

dt = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
top_dir = '/Users/austin/test_files/nellie_cpp_sg_test'
substring = 'ch02'
output_dir = f'/Users/austin/test_files/out_sg_test'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
nellie_needed_dir = os.path.join(output_dir, 'nellie_necessities')
if not os.path.exists(nellie_needed_dir):
    os.makedirs(nellie_needed_dir)
run_all_directories_parallel(top_dir, substring, output_dir, mp=False)
