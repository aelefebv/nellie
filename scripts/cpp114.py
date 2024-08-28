import os

from nellie.run import run_all_directories_parallel

top_dir = '/scratch3/prateek/isr_animate/20231215_CPPX114_ISR_Washout/hs/61fd3e14-937f-4274-9f0f-06b678605ae7'
substring = 'ch04'
output_dir = '/scratch3/austin/projects/isr_animate_2/20231215_CPPX114_ISR_Washout'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
nellie_needed_dir = os.path.join(output_dir, 'nellie_necessities')
if not os.path.exists(nellie_needed_dir):
    os.makedirs(nellie_needed_dir)
run_all_directories_parallel(top_dir, substring, output_dir)
