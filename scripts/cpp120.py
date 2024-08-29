import os

from nellie.run import run_all_directories_parallel

top_dir = '/scratch3/prateek/isr_animate/20240315_CPPX120_ISR_Washout/84a5de3a-23c1-4ce0-8797-f4a647407b89/images'
substring = 'ch04'
output_dir = '/scratch3/austin/projects/isr_animate_4/20240315_CPPX120_ISR_Washout'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
nellie_needed_dir = os.path.join(output_dir, 'nellie_necessities')
if not os.path.exists(nellie_needed_dir):
    os.makedirs(nellie_needed_dir)
run_all_directories_parallel(top_dir, substring, output_dir)
