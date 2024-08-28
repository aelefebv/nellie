import os

from nellie.run import run_all_directories_parallel

top_dir = '/scratch3/prateek/isr_animate/20240215_CPPX115_ISR_Washout/a1768945-d4fb-459d-98a2-d926ef18fc5a/images'
substring = 'ch04'
output_dir = '/scratch3/austin/projects/isr_animate_2/20240215_CPPX115_ISR_Washout'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
nellie_needed_dir = os.path.join(output_dir, 'nellie_necessities')
if not os.path.exists(nellie_needed_dir):
    os.makedirs(nellie_needed_dir)
run_all_directories_parallel(top_dir, substring, output_dir)
