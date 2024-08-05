# This file should be run under nerfstudio env

import subprocess

cuda_device=0

data = '/mnt/ws-frb/projects/gaussian_splatting/data_for_original_gs/replica/office0'
output_path = '/mnt/ws-frb/users/frank/frank/gaussian_splatting/'
iter = '2000'
frame_ids = '79'

# Evaluation
rendering_flags = '--skip_train --skip_test'
evaluating_flags = '--mask --seen_mask' # --viz

# # Render for Viz
# rendering_flags = '--skip_mask' #'--skip_train --skip_mask'
# evaluating_flags = ''

with open('./demo_file_name/exp_name_replica_office0.txt', 'r') as file:
    file_names = file.readlines()

# Remove newline characters from the file names
file_names = [name.strip() for name in file_names if name[0]!='#']

# Iterate over each file name and run the command
for name in file_names:
    command = f"CUDA_VISIBLE_DEVICES={cuda_device} python render.py -s {data} -m {output_path}{name} --iteration {iter} \
                    --single_frame_id {frame_ids} {rendering_flags}"
    subprocess.run(command, shell=True)

    command = f"CUDA_VISIBLE_DEVICES={cuda_device} python metrics.py -m {output_path}{name} --iteration {iter} {evaluating_flags}"
    subprocess.run(command, shell=True)
