# This file should be run under nerfstudio env

import subprocess

folder_path='replica_office0_id79/ndt0.1/'
data='/mnt/ws-frb/projects/gaussian_splatting/data_for_original_gs/replica/office0'
iter = '2000'
frame_ids = '79' # 1 shots replica office0

# # Evaluation
# rendering_flags = '--skip_test' # --skip_train --skip_test --skip_mask
# evaluating_flags = '--mask --seen_mask'  # --no_mask --mask --seen_mask

# Render for Viz
rendering_flags = '--skip_train --skip_mask'
evaluating_flags = ''

with open('./name.txt', 'r') as file:
    file_names = file.readlines()

# Remove newline characters from the file names
file_names = [name.strip() for name in file_names]

# Iterate over each file name and run the command
for name in file_names:
    command = f"python render.py -s {data} -m /mnt/ws-frb/users/frank/frank/gaussian_splatting/{name} --iteration {iter} \
                    --single_frame_id {frame_ids} {rendering_flags}"
    subprocess.run(command, shell=True)

    command = f"python metrics.py -m /mnt/ws-frb/users/frank/frank/gaussian_splatting/{name} {evaluating_flags}"
    subprocess.run(command, shell=True)