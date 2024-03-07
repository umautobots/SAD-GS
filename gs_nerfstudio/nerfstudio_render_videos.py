# This file should be run under nerfstudio env

import subprocess

# camera_path = '1shot_id79_camera_path'
# camera_path = 'tum_1shot_id0_camera_path'
camera_path = 'tum_office_id1114_camera_path'
iter = '3300'
with open('./tum_office_name.txt', 'r') as file:
    file_names = file.readlines()

# Remove newline characters from the file names
file_names = [name.strip() for name in file_names]

# Iterate over each file name and run the command
for name in file_names:
    command = f"python ~/gs_nerfstudio/nerfstudio/nerfstudio/scripts/gaussian_splatting/render.py camera-path --model-path /mnt/ws-frb/users/frank/frank/gaussian_splatting/{name} --camera-path-filename ~/Downloads/{camera_path}.json --output-path ~/gs_nerfstudio/video/{name}.mp4 --load-iteration {iter}"
    subprocess.run(command, shell=True)