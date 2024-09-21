# This file should be run under sugar env

import subprocess

# folder_path='replica_office0_id79/ndt0.1/'
# data='/mnt/ws-frb/projects/gaussian_splatting/data_for_original_gs/replica/office0'
# frame_ids = '79' # 1 shots replica office0

folder_path='replica_room0_id79/ndt0.1/'
data='/mnt/ws-frb/projects/gaussian_splatting/data_for_original_gs/replica/room0'
frame_ids = '79' # 1 shots replica office0

with open('./name_room0.txt', 'r') as file:
    file_names = file.readlines()

# Remove newline characters from the file names
file_names = [name.strip() for name in file_names if name[0]!='#']

# Iterate over each file name and run the command
for name in file_names:
    command = f"python eval_pointcloud.py -s {data} -m /mnt/ws-frb/users/frank/frank/gaussian_splatting/{name} --single_frame_id {frame_ids}"
    print(command)
    subprocess.run(command, shell=True)