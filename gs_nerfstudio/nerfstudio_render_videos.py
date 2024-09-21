# This file should be run under nerfstudio env

import subprocess

# camera_path = '1shot_id79_camera_path'
# camera_path = 'tum_1shot_id0_camera_path'
# camera_path = 'tum_office_id1114_camera_path'

# camera_path = '1shot_id79_camera_path_surface'
# iter = '2000' #'3300'


# camera_path = 'iphone_0325_2_camera_path'
# iter = '4200'

camera_path = 'iphone_myface1_camera_path'
iter = '2200'

#./tum_office_name.txt
with open('./iphone_name.txt', 'r') as file:
    file_names = file.readlines()

# Remove newline characters from the file names
file_names = [name.strip() for name in file_names if name[0]!='#']

# Iterate over each file name and run the command
for name in file_names:
    if name=='':
        continue
    command = f"python ~/gs_nerfstudio/nerfstudio/nerfstudio/scripts/gaussian_splatting/render.py camera-path --model-path /mnt/ws-frb/users/frank/frank/gaussian_splatting/{name} --camera-path-filename ~/Downloads/{camera_path}.json --output-path ~/gs_nerfstudio/video/{name}.mp4 --load-iteration {iter}"
    subprocess.run(command, shell=True)
    
    # # extract the first image frame
    # command = f"ffmpeg -i ~/gs_nerfstudio/video/{name}.mp4 -vframes 1 -ss 00:00:00 -f image2 ~/gs_nerfstudio/video/{name}.jpg"
    # subprocess.run(command, shell=True)