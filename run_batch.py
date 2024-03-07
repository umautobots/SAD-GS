import subprocess
import sys

cuda_device='1'

iter='2000'
# iter='3300'

folder_path='replica_office0_id79/ndt0.1/'
# folder_path='tum_1shot_id0_0218/ndt0.1/'
# folder_path='tum_office_1shot_id1114/ndt0.1/'

data='/mnt/ws-frb/projects/gaussian_splatting/data_for_original_gs/replica/office0'
# data='/mnt/ws-frb/projects/gaussian_splatting/data/TUM_RGBD/rgbd_dataset_freiburg2_xyz'
# data='/mnt/ws-frb/projects/gaussian_splatting/data/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household'

# frame_ids = '0,20,32,44,79,94,122,158' # 8 shots
# frame_ids = '0,20,32,44,79' # 5 shots
# frame_ids = '0' # 1 shots
frame_ids = '79' # 1 shots replica office0
# frame_ids = '1114' # tum office

opacity_reset_interval='1000' # default: 3000

# Baselines

command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
            -m /mnt/ws-frb/users/frank/frank/gaussian_splatting/{folder_path}/only_c --single_frame_id {frame_ids}\
            --save_iterations 1 500 1000 2000 3000 --iterations {iter} --checkpoint_iterations {iter} --init_w_gaussian --voxel_size 0.1 --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval}"
subprocess.run(command, shell=True)

command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
            -m /mnt/ws-frb/users/frank/frank/gaussian_splatting/{folder_path}/only_ds --single_frame_id {frame_ids}\
            --save_iterations 1 500 1000 2000 3000 --iterations {iter} --checkpoint_iterations {iter} --init_w_gaussian --voxel_size 0.1 --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval} --DS 0.1 --no_color_loss"
subprocess.run(command, shell=True)

command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
            -m /mnt/ws-frb/users/frank/frank/gaussian_splatting/{folder_path}/0.1ds --single_frame_id {frame_ids}\
            --save_iterations 1 500 1000 2000 3000 --iterations {iter} --checkpoint_iterations {iter} --init_w_gaussian --voxel_size 0.1 --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval} --DS 0.1 "
subprocess.run(command, shell=True)

command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
            -m /mnt/ws-frb/users/frank/frank/gaussian_splatting/{folder_path}/dist --single_frame_id {frame_ids}\
            --save_iterations 1 500 1000 2000 3000 --iterations {iter} --checkpoint_iterations {iter} --init_w_gaussian --voxel_size 0.1 --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval} --dist"
subprocess.run(command, shell=True)

command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
            -m /mnt/ws-frb/users/frank/frank/gaussian_splatting/{folder_path}/reset_opa --single_frame_id {frame_ids}\
            --save_iterations 1 500 1000 2000 3000 --iterations {iter} --checkpoint_iterations {iter} --init_w_gaussian --voxel_size 0.1 --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval} --reset_opa_near"
subprocess.run(command, shell=True)

command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
            -m /mnt/ws-frb/users/frank/frank/gaussian_splatting/{folder_path}/reset_opa_fovmask --single_frame_id {frame_ids}\
            --save_iterations 1 500 1000 2000 3000 --iterations {iter} --checkpoint_iterations {iter} --init_w_gaussian --voxel_size 0.1 --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval} --reset_opa_near --fov_mask"
subprocess.run(command, shell=True)

command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
            -m /mnt/ws-frb/users/frank/frank/gaussian_splatting/{folder_path}/0.1ds_reset_opa_fovmask --single_frame_id {frame_ids}\
            --save_iterations 1 500 1000 2000 3000 --iterations {iter} --checkpoint_iterations {iter} --init_w_gaussian --voxel_size 0.1 --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval} --DS 0.1 --reset_opa_near --fov_mask"
subprocess.run(command, shell=True)

command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
            -m /mnt/ws-frb/users/frank/frank/gaussian_splatting/{folder_path}/reset_opa2 --single_frame_id {frame_ids}\
            --save_iterations 1 500 1000 2000 3000 --iterations {iter} --checkpoint_iterations {iter} --init_w_gaussian --voxel_size 0.1 --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval} --reset_opa_near --reset_opa_far"
subprocess.run(command, shell=True)

command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
            -m /mnt/ws-frb/users/frank/frank/gaussian_splatting/{folder_path}/reset_opa2_fovmask --single_frame_id {frame_ids}\
            --save_iterations 1 500 1000 2000 3000 --iterations {iter} --checkpoint_iterations {iter} --init_w_gaussian --voxel_size 0.1 --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval} --reset_opa_near --reset_opa_far --fov_mask"
subprocess.run(command, shell=True)


# Ours

# Baseline
command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
            -m /mnt/ws-frb/users/frank/frank/gaussian_splatting/{folder_path}/M0.01_voxacc1_cls_free_full_reset_opa --single_frame_id {frame_ids}\
            --save_iterations 1 500 1000 2000 3000 --iterations {iter} --checkpoint_iterations {iter} --init_w_gaussian --voxel_size 0.1 --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval} --cls_loss --full_reset_opa"
subprocess.run(command, shell=True)

# command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
#             -m /mnt/ws-frb/users/frank/frank/gaussian_splatting/{folder_path}/M0.01_voxacc1_cls2_dist --single_frame_id {frame_ids}\
#             --save_iterations 1 500 1000 2000 3000 --iterations {iter} --checkpoint_iterations {iter} --init_w_gaussian --voxel_size 0.1 --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval} --cls_loss --dist"
# subprocess.run(command, shell=True)

# command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
#             -m /mnt/ws-frb/users/frank/frank/gaussian_splatting/{folder_path}/M0.01_voxacc1_cls2_reset_opa2_fovmask --single_frame_id {frame_ids}\
#             --save_iterations 1 500 1000 2000 3000 --iterations {iter} --checkpoint_iterations {iter} --init_w_gaussian --voxel_size 0.1 --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval} --cls_loss --reset_opa_near --reset_opa_far --fov_mask"
# subprocess.run(command, shell=True)

# command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
#             -m /mnt/ws-frb/users/frank/frank/gaussian_splatting/{folder_path}/M0.01_voxacc1_cls2_reset_opa2_maxscale0.2_fovmask --single_frame_id {frame_ids}\
#             --save_iterations 1 500 1000 2000 3000 --iterations {iter} --checkpoint_iterations {iter} --init_w_gaussian --voxel_size 0.1 --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval} --cls_loss --reset_opa_near --reset_opa_far --scale_loss --fov_mask"
# subprocess.run(command, shell=True)

# command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
#             -m /mnt/ws-frb/users/frank/frank/gaussian_splatting/{folder_path}/M0.01_voxacc1_cls2_full_reset_opa_maxscale0.2 --single_frame_id {frame_ids}\
#             --save_iterations 1 500 1000 2000 3000 --iterations {iter} --checkpoint_iterations {iter} --init_w_gaussian --voxel_size 0.1 --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval} --cls_loss --full_reset_opa --scale_loss"
# subprocess.run(command, shell=True)

# command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
#             -m /mnt/ws-frb/users/frank/frank/gaussian_splatting/{folder_path}/M0.01_voxacc1_cls2_reset_opa_fovmask --single_frame_id {frame_ids}\
#             --save_iterations 1 500 1000 2000 3000 --iterations {iter} --checkpoint_iterations {iter} --init_w_gaussian --voxel_size 0.1 --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval} --cls_loss --reset_opa_near --fov_mask"
# subprocess.run(command, shell=True)

# command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
#             -m /mnt/ws-frb/users/frank/frank/gaussian_splatting/{folder_path}/M0.01_voxacc1_cls2_reset_opa_maxscale0.2_fovmask --single_frame_id {frame_ids}\
#             --save_iterations 1 500 1000 2000 3000 --iterations {iter} --checkpoint_iterations {iter} --init_w_gaussian --voxel_size 0.1 --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval} --cls_loss --reset_opa_near --scale_loss --fov_mask"
# subprocess.run(command, shell=True)

# # Testing
# iter='2500'
# command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
#             -m /mnt/ws-frb/users/frank/frank/gaussian_splatting/{folder_path}/M0.01_voxacc1_cls2_reset_opa2_maxscale0.2_fovmask_refine500_v2 --single_frame_id {frame_ids}\
#             --start_checkpoint /mnt/ws-frb/users/frank/frank/gaussian_splatting/{folder_path}/M0.01_voxacc1_cls2_reset_opa2_maxscale0.2_fovmask_refine500_v2/chkpnt2000.pth\
#             --save_iterations 1 500 1000 2000 3000 --iterations {iter} --checkpoint_iterations 2000 --init_w_gaussian --voxel_size 0.1 --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval} --opacity_reset_interval 2001 --cls_loss --reset_opa_near --reset_opa_far --scale_loss --fov_mask"
# subprocess.run(command, shell=True)