#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import sse, psnr, masked_psnr
from argparse import ArgumentParser

import matplotlib.pyplot as plt

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def readMasks(masks_dir):
    masks = []
    image_names = []
    for fname in sorted(os.listdir(masks_dir)):
        mask = Image.open(masks_dir / fname)
        masks.append(tf.to_tensor(mask).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return masks, image_names

def evaluate(model_paths, mask_type, viz=False):

    img_eval_dict = {}
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}

    for scene_dir in model_paths:
        print("Scene:", scene_dir)
        
        img_eval_dict[scene_dir] = {}
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        suffix=''

        if mask_type=='seen_mask':
            test_dir = Path(scene_dir) / "test_seen_masked"
            suffix='_seen_masked'
        elif mask_type=='eroded_seen_mask':
            test_dir = Path(scene_dir) / "eroded_seen_mask" / "test_seen_masked"
            suffix='_eroded_seen_masked'
        elif mask_type=='mask':
            test_dir = Path(scene_dir) / "test_masked"
            suffix='_masked'
        else:
            test_dir = Path(scene_dir) / "test"

        for method in os.listdir(test_dir):
            if method!='ours_'+str(args.iterations):
                continue
            print("Method:", method)
            
            if viz:
                viz_path = scene_dir+f'/{method}_viz_{mask_type}/'
                os.makedirs(viz_path, exist_ok=True)

            img_eval_dict[scene_dir][method] = {}

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir/ "gt"
            renders_dir = method_dir / "renders"
            renders, gts, image_names = readImages(renders_dir, gt_dir)
            valid_image_names = []

            if mask_type=='seen_mask' or mask_type=='mask' or mask_type=='eroded_seen_mask':
                masks_dir = method_dir / "masks"
                masks, mask_image_names = readMasks(masks_dir)

            if mask_type=='seen_mask' or mask_type=='mask' or mask_type=='eroded_seen_mask':
                img_psnrs = []
                img_ssims = []
                img_lpipss = []
                psnrs = []
                sses = []
                num_valid_pixels = []
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    # masked_psnr_ = masked_psnr(renders[idx], gts[idx], masks[idx])
                    # if masked_psnr_ is not None:
                    #     psnrs.append(masked_psnr_)
                    
                    sse_ = sse(renders[idx], gts[idx])
                    sses.append(sse_)
                    
                    img_psnr_ = psnr(renders[idx], gts[idx])
                    img_ssim_ = ssim(renders[idx], gts[idx])
                    img_lpips_ = lpips(renders[idx], gts[idx])
                                        
                    num_valid_pixels_ = masks[idx].squeeze().sum()        
                    
                    if num_valid_pixels_ > masks[idx].squeeze().numel()/4.:
                        img_psnrs.append(img_psnr_)
                        img_ssims.append(img_ssim_)
                        img_lpipss.append(img_lpips_)
                    else:
                        img_psnr_ = torch.tensor(-1)
                        img_ssim_ = torch.tensor(-1)
                        img_lpips_ = torch.tensor(-1)

                    if num_valid_pixels_>0:
                        valid_image_names.append(image_names[idx])
                        num_valid_pixels.append(num_valid_pixels_)
                        psnr_ = 20 * torch.log10(1.0 / torch.sqrt(sse_/num_valid_pixels_+1e-5))
                        psnrs.append(psnr_)
                    
                    if viz:
                        mse = (sse_/num_valid_pixels_).item()
                        mse = "{:.4f}".format(mse)
                        render_ = renders[idx].squeeze().permute(1,2,0).cpu().numpy()
                        gt_ = gts[idx].squeeze().permute(1,2,0).cpu().numpy()
                        mask_ = masks[idx].squeeze().permute(1,2,0).cpu().numpy()
                        err_img = (((renders[idx] - gts[idx])) ** 2).squeeze().permute((1,2,0))
                        
                        # mask_out_idx = (mask_[:,:,0]==0)
                        
                        # render_[:,:,0][mask_out_idx] = 0.62
                        # render_[:,:,1][mask_out_idx] = 0.13
                        # render_[:,:,2][mask_out_idx] = 0.94
                        
                        # gt_[:,:,0][mask_out_idx] = 0.62
                        # gt_[:,:,1][mask_out_idx] = 0.13
                        # gt_[:,:,2][mask_out_idx] = 0.94
                        
                        # err_img[:,:,0][mask_out_idx] = 0.62
                        # err_img[:,:,1][mask_out_idx] = 0.13
                        # err_img[:,:,2][mask_out_idx] = 0.94
                                                
                        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                        axs[0].imshow(render_)
                        axs[0].set_title(f'render\n mse: {mse}\n psnr:{img_psnr_.item()}\n ssim:{img_ssim_.item()}\n lpips:{img_lpips_.item()}')
                        axs[0].axis('off')
                        axs[1].imshow(gt_)
                        axs[1].set_title('gt')
                        axs[1].axis('off')
                        axs[2].imshow(err_img.cpu().detach().numpy())
                        axs[2].set_title('err_img')
                        axs[2].axis('off')
                        fig.tight_layout()
                        # plt.show()
                        plt.savefig(viz_path+image_names[idx])
                        plt.close()

                print("  individual PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))

                total_mse = torch.tensor(sses).sum() / torch.tensor(num_valid_pixels).sum()
                total_psnr = 20 * torch.log10(1.0 / torch.sqrt(total_mse))
                print("  total PSNR : {:>12.7f}".format(total_psnr, ".5"))
                print("")

                # full_dict[scene_dir][method].update({"PSNR": torch.tensor(psnrs).mean().item()})
                full_dict[scene_dir][method].update({"PSNR": total_psnr.item()})
                per_view_dict[scene_dir][method].update({"PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "Num Valid Pixels": {name: num_valid_pixels_ for num_valid_pixels_, name in zip(torch.tensor(num_valid_pixels).tolist(), image_names)}})
                
                avg_img_psnr = torch.tensor(img_psnrs).mean()
                avg_img_ssim = torch.tensor(img_ssims).mean()
                avg_img_lpips = torch.tensor(img_lpipss).mean()
                
                print("Image PSNR: ", avg_img_psnr.item())
                print("Image SSIM: ", avg_img_ssim.item())
                print("Image LPIPS: ", avg_img_lpips.item())
                
                img_eval_dict[scene_dir][method].update({"PSNR": avg_img_psnr.item(),
                                                         "SSIM": avg_img_ssim.item(),
                                                         "LPIPS": avg_img_lpips.item()})
            else:
                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/"+method+"_img_eval_results"+suffix+".json", 'w') as fp:
                json.dump(img_eval_dict[scene_dir], fp, indent=True)
                
            with open(scene_dir + "/"+method+"_results"+suffix+".json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/"+method+"_per_view"+suffix+".json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument("--iterations", type=int, default=2000)
    
    parser.add_argument('--no_mask', action="store_true")
    parser.add_argument('--mask', action="store_true")
    parser.add_argument('--seen_mask', action="store_true")
    parser.add_argument('--eroded_seen_mask', action="store_true")
    parser.add_argument('--viz', action="store_true")
    args = parser.parse_args()

    mask_types=[]
    if args.no_mask:
        mask_types.append("None")
    if args.mask:
        mask_types.append("mask")
    if args.eroded_seen_mask:
        mask_types.append("eroded_seen_mask")
    if args.seen_mask:
        mask_types.append("seen_mask")
    if len(mask_types)==0:
        print('Please specify the evaluation mode --[no_mask | mask | seen_mask]')

    for mask_type in mask_types:
        evaluate(args.model_paths, mask_type, args.viz)
