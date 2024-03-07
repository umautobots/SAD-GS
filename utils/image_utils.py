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

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def sse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).sum(1, keepdim=True)

def masked_psnr(img1, img2, mask): # Not rigorous, just for reference
    num_valid_pixels = mask.squeeze()[0].sum()
    if num_valid_pixels>0:
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).sum(1, keepdim=True) / num_valid_pixels
        if mse==0:
            mse+=1e-5 # TODO: This value can largely affect the psnr value
        return 20 * torch.log10(1.0 / torch.sqrt(mse))
    else:
        return None
