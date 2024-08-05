# **SAD-GS**: **S**hape-**A**ligned **D**epth-supervised **G**aussian **S**platting

<p align="center">
<strong>Pou-Chun Kung</strong>, <strong>Seth Isaacson</strong>, <strong>Ram Vasudevan</strong>, and <strong>Katherine A. Skinner</strong> <br>
{pckung, sethgi, ramv, kskin}@umich.edu
</p>

### [Paper](https://openaccess.thecvf.com/content/CVPR2024W/NRI/papers/Kung_SAD-GS_Shape-aligned_Depth-supervised_Gaussian_Splatting_CVPRW_2024_paper.pdf) | [Project Page](https://umautobots.github.io/sad_gs)


<div align="center">
<img src="./assets/replica_rendering_bev_full_width.png" width="100%" />
</div>
<p align="center">
Novel view synthesis results from Gaussian Splat using only a single RGBD frame for training.
</p>

**Abstract**: *This paper proposes SAD-GS, a depth-supervised Gaussian Splatting (GS) method that provides accurate 3D geometry reconstruction by introducing a shapealigned depth supervision strategy. Depth information is widely used in various GS applications, such as dynamic scene reconstruction, real-time simultaneous localization and mapping, and few-shot reconstruction. However, existing depth-supervised methods for GS all focus on the center and neglect the shape of Gaussians during training. This oversight can result in inaccurate surface geometry in the reconstruction and can harm downstream tasks like novel view synthesis, mesh reconstruction, and robot path planning. To address
this, this paper proposes a shape-aligned loss, which aims to produce a smooth and precise reconstruction by
adding extra constraints to the Gaussian shape. The proposed method is evaluated qualitatively and quantitatively on two publicly available datasets. The evaluation demonstrates that the proposed method provides state-of-the-art novel view rendering quality and mesh accuracy compared to existing depth-supervised GS methods.*

## Contact Information

For any questions about running the code, please open a GitHub issue and provide a detailed explanation of the problem including steps to reproduce, operating system details, and hardware. Please open issues with feature requests, we're happy to help you fit the code to your needs!

For research inquiries, please contact:

- Pou-Chun (Frank) Kung: pckung [at] umich [dot] edu

## Cloning the Repository
```shell
git clone https://github.com/umautobots/SAD-GS --recursive
```

## Prerequisites
Setup the gaussian_splatting conda environment:
```shell
conda env create --file environment.yml
conda activate gaussian_splatting
```
Please see [Gaussian Splatting repo](https://github.com/graphdeco-inria/gaussian-splatting) for more details.

## Quick Start

Download the example Replica dataset from [google drive](https://drive.google.com/drive/folders/1ksnH6eUH1i5UqCwC0_1lY5oYaJCie-Re?usp=sharing).

Please change `data` to the path of the Replica dataset and `output_path` to where you want to save the results in `demo_run_batch_replica.py` and `demo_eval_batch_replica`. You can also change the `cuda_device` to use the assigned gpu accordingly.

To run all the baselines and the proposed SAD-GS on the Replica dataset
```shell
python demo_run_batch_replica.py
```
This command will run all the baseline configurations and the proposed SAG-GS. The results will be saved in the `output_path` folder.

To run evaluation and visualize renders:
```shell
python demo_eval_batch_replica.py
```
This command will run novel view rendering using the trained Gaussian Splat. Then, evaluate the render with ground-truth images. The render evaluation (PSNR/LPIPS/SSIM) for full image and seen region in the paper can be found in `ours_2000_img_eval_results_masked.json` and `ours_2000_img_eval_results_seen_masked.json` files. You can also find the rendered images in the `test_masked` and `test_seen_masked` folders.

## BibTeX

This work has been accepted for publication in the IEEE Robotics and Automation Letters. Please cite as follows:

```
@InProceedings{Kung_2024_CVPR,
    author    = {Kung, Pou-Chun and Isaacson, Seth and Vasudevan, Ram and Skinner, Katherine A.},
    title     = {SAD-GS: Shape-aligned Depth-supervised Gaussian Splatting},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {2842-2851}
}
```

## License

<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/umautobots/SAD-GS">SAD-GS</a> by MCity at the University of Michigan is licensed under <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1"></a></p>

For inquiries about commercial licensing, please reach out to the authors.
