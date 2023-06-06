# Advanced Topics in 3D Computer Vision
Repository for the Project of Advanced Topics in 3D Computer Vision

# File Structure #

```
.
├── README.md
├── apps
│   ├── eval_3d.py
│   ├── train_dmc.py
│   └── train_normal_net.py
├── assets
│   ├── obj
│   ├── smplx
│   ├── teaser.jpg
│   └── tex
├── binvox (not necessary, needed for creating obj3d files)
├── checkpoints # model checkpoints
│   └── demo
├── configs
│   ├── dmc_demo.yaml
│   ├── dmc_train.yaml
│   └── normal_train.yaml
├── data # Your datasets
│   ├── Synthetic 
│   ├── MultiHuman # has single, single_occluded, ...
│   ├── multihuman_single_raw
│   └── demo
├── demo.sh
├── external
│   ├── EasyMocap-master
│   ├── Self-Correction-Human-Parsing
│   └── __init__.py
├── lib
│   ├── __init__.py
│   ├── __pycache__
│   ├── data
│   ├── ext_transform.py
│   ├── geometry.py
│   ├── loss_util.py
│   ├── mesh_util.py
│   ├── model
│   ├── net_util.py
│   ├── options.py
│   ├── sample_util.py
│   ├── sdf.py
│   ├── smpl_util.py
│   └── train_util.py
├── meetings # this is used if you 
│   └── week_3
├── render_dataset.sh
├── render_two.sh
├── results
│   └── dmc_demo
├── scripts
│   ├── cloudsend.sh
│   ├── convert_params.py
│   ├── download.py
│   └── vis.py
├── taichi_render_gpu
│   ├── README.md
│   ├── render_multi.py
│   ├── render_smpl.py
│   ├── setup.py
│   └── taichi_three
└── train.sh
```

# Running DeepMultiCap (Baseline) #

Make sure to include the dataset under `data`, namely `MultiHuman` and `multihuman_single_raw` (see file structure).\
Make sure to have pretrained checkpoints of *DeepMultiCap* downloaded in `checkpoints/demo/`.

1. Generate image, normal, masks and depth from object files:
```
cd taichi_render_gpu
python render_multi.py --data_root ../data/MultiHuman/single/obj --texture_root ../data/multihuman_single_raw/multihuman_single --save_path ../data/multihuman_single_inputs --num_angles 4
```

- You should now have `depth`, `img`, `mask`, `normal`, and `parameter` in your `data/multihuman_single_inputs` folder.
- These images should look like this:
<div style="text-align:center">
<img src="/assets/reproducing/reprojection/0.png" width="100" />
<img src="/assets/reproducing/reprojection/90.png" width="100" /> 
<img src="/assets/reproducing/reprojection/180.png" width="100" />
<img src="/assets/reproducing/reprojection/270.png" width="100" />
</div>

- These images do not contain any colors, because DeepMultiCap has a weird file structure and no documentation at all :(

2. Generate smpl global maps
```
python render_smpl.py --dataroot ../data/multihuman_single_inputs --obj_path ../data/MultiHuman/single/smplx --faces_path ../lib/data/smplx_multi.obj --yaw_list 0 90 180 270
```

- This should now generate a folder called `smpl_pos`


3. Copy the smplx folder of `data/MultiHuman/single/smplx` into `data/multihuman_single_inputs`. "Normally", these should include the estimated smpl models from another method.


4. Generate reconstructions and visualization
```
# go back to project root folder
python apps/eval_3d.py --config configs/multihuman_single.yaml --dataroot data/multihuman_single_inputs
```

- Now the reconstructions should be in `results/multihuman_single`
- The results look similar to:

<div style="text-align:center">
  <img src="/assets/reproducing/results/multihuman_single.jpg" width="200" />
</div>

5. Evaluation (Not implemented)


# Generating Input for Synthetic Data

## Generating Masks 

We use [this repository](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) for generating the masks.

1. Follow the installment steps from the github repository (difficult lel)
2. Make sure that the Synthetic data is under `data/Synthetic/<first_trial>`
3. Make sure that all images are in the same folder
```
cd external/Self-Correction-Human-Parsing
python process.py -src_img ../../data/Synthetic/first_trial -dst_img mhp_extension/data/synthetic_first_trial/global_pic
```

4. Create coco style annotations for the images you just copied
```
cd mhp_extension
python ./coco_style_annotation_creator/test_human2coco_format.py --dataset "synthetic_first_trial" --json_save_dir "./data/synthetic_first_trial/annotations" --test_img_dir "./data/synthetic_first_trial/global_pic"
```

5. Generate instance prediction for images

```
python finetune_net.py --num-gpus 1 --config-file configs/Misc/synthetic_first_trial.yaml --eval-only MODEL.WEIGHTS pretrain_model/detectron2_maskrcnn_cihp_finetune.pth TEST.AUG.ENABLED False DATALOADER.NUM_WORKERS 0
```

6. Crop images by prediction bounding boxes
   
```
python make_crop_and_mask_w_mask_nms.py --img_dir "./data/synthetic_first_trial/global_pic" --save_dir "./data/synthetic_first_trial" --img_list "./data/synthetic_first_trial/annotations/synthetic_first_trial.json" --det_res "./data/synthetic_first_trial/detectron_2_prediction/inference/instances_predictions.pth"
```

7. Generate txt files for images in `global pic` and `crop_pic`

```
python generate_txt_file.py --folder_path "data/synthetic_first_trial/global_pic" --txt_file_name "global_pic.txt"
python generate_txt_file.py --folder_path "data/synthetic_first_trial/crop_pic" --txt_file_name "crop_pic.txt"
```

8. Generate parsed images for cropped images, global images
```
cd ..
python ./mhp_extension/global_local_parsing/global_local_evaluate.py --data-dir "./mhp_extension/data/synthetic_first_trial" --split-name "crop_pic" --model-restore "./mhp_extension/pretrain_model/exp_schp_multi_cihp_local.pth" --log-dir "./mhp_extension/data/synthetic_first_trial" --save-results
python ./mhp_extension/global_local_parsing/global_local_evaluate.py --data-dir "./mhp_extension/data/synthetic_first_trial" --split-name "global_pic" --model-restore "./mhp_extension/pretrain_model/exp_schp_multi_cihp_global.pth" --log-dir "./mhp_extension/data/synthetic_first_trial" --save-results
```

9. Install `pip install joblib` if necessary

10. Fuse the inputs and get your results!
```
python mhp_extension/logits_fusion.py --test_json_path "./mhp_extension/data/synthetic_first_trial/crop.json" --global_output_dir "./mhp_extension/data/synthetic_first_trial/global_pic_parsing" --gt_output_dir "./mhp_extension/data/synthetic_first_trial/crop_pic_parsing"  --mask_output_dir "./mhp_extension/data/synthetic_first_trial/crop_mask" --save_dir "./mhp_extension/data/synthetic_first_trial/mhp_fusion_parsing"
```
- Results are now in `./mhp_extension/data/synthetic_first_trial/mhp_fusion_parsing/global_tag`
  
11. Copy the results into the data folder:
```
python process.py --depth_folder ../../data/Synthetic/first_trial/Depth --depth_target ../../data/Synthetic/first_trial/depth_npz --img_folder ../../data/Synthetic/first_trial --img_target ../../data/Synthetic/first_trial/img --normal_folder ../../data/Synthetic/first_trial/Normal --normal_target ../../data/Synthetic/first_trial/normal_post_process --mask_folder ./mhp_extension/data/synthetic_first_trial/mhp_fusion_parsing/global_tag --mask_target ../../data/Synthetic/first_trial/masks
```

## Uploading To Nextcloud Repository ##

Use the script provided in `scripts/cloudsend.sh` to upload data into the nextcloud repository.\
Make sure to make the script executable if you are on *nix machine.

Example:
```bash
# Upload the folder/file <data> to the nextcloud repository
 ./scripts/cloudsend.sh -p 'fcanys2333' '<data>' 'https://nextcloud.in.tum.de/index.php/s/RjdwM59XHAkWkMC'
```

# Installing EasyMocap #
Download SMPL models:
```
pip install gdown wget
# You might need to rename the HrNet weight file
python scripts/download.py
```

Prepare your Conda environment (if necessary):
```
conda create -n easymocap python=3.9 -y
conda activate easymocap
```

Install appropriate torch and torchvision version:
```
pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html`
pip install torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```


Install remaining requirements:
```
cd external/EasyMocap-master
python -m pip install -r requirements.txt
python3 -m pip install pyrender
python setup.py develop
```

Convert Dataset into easymocap format (extri and intri still bugged):
```
python scripts/convert_params.py -i data/Synthetic/first_trial/camera_info.json -o data/Synthetic/first_trial_easymocap -d data/Synthetic/first_trial -f 30
```

Extract the images from videos:
```
data=/path/to/data
python scripts/preprocess/extract_video.py ${data} --no2d
```

Create 2D keypoints:
```
python apps/preprocess/extract_keypoints.py ${data} --mode yolo-hrnet
```

Create 3D keypoints:
```
python3 apps/demo/mvmp.py ${data} --out ${data}/output --annot annots --cfg config/exp/mvmp1f.yml --undis --vis_det --vis_repro
```

Track 3D keypoints:
```
python3 apps/demo/auto_track.py ${data}/output ${data}/output-track --track3d
```

Fit SMPL model:
```
python3 apps/demo/smpl_from_keypoints.py ${data} --skel ${data}/output-track/keypoints3d --out ${data}/output-track/smpl --verbose --opts smooth_poses 1e1
```




# [DeepMultiCap: Performance Capture of Multiple Characters Using Sparse Multiview Cameras (ICCV 2021)](http://www.liuyebin.com/dmc/dmc.html)
[Yang Zheng*](https://y-zheng18.github.io/zy.github.io/), Ruizhi Shao*, [Yuxiang Zhang](https://zhangyux15.github.io/), [Tao Yu](http://ytrock.com/), [Zerong Zheng](http://zhengzerong.github.io/), [Qionghai Dai](http://media.au.tsinghua.edu.cn/english/team/qhdai.html), [Yebin Liu](http://www.liuyebin.com/).

[![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2105.00261)

This repository contains the official pytorch implementation of ”*DeepMultiCap: Performance Capture of Multiple Characters Using Sparse Multiview Cameras*“.

![Teaser Image](assets/teaser.jpg)

### News
* **[2021/9/26]** We added more scans to [MultiHuman dataset](https://github.com/y-zheng18/MultiHuman-Dataset). You can use MultiHuman to train/fine-tune our model or your own models!
* **[2021/9/18]** [MultiHuman dataset](https://github.com/y-zheng18/MultiHuman-Dataset) for evaluation purpose is available!

## Requirements
- [PyTorch](https://pytorch.org/)
- torchvision
- trimesh
- numpy
- matplotlib
- PIL
- skimage
- tqdm
- cv2
- json
- taichi==0.6.39 or 0.7.15
- taichi_three
- taichi_glsl==0.0.10

## Pretrained Model
We have provided pretrained model in [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:u:/g/personal/shaorz20_mails_tsinghua_edu_cn/EdVJtlpRplRHvGzQENV8ESQB4E_0ZY3B9l76XHuEowj1YA?e=MZqUxM)


## Training
To train DeepMultiCap model, please run the following code.
```
sh render_dataset.sh
sh train.sh
```

## Demo
Run the following code to inference on the MultiHuman dataset.
```
sh render_two.sh
sh demo.sh
```

## Citation
```
@inproceedings{zheng2021deepmulticap,
title={DeepMultiCap: Performance Capture of Multiple Characters Using Sparse Multiview Cameras},
author={Zheng, Yang and Shao, Ruizhi and Zhang, Yuxiang and Yu, Tao and Zheng, Zerong and Dai, Qionghai and Liu, Yebin},
booktitle={IEEE Conference on Computer Vision (ICCV 2021)},
year={2021},
}
```
