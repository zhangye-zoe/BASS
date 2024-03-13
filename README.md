# Boundary-aware Contrastive Learning for Semi-supervised Nuclei Instance Segmentation



# Installation

This code is based on mmdetection v2.18.
Please install the code according to the [mmdetection step](https://github.com/open-mmlab/mmdetection/blob/v2.18.0/docs/get_started.md) first.

### data preparation

```bash
BASS
├──data
|  ├──CryoNuSeg
|  |  ├──train
|  |  |  ├──mask
|  |  |  ├──patch
|  |  ├──valid
|  |  |  ├──mask
|  |  |  ├──patch
|  |  ├──test
|  |  |  ├──mask
|  |  |  ├──patch
|  |  ├──train_1_2_annotation.json
|  |  ├──train_1_4_annotation.json
|  |  ├──train_1_8_annotation.json
|  |  ├──un_train_1_2_annotation.json
|  |  ├──un_train_1_4_annotation.json
|  |  ├──un_train_1_8_annotation.json
|  |  ├──valid_1_2_annotation.json
|  |  ├──test_1_2_annotation.json
```

# Running scripts

## CryoNuseg
We take the experiment with the 1/2 labeled images for example.

First, to train the supervised model, run:
```bash
bash tools/dist_train.sh configs/noisyboundaries/cryonuseg/mask_rcnn_r50_fpn_1x_cityscapes_sup.py 1
```
Then, with the supervised model, generating pseudo labels for semi-supervised learning:
```bash
bash scripts/cryonuseg/extract_pl.sh 1 labels/rcity.pkl labels/cryonuseg_1_2_pl.json 
```
Final, perform semi-supervised learning:
```bash
bash tools/dist_train.sh configs/noisyboundaries/cryonuseg/mask_rcnn_r50_fpn_1x_coco_pl_clc.py 1
```

