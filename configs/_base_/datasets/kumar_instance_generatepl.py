# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/Kumar/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(667, 400), (833, 500), (1000, 600), (1167, 700), (1333, 800), (1500, 900), (1667, 1000),
                   (1833, 1100), (2000, 1200)],
        flip=True, # False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_1_2_annotation.json',
        img_prefix=data_root + 'train/patch',
        ann_dir = "/data4/zhangye/noisyboundaries/data/Kumar/train/mask",
        inst_suffix = '.npy',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'valid_annotation.json',
        img_prefix=data_root + 'valid/patch',
        ann_dir = "/data4/zhangye/noisyboundaries/data/Kumar/valid/mask",
        inst_suffix = '.npy',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'un_train_1_2_annotation.json',
        img_prefix=data_root + 'train/patch',
        ann_dir = "/data4/zhangye/noisyboundaries/data/Kumar/train/mask",
        inst_suffix = '.npy',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
