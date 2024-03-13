# dataset settings
dataset_type = 'CocoPSDataset'
data_root = 'data/CryoNuSeg/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    # dict(type='DefaultFormatBundlePS', collect_keys=['proposals', 'gt_bboxes', 'gt_bboxes_ids', 
    #                                                 'gt_bboxes_ignore', 'gt_labels', 'gt_is_novel']),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_bboxes_ids', 
    #                            'gt_labels', 'gt_masks', 'gt_is_novel'],
    #                      meta_keys=('filename', 'ori_filename', 'ori_shape',
    #                                 'img_shape', 'pad_shape', 'scale_factor', 'flip',
    #                                 'flip_direction', 'img_norm_cfg', 'ann_info')),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
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
        ann_file=data_root + 'train_1_8_annotation.json',
        ann_dir = "/data4/zhangye/noisyboundaries/data/CryoNuSeg/train/mask",
        img_prefix=data_root + 'train/patch',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'valid_annotation.json',
        ann_dir = "/data4/zhangye/noisyboundaries/data/CryoNuSeg/valid/mask",
        img_prefix=data_root + 'valid/patch',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test_annotation.json',
        ann_dir = "/data4/zhangye/noisyboundaries/data/CryoNuSeg/test/mask",
        img_prefix=data_root + 'test/patch',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
