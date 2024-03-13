
_base_ = [
    '../../_base_/models/mask_rcnn_r50_fpn.py',
    '../../_base_/datasets/cryonuseg_instance_pl.py',
    #'../../_base_/schedules/schedule_1x.py', 
    '../../_base_/default_runtime.py'
]

#dataset settings
dataset_type = 'CocoPSDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_root = 'data/CryoNuSeg/'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadPSAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),  
    dict(type='DefaultFormatBundlePS', collect_keys=['proposals', 'gt_bboxes', 'gt_bboxes_ids', 
                                                    'gt_bboxes_ignore', 'gt_labels', 'gt_is_novel']),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_bboxes_ids', 
                               'gt_labels', 'gt_masks', 'gt_is_novel'],
                         meta_keys=('filename', 'ori_filename', 'ori_shape',
                                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                                    'flip_direction', 'img_norm_cfg', 'ann_info')),
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
data=dict(
    train = dict(
        type='ConcatDataset',
        datasets=[
           dict(type='RepeatDataset',
                times=19,
                dataset = dict(type=dataset_type,
                    ann_file=data_root + 'train_1_2_annotation.json',
                    img_prefix=data_root + 'train/patch',
                    pipeline=train_pipeline)),
           dict(type=dataset_type,
                ann_file='labels/cryonuseg_1@20_pl.json',
                img_prefix=data_root + 'train/patch',
                pipeline=train_pipeline),
        ]),


    val=dict(
        type=dataset_type,
        ann_file=data_root + 'valid_annotation.json',
        img_prefix=data_root + 'valid/patch',
        pipeline=test_pipeline,
        base_set='nonvoc',
        novel_set='voc'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test_annotation.json',
        img_prefix=data_root + 'test/patch',
        pipeline=test_pipeline,
        base_set='nonvoc',
        novel_set='voc'),
)


# Models
model = dict(
    type='ContrastMask',
    roi_head=dict(
        type='StandardRoIHeadPS',
        bbox_head=dict(
            type='Shared2FCBBoxWithCAMHead',
            with_avg_pool=True,
        ),
        mask_head=dict(
            type="FCNMaskCamHead",
            in_channels=256,
            class_agnostic=True,
            contrastive_enable=True,
            contrastive_head=dict(
                type="ContrastiveHead",
                num_convs=8,
                num_projectfc=3,
                thred_u=0.1,
                scale_u=1.0,
                percent=0.3,
                fc_norm_cfg=dict(type='BN',)),
        ),
        contrastive_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=28, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
    )
)

# Default_runtime
checkpoint_config = dict(out_dir="./ckpt")
custom_imports = dict(
    imports=[
        'ilib.dataset.cocops', 'ilib.dataset.pipelines', 
        'ilib.utils.tensorboard_utils', 'ilib.utils.sample_utils',
        'ilib.model.contrastmask',
        'ilib.model.contrastmask_head.roi_head', 
        'ilib.model.contrastmask_head.mask_head',
        'ilib.model.contrastmask_head.bbox_head', 
        'ilib.model.contrastmask_head.contrastive_head'],
    allow_failed_imports=False
)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='SelfTensorboardLoggerHook', 
             log_dir="./tensorboards", 
             contr_upper_ep=4.0, 
             contr_start_ep=0.0, 
             init_value=0.25)
    ]
)

# Schedule 3x, same as the default setting of mmdetection
# lr_config = dict(
#     step=[8, 11]
# )
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# runner = dict(type='EpochBasedRunner', max_epochs=50)
# checkpoint_config = dict(by_epoch=True, interval=1)
# evaluation = dict(metric=['bbox', 'segm'], interval=1)

# model = dict(
#     roi_head=dict(
#         type='StandardRoIHeadNTM',
#         bbox_head=dict(
#             type='Shared2FCBBoxHead',
#             in_channels=256,
#             fc_out_channels=1024,
#             roi_feat_size=7,
#             num_classes=80,
#             bbox_coder=dict(
#                 type='DeltaXYWHBBoxCoder',
#                 target_means=[0., 0., 0., 0.],
#                 target_stds=[0.1, 0.1, 0.2, 0.2]),
#             reg_class_agnostic=False,
#             loss_cls=dict(
#                 type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#             loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
#         mask_head=dict(
#             type='FCNMaskHeadNTM',
#             num_convs=4,
#             in_channels=256,
#             conv_out_channels=256,
#             num_classes=80,
#             num_fcs=2,
#             downsample_factor=2,
#             fc_out_channels=1024,
#             loss_mask=dict(
#                 type='CrossEntropyLoss', use_mask=True, use_bpm=True, loss_weight=1.))
# ))

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    by_epoch=False,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[60000, 80000])
runner = dict(type='IterBasedRunner', max_iters=90000)
checkpoint_config = dict(by_epoch=False, interval=500)
evaluation = dict(metric=['bbox', 'segm'], interval=50)

