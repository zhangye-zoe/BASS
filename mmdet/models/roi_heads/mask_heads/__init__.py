# Copyright (c) OpenMMLab. All rights reserved.
from .coarse_mask_head import CoarseMaskHead
from .dynamic_mask_head import DynamicMaskHead
from .fcn_mask_head import FCNMaskHead
from .fcn_mask_head_ntm import FCNMaskHeadNTM
from .feature_relay_head import FeatureRelayHead
from .fused_semantic_head import FusedSemanticHead
from .global_context_head import GlobalContextHead
from .grid_head import GridHead
from .htc_mask_head import HTCMaskHead
from .mask_point_head import MaskPointHead
from .maskiou_head import MaskIoUHead
from .scnet_mask_head import SCNetMaskHead
from .scnet_semantic_head import SCNetSemanticHead
from .contrastive_head import PixelContrastiveHead
from .fcn_mask_head_clc import FCNMaskHeadCLC
# from ilib.model.contrastmask_head import mask_head, contrastive_head

__all__ = [
    'FCNMaskHead', 'HTCMaskHead', 'FusedSemanticHead', 'GridHead',
    'MaskIoUHead', 'CoarseMaskHead', 'MaskPointHead', 'SCNetMaskHead',
    'SCNetSemanticHead', 'GlobalContextHead', 'FeatureRelayHead',
    'DynamicMaskHead', 'FCNMaskHeadNTM'
]
