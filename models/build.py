# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer_ditask import SwinTransformerDiTASK
from .swin_transformer import SwinTransformer
from .swin_mtl import MultiTaskSwin
from .vision_transformer import VisionTransformer
from .vision_transformer_ditask import VisionTransformerDiTASK
from .clip_mtl import MultiTaskCLIP
# from .pvt import PyramidVisionTransformer, pvt_tiny
from .pvt_ditask import PyramidVisionTransformerMTL, pvt_small_mtl
from .pvt_mtl import MultiTaskPVT
from .pvt_v2 import PyramidVisionTransformerV2, pvt_v2_b2, pvt_v2_b5
import timm


def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    if model_type == 'swin':
        if config.MODEL.DITASK.ENABLED:
            model = SwinTransformerDiTASK(img_size=config.DATA.IMG_SIZE,
                                          patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                          in_chans=config.MODEL.SWIN.IN_CHANS,
                                          num_classes=config.MODEL.NUM_CLASSES,
                                          embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                          depths=config.MODEL.SWIN.DEPTHS,
                                          num_heads=config.MODEL.SWIN.NUM_HEADS,
                                          window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                          mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                          qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                          qk_scale=config.MODEL.SWIN.QK_SCALE,
                                          drop_rate=config.MODEL.DROP_RATE,
                                          drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                          ape=config.MODEL.SWIN.APE,
                                          norm_layer=layernorm,
                                          patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                          use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                          fused_window_process=config.FUSED_WINDOW_PROCESS,
                                          tasks=config.TASKS,
                                          DITASK=config.MODEL.DITASK)
        else:
            model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                    patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                    in_chans=config.MODEL.SWIN.IN_CHANS,
                                    num_classes=config.MODEL.NUM_CLASSES,
                                    embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                    depths=config.MODEL.SWIN.DEPTHS,
                                    num_heads=config.MODEL.SWIN.NUM_HEADS,
                                    window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                    mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                    qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                    qk_scale=config.MODEL.SWIN.QK_SCALE,
                                    drop_rate=config.MODEL.DROP_RATE,
                                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                    ape=config.MODEL.SWIN.APE,
                                    norm_layer=layernorm,
                                    patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                    use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                    fused_window_process=config.FUSED_WINDOW_PROCESS)
    
    elif model_type == "vit":
        if config.MODEL.DITASK.ENABLED:
            model = VisionTransformerDiTASK(
                img_size=config.DATA.IMG_SIZE,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depth=sum(config.MODEL.SWIN.DEPTHS),
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            num_heads=config.MODEL.SWIN.NUM_HEADS[0],
            qkv_bias=config.MODEL.CLIP.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=layernorm,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            pre_norm=False,
            tasks=config.TASKS,
            DITASK=config.MODEL.DITASK
            )
        else:
            model = VisionTransformer(
            img_size=config.DATA.IMG_SIZE,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.CLIP.EMBED_DIM,
            depth=config.MODEL.CLIP.DEPTH,
            patch_size=config.MODEL.CLIP.PATCH_SIZE,
            num_heads=config.MODEL.CLIP.NUM_HEADS,
            qkv_bias=config.MODEL.CLIP.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=layernorm,
            in_chans=config.MODEL.CLIP.IN_CHANS,
            pre_norm=config.MODEL.CLIP.PRE_NORM
        )
    
    elif model_type == "pvt":
        model = timm.create_model(
                config.MODEL.NAME,
                num_classes=config.MODEL.NUM_CLASSES,
                img_size=config.DATA.IMG_SIZE,
                tasks=config.TASKS,
                DITASK=config.MODEL.DITASK
        )

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model


def build_mtl_model(backbone, config):
    if config.MODEL.TYPE == "swin":
        model = MultiTaskSwin(backbone, config)
    elif config.MODEL.TYPE == "vit":
        model = MultiTaskCLIP(backbone, config)
    elif config.MODEL.TYPE == "pvt":
        model = MultiTaskPVT(backbone, config)
    return model