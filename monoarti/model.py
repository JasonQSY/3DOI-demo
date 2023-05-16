from functools import partial
import torch

from .transformer import INTR
from .sam_transformer import SamTransformer
from .sam import ImageEncoderViT, MaskDecoder, PromptEncoder, TwoWayTransformer

def build_demo_model():
    # model = INTR(
    #     backbone_name='resnet50',
    #     image_size=[768, 1024],
    #     num_queries=15,
    #     freeze_backbone=False,
    #     transformer_hidden_dim=256,
    #     transformer_dropout=0,
    #     transformer_nhead=8,
    #     transformer_dim_feedforward=2048,
    #     transformer_num_encoder_layers=6,
    #     transformer_num_decoder_layers=6,
    #     transformer_normalize_before=False,
    #     transformer_return_intermediate_dec=True,
    #     layers_movable=1,
    #     layers_rigid=1,
    #     layers_kinematic=1,
    #     layers_action=1,
    #     layers_axis=3,
    #     layers_affordance=3,
    #     depth_on=True,
    # )

    # sam_vit_b
    encoder_embed_dim=768
    encoder_depth=12
    encoder_num_heads=12
    encoder_global_attn_indexes=[2, 5, 8, 11]

    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    model = SamTransformer(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            properties_on=True,
        ),
        affordance_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            properties_on=False,
        ),
        depth_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            properties_on=False,
        ),
        transformer_hidden_dim=prompt_embed_dim,
        backbone_name='vit_b',
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    return model
