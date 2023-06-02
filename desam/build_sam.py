# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
from pathlib import Path
import urllib.request
import torch

from .modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
)


def build_sam_vit_h(
    checkpoint=None,
    new_mask_decoder=False,
    mask_decoder_depth=2,
    mask_decoder_mlp_dim=2048,
    mask_decoder_num_heads=8,
    prompt_embed_dim=256,
    iou_head_depth=3,
    iou_head_hidden_dim=256
    ):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        new_mask_decoder=new_mask_decoder,
        mask_decoder_depth=mask_decoder_depth,
        mask_decoder_mlp_dim=mask_decoder_mlp_dim,
        mask_decoder_num_heads=mask_decoder_num_heads,
        prompt_embed_dim=prompt_embed_dim,
        iou_head_depth=iou_head_depth,
        iou_head_hidden_dim=iou_head_hidden_dim
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(
    checkpoint=None,
    new_mask_decoder=False,
    mask_decoder_depth=2,
    mask_decoder_mlp_dim=2048,
    mask_decoder_num_heads=8,
    prompt_embed_dim=256,
    iou_head_depth=3,
    iou_head_hidden_dim=256
    ):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        new_mask_decoder=new_mask_decoder,
        mask_decoder_depth=mask_decoder_depth,
        mask_decoder_mlp_dim=mask_decoder_mlp_dim,
        mask_decoder_num_heads=mask_decoder_num_heads,
        prompt_embed_dim=prompt_embed_dim,
        iou_head_depth=iou_head_depth,
        iou_head_hidden_dim=iou_head_hidden_dim
    )


def build_sam_vit_b(
    checkpoint=None,
    new_mask_decoder=False,
    mask_decoder_depth=2,
    mask_decoder_mlp_dim=2048,
    mask_decoder_num_heads=8,
    prompt_embed_dim=256,
    iou_head_depth=3,
    iou_head_hidden_dim=256
    ):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        new_mask_decoder=new_mask_decoder,
        mask_decoder_depth=mask_decoder_depth,
        mask_decoder_mlp_dim=mask_decoder_mlp_dim,
        mask_decoder_num_heads=mask_decoder_num_heads,
        prompt_embed_dim=prompt_embed_dim,
        iou_head_depth=iou_head_depth,
        iou_head_hidden_dim=iou_head_hidden_dim
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    new_mask_decoder=False,
    mask_decoder_depth=2,
    mask_decoder_mlp_dim=2048,
    mask_decoder_num_heads=8,
    prompt_embed_dim=256,
    iou_head_depth=3,
    iou_head_hidden_dim=256
):
    
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
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
                depth=mask_decoder_depth,
                embedding_dim=prompt_embed_dim,
                mlp_dim=mask_decoder_mlp_dim,
                num_heads=mask_decoder_num_heads,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=iou_head_depth,
            iou_head_hidden_dim=iou_head_hidden_dim,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    checkpoint = Path(checkpoint)
    if checkpoint.name == "sam_vit_b_01ec64.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_b_01ec64.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == 'y':
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-B checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")
    elif checkpoint.name == "sam_vit_h_4b8939.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_h_4b8939.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == 'y':
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-H checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")
    elif checkpoint.name == "sam_vit_l_0b3195.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_l_0b3195.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == 'y':
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-L checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")

        
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
            
        if new_mask_decoder:
            model_dict = sam.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if (k in model_dict and 'mask_decoder' not in k)}
            model_dict.update(pretrained_dict)
            sam.load_state_dict(model_dict)
        else:
            sam.load_state_dict(state_dict, strict=False)
    return sam
