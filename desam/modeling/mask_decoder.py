# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type, Union, Sequence
from .common import LayerNorm2d
import numpy as np
from monai.networks.blocks.dynunet_block import get_conv_layer, get_act_layer, get_norm_layer


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
    
class UnetResSEBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str] = "instance",
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims, 
            out_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            dropout=dropout, 
            conv_only=True
        )
        self.conv3 = get_conv_layer(
            spatial_dims, 
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=stride, 
            dropout=dropout, 
            conv_only=True
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True

        self.se = SELayer(channel=out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)
        if self.downsample:
            residual = self.conv3(residual)
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out
    

class PreUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str] = 'instance',
        num_layer: int = 1,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
        """

        super().__init__()

        self.input_channels = in_channels
        self.output_channels = out_channels
        
        self.block_init = UnetResSEBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        
        self.residual_block = nn.ModuleList(
            [
                nn.Sequential(
                    get_conv_layer(
                        spatial_dims=spatial_dims,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=upsample_kernel_size,
                        stride=upsample_kernel_size,
                        conv_only=True,
                        is_transposed=True,
                    ),
                    UnetResSEBlock(
                        spatial_dims=spatial_dims,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        norm_name=norm_name,
                    ),
                )
                for i in range(num_layer)
            ]
        )
        
    def forward(self, x):
        x = self.block_init(x)
        for blk in self.residual_block:
            x = blk(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str] = 'instance',
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.input_channels = in_channels
        self.output_channels = out_channels

        self.transp_conv = get_conv_layer(
            spatial_dims,
            self.input_channels,
            self.output_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        self.res_block = UnetResSEBlock(
            spatial_dims,
            self.output_channels + self.output_channels,
            self.output_channels,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        inp = self.transp_conv(inp)
        out = torch.cat((inp, skip), dim=1)
        out = self.res_block(out)
        return out
    

class PIMMDecoder(nn.Module):
    def __init__(
        self,
        endoder_transformer_dim: int = 1280,
        upsample_transformer_dim: int = 256,
        sam_features_length: int = 3,
        do_deep_supervision: bool = False,
    ) -> None:
        super().__init__()        
        self.sam_features_length = sam_features_length
        self.do_deep_supervision = do_deep_supervision
        
        # mask branch
        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.seg_outputs = []
        
        self.encoder_embed_size = [int(upsample_transformer_dim // 2 ** i) for i in range(sam_features_length)]
        
        for d in range(self.sam_features_length):
            in_channels = endoder_transformer_dim
            out_channels = self.encoder_embed_size[d]
            upsample_kernel_size = 2
            spatial_dims = 2
            self.conv_blocks_context.append(
                PreUpBlock(
                    spatial_dims=spatial_dims, 
                    in_channels=in_channels, out_channels=out_channels,  
                    upsample_kernel_size=upsample_kernel_size,
                    num_layer=d
                )
            )
        
        for d in range(self.sam_features_length-1):
            in_channels = self.encoder_embed_size[d]
            out_channels = self.encoder_embed_size[d+1]
            upsample_kernel_size = 2
            self.conv_blocks_localization.append(
                UpBlock(
                    spatial_dims=spatial_dims, 
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    upsample_kernel_size=upsample_kernel_size
                )
            )
            
        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(
                nn.Conv2d(
                    self.conv_blocks_localization[ds].output_channels, 1,
                    1, 1, 0, 1, 1, False
                )
            )
            
        # fusion mask embeddings
        self.mask_embedding_fusion = UnetResSEBlock(
            spatial_dims=spatial_dims,
            in_channels=self.encoder_embed_size[0] + self.encoder_embed_size[0],
            out_channels=self.encoder_embed_size[0],
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )
            
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)

    def forward(
        self,
        mask_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        # encoder
        skips = []
        seg_outputs = []
        
        for d in range(len(self.conv_blocks_context)):
            embed = self.conv_blocks_context[d](image_embeddings[-(d + 1)])
            if d == 0:
                embed = torch.cat((mask_embeddings, embed), dim=1)
                embed = self.mask_embedding_fusion(embed)
            skips.append(embed)
            
        # decoder
        for u in range(len(self.conv_blocks_localization)):
            if u == 0:
                enc_x = skips[0]
                dec_x = skips[1]
            else:
                dec_x = skips[u + 1]
            enc_x = self.conv_blocks_localization[u](enc_x, dec_x)
            seg_outputs.append(self.seg_outputs[u](enc_x))
        
        # Prepare output
        if self.do_deep_supervision:
            return seg_outputs[::-1]
        else:
            return seg_outputs[-1]
        
        
class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        endoder_transformer_dim: int = 1280,
        upsample_transformer_dim: int = 256, 
        sam_features_length: int = 3,
        do_deep_supervision: bool = False,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )
        
        # mask branch
        self.pimm = PIMMDecoder(
            endoder_transformer_dim=endoder_transformer_dim,
            upsample_transformer_dim=upsample_transformer_dim,
            sam_features_length=sam_features_length,
            do_deep_supervision=do_deep_supervision,
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        
        mask_embedding, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings[-1],
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        
        masks = self.pimm(mask_embedding, image_embeddings[:-1])

        # Select the correct mask or masks for output
        if multimask_output:
            raise
        
        mask_slice = slice(0, 1)
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        src = src.transpose(1, 2).view(b, c, h, w)

        # Generate mask quality predictions
        iou_token_out = hs[:, 0, :]        
        iou_pred = self.iou_prediction_head(iou_token_out)

        return src, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
