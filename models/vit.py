"""
Modified from:
    - https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
    - ViT paper: https://arxiv.org/abs/2010.11929
"""

from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn

from torchvision.ops.misc import ConvNormActivation

from .utils import program_conv_filters
from .activation import get_activation_class


__all__ = [
    "VisionTransformer",
    "vit_b_8",
    "vit_b_16",
    "vit_b_32",
    "vit_l_8",
    "vit_l_16",
    "vit_l_32",
]


class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU


class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float, nn_act: nn.Module):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn_act()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        nn_act: nn.Module,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout, nn_act)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (seq_length, batch_size, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        nn_act: nn.Module,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                nn_act,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        seq_len_2d: Tuple[int, int],
        size_min: int,
        size_max: int,
        in_channels: int,
        out_dims: int,
        use_age: str,
        fc_stages: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = "gelu",
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
        **kwargs: Any,
    ):
        super().__init__()

        if use_age not in ["fc", "conv", "no"]:
            raise ValueError(f"{self.__class__.__name__}.__init__(use_age) " f"receives one of ['fc', 'conv', 'no'].")
        if fc_stages < 1:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(fc_stages) receives " f"an integer equal to ore more than 1."
            )

        self.use_age = use_age
        if self.use_age == "conv":
            in_channels += 1
        self.fc_stages = fc_stages

        self.nn_act = get_activation_class(activation, class_name=self.__class__.__name__)
        self.activation = activation

        self.image_h, self.image_w = seq_len_2d
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = out_dims
        self.norm_layer = norm_layer

        if conv_stem_configs is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.__init__(conv_stem_configs) " f"functionality is not implemented yet."
            )
            # As per https://arxiv.org/abs/2106.14881
            # seq_proj = nn.Sequential()
            # prev_channels = in_channels
            # for i, conv_stem_layer_config in enumerate(conv_stem_configs):
            #     seq_proj.add_module(
            #         f"conv_bn_relu_{i}",
            #         ConvNormActivation(
            #             in_channels=prev_channels,
            #             out_channels=conv_stem_layer_config.out_channels,
            #             kernel_size=conv_stem_layer_config.kernel_size,
            #             stride=conv_stem_layer_config.stride,
            #             norm_layer=conv_stem_layer_config.norm_layer,
            #             activation_layer=conv_stem_layer_config.activation_layer,
            #         ),
            #     )
            #     prev_channels = conv_stem_layer_config.out_channels
            # seq_proj.add_module(
            #     "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            # )
            # self.conv_proj: nn.Module = seq_proj
        else:
            self.n_h, h_conv_filter = self._decide_patch_size(
                in_size=self.image_h, target_num_min=size_min, target_num_max=size_max
            )
            self.n_w, w_conv_filter = self._decide_patch_size(
                in_size=self.image_w, target_num_min=size_min, target_num_max=size_max
            )
            conv_filter = {k: (h_conv_filter[k], w_conv_filter[k]) for k in h_conv_filter.keys() if k != "pool"}
            self.conv_proj = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, **conv_filter)

        # Add a class token
        self.seq_length = self.n_h * self.n_w + 1
        self.output_length = self.n_h * self.n_w + 1
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.encoder = Encoder(
            self.seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            self.nn_act,
            norm_layer,
        )

        prev_dim = hidden_dim
        if self.use_age == "fc":
            prev_dim = prev_dim + 1
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()

        for i in range(self.fc_stages - 1):
            # TODO: Dropout or Normalization layers can be added here.
            heads_layers[f"linear{i + 1}"] = nn.Linear(prev_dim, prev_dim // 2)
            heads_layers[f"dropout{i + 1}"] = nn.Dropout(dropout)
            heads_layers[f"act{i + 1}"] = self.nn_act()
            prev_dim = prev_dim // 2
        heads_layers["head"] = nn.Linear(prev_dim, out_dims)
        self.heads = nn.Sequential(heads_layers)

        self.reset_weights()

    def reset_weights(self):
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        for i in range(self.fc_stages - 1):
            linear_name = f"linear{i + 1}"
            if hasattr(self.heads, linear_name) and isinstance(getattr(self.heads, linear_name), nn.Linear):
                fan_in = getattr(self.heads, linear_name).in_features
                if self.activation == "tanh":
                    nn.init.trunc_normal_(getattr(self.heads, linear_name).weight, std=math.sqrt(1 / fan_in))
                else:
                    nn.init.trunc_normal_(getattr(self.heads, linear_name).weight, std=math.sqrt(2.0 / fan_in))
                nn.init.zeros_(getattr(self.heads, linear_name).bias)
        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _decide_patch_size(self, in_size: int, target_num_min: int, target_num_max: int):
        success = False

        for target_num in reversed(range(target_num_min, target_num_max + 1)):
            kernel_size_base = int(np.ceil(in_size / target_num))

            for kernel_size in range(kernel_size_base, kernel_size_base + kernel_size_base // 2 + 1):
                conv_filter_list = [{"kernel_size": kernel_size}]
                try:
                    program_conv_filters(
                        sequence_length=in_size,
                        conv_filter_list=conv_filter_list,
                        output_lower_bound=target_num,
                        output_upper_bound=target_num,
                        pad=False,
                        stride_to_pool_ratio=0.0001,
                        trials=20,
                        verbose=False,
                    )
                except RuntimeError as e:
                    # print('Failed:', target_num, kernel_size_base, conv_filter_list)
                    pass
                else:
                    return target_num, conv_filter_list[0]

        if success is False:
            raise RuntimeError(
                f"{self.__class__.__name__}._decide_patch_size() " f"failed to calculate the proper patch size"
            )

    def get_output_length(self):
        return self.output_length

    def get_num_fc_stages(self):
        return self.fc_stages

    def compute_feature_embedding(self, x, age, target_from_last: int = 0):
        # Reshape and permute the input tensor
        n, _, h, w = x.size()
        if self.use_age == "conv":
            age = age.reshape((n, 1, 1, 1)).expand(n, 1, h, w)
            x = torch.cat((x, age), dim=1)

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)

        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, self.n_h * self.n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        if self.use_age == "fc":
            x = torch.cat((x, age.reshape(-1, 1)), dim=1)

        if target_from_last == 0:
            x = self.heads(x)
        else:
            if target_from_last > self.fc_stages:
                raise ValueError(
                    f"{self.__class__.__name__}.compute_feature_embedding(target_from_last) receives "
                    f"an integer equal to or smaller than fc_stages={self.fc_stages}."
                )

            for l in range(self.fc_stages - target_from_last):
                x = self.heads[l](x)
        return x

    def forward(self, x: torch.Tensor, age: torch.Tensor):
        x = self.compute_feature_embedding(x, age)
        # return F.log_softmax(x, dim=1)
        return x


def _vision_transformer(
    seq_len_2d: Tuple[int, int],
    size_min: int,
    size_max: int,
    in_channels: int,
    out_dims: int,
    use_age: str,
    fc_stages: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    **kwargs: Any,
) -> VisionTransformer:
    model = VisionTransformer(
        seq_len_2d=seq_len_2d,
        size_min=size_min,
        size_max=size_max,
        in_channels=in_channels,
        out_dims=out_dims,
        use_age=use_age,
        fc_stages=fc_stages,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    return model


def vit_b_8(**kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_b_8 architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.
    """
    return _vision_transformer(
        arch="vit_b_8",
        size_min=7,
        size_max=9,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        **kwargs,
    )


def vit_b_16(**kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_b_16 architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.
    """
    return _vision_transformer(
        arch="vit_b_16",
        size_min=14,
        size_max=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        **kwargs,
    )


def vit_b_32(**kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_b_32 architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.
    """
    return _vision_transformer(
        arch="vit_b_32",
        size_min=22,
        size_max=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        **kwargs,
    )


def vit_l_8(**kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_l_8 architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.
    """
    return _vision_transformer(
        arch="vit_l_8",
        size_min=7,
        size_max=9,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        **kwargs,
    )


def vit_l_16(**kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_l_16 architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.
    """
    return _vision_transformer(
        arch="vit_l_16",
        size_min=14,
        size_max=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        **kwargs,
    )


def vit_l_32(**kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_l_32 architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.
    """
    return _vision_transformer(
        arch="vit_l_32",
        size_min=22,
        size_max=32,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        **kwargs,
    )


def interpolate_embeddings(
    image_size: int,
    patch_size: int,
    model_state: "OrderedDict[str, torch.Tensor]",
    interpolation_mode: str = "bicubic",
    reset_heads: bool = False,
) -> "OrderedDict[str, torch.Tensor]":
    """This function helps interpolating positional embeddings during checkpoint loading,
    especially when you want to apply a pretrained model on images with different resolution.

    Args:
        image_size (int): Image size of the new model.
        patch_size (int): Patch size of the new model.
        model_state (OrderedDict[str, torch.Tensor]): State dict of the pretrained model.
        interpolation_mode (str): The algorithm used for upsampling. Default: bicubic.
        reset_heads (bool): If true, not copying the state of heads. Default: False.

    Returns:
        OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
    """
    # Shape of pos_embedding is (1, seq_length, hidden_dim)
    pos_embedding = model_state["encoder.pos_embedding"]
    n, seq_length, hidden_dim = pos_embedding.shape
    if n != 1:
        raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")

    new_seq_length = (image_size // patch_size) ** 2 + 1

    # Need to interpolate the weights for the position embedding.
    # We do this by reshaping the positions embeddings to a 2d grid, performing
    # an interpolation in the (h, w) space and then reshaping back to a 1d grid.
    if new_seq_length != seq_length:
        # The class token embedding shouldn't be interpolated so we split it up.
        seq_length -= 1
        new_seq_length -= 1
        pos_embedding_token = pos_embedding[:, :1, :]
        pos_embedding_img = pos_embedding[:, 1:, :]

        # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
        pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
        seq_length_1d = int(math.sqrt(seq_length))
        torch._assert(seq_length_1d * seq_length_1d == seq_length, "seq_length is not a perfect square!")

        # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
        pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
        new_seq_length_1d = image_size // patch_size

        # Perform interpolation.
        # (1, hidden_dim, seq_l_1d, seq_l_1d) -> (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
        new_pos_embedding_img = nn.functional.interpolate(
            pos_embedding_img,
            size=new_seq_length_1d,
            mode=interpolation_mode,
            align_corners=True,
        )

        # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)
        new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)

        # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
        new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
        new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)

        model_state["encoder.pos_embedding"] = new_pos_embedding

        if reset_heads:
            model_state_copy: "OrderedDict[str, torch.Tensor]" = OrderedDict()
            for k, v in model_state.items():
                if not k.startswith("heads"):
                    model_state_copy[k] = v
            model_state = model_state_copy

    return model_state
