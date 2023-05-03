import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional

from .activation import get_activation_class
from .activation import get_activation_functional
from models.utils import program_conv_filters
from models.utils import make_pool_or_not

# __all__ = []


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding generator proposed in 'Attention is All You Need' paper.
    """

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model, requires_grad=False)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return self.pe[: x.size(0)]


class CNNTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_dims: int,
        seq_length: int,
        fc_stages: int,
        use_age: str,
        base_channels=256,
        n_encoders=6,
        n_heads=8,
        dropout=0.2,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation="relu",
        base_pool: str = "max",
        final_pool: str = "average",
        **kwargs,
    ):
        super().__init__()

        if use_age not in ["fc", "conv", "no"]:
            raise ValueError(f"{self.__class__.__name__}.__init__(use_age) " f"receives one of ['fc', 'conv', 'no'].")

        if final_pool not in ["average", "max"] or base_pool not in ["average", "max"]:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(final_pool, base_pool) both "
                f"receives one of ['average', 'max']."
            )

        if fc_stages < 1:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(fc_stages) receives " f"an integer equal to ore more than 1."
            )

        self.use_age = use_age
        if self.use_age == "conv":
            in_channels += 1
        self.fc_stages = fc_stages

        self.nn_act = get_activation_class(activation, class_name=self.__class__.__name__)
        self.F_act = get_activation_functional(activation, class_name=self.__class__.__name__)

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        if base_pool == "average":
            self.base_pool = nn.AvgPool1d
        elif base_pool == "max":
            self.base_pool = nn.MaxPool1d

        conv_filter_list = [
            {"kernel_size": 21},
            {"kernel_size": 9},
            {"kernel_size": 9},
            {"kernel_size": 9},
        ]
        self.sequence_length = seq_length
        self.output_length = program_conv_filters(
            sequence_length=seq_length,
            conv_filter_list=conv_filter_list,
            output_lower_bound=32,
            output_upper_bound=48,
            class_name=self.__class__.__name__,
        )

        cf = conv_filter_list[0]
        self.pool0 = make_pool_or_not(self.base_pool, cf["pool"])
        self.conv0 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size=cf["kernel_size"],
            stride=cf["stride"],
            padding=cf["kernel_size"] // 2,
            bias=False,
        )
        self.norm0 = norm_layer(base_channels)
        self.act0 = self.nn_act()

        cf = conv_filter_list[1]
        self.pool1 = make_pool_or_not(self.base_pool, cf["pool"])
        self.conv1 = nn.Conv1d(
            in_channels=base_channels,
            out_channels=2 * base_channels,
            kernel_size=cf["kernel_size"],
            stride=cf["stride"],
            padding=cf["kernel_size"] // 2,
            bias=False,
        )
        self.norm1 = norm_layer(2 * base_channels)
        self.act1 = self.nn_act()

        cf = conv_filter_list[2]
        self.pool2 = make_pool_or_not(self.base_pool, cf["pool"])
        self.conv2 = nn.Conv1d(
            in_channels=2 * base_channels,
            out_channels=4 * base_channels,
            kernel_size=cf["kernel_size"],
            stride=cf["stride"],
            padding=cf["kernel_size"] // 2,
            bias=False,
        )
        self.norm2 = norm_layer(4 * base_channels)
        self.act2 = self.nn_act()

        self.pos_encoder = PositionalEncoding(4 * base_channels)
        encoder_layers = nn.TransformerEncoderLayer(
            4 * base_channels, nhead=n_heads, dim_feedforward=16 * base_channels, dropout=dropout, activation=self.F_act
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_encoders)

        cf = conv_filter_list[3]
        self.pool3 = make_pool_or_not(self.base_pool, cf["pool"])
        self.conv3 = nn.Conv1d(
            in_channels=4 * base_channels,
            out_channels=2 * base_channels,
            kernel_size=cf["kernel_size"],
            stride=cf["stride"],
            padding=cf["kernel_size"] // 2,
            bias=False,
        )
        self.norm3 = norm_layer(2 * base_channels)
        self.act3 = self.nn_act()
        current_channels = 2 * base_channels

        if final_pool == "average":
            self.final_pool = nn.AdaptiveAvgPool1d(1)
        elif final_pool == "max":
            self.final_pool = nn.AdaptiveMaxPool1d(1)

        fc_stage = []
        if self.use_age == "fc":
            current_channels = current_channels + 1

        for i in range(fc_stages - 1):
            layer = nn.Sequential(
                nn.Linear(current_channels, current_channels // 2, bias=False),
                nn.Dropout(p=dropout),
                nn.BatchNorm1d(current_channels // 2),
                self.nn_act(),
            )
            current_channels = current_channels // 2
            fc_stage.append(layer)
        fc_stage.append(nn.Linear(current_channels, out_dims))
        self.fc_stage = nn.Sequential(*fc_stage)

        self.reset_weights()

    def reset_weights(self):
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def get_output_length(self):
        return self.output_length

    def get_num_fc_stages(self):
        return self.fc_stages

    def compute_feature_embedding(self, x, age, target_from_last: int = 0):
        N, C, L = x.size()

        if self.use_age == "conv":
            age = age.reshape((N, 1, 1))
            age = torch.cat([age for i in range(L)], dim=2)
            x = torch.cat((x, age), dim=1)

        # conv-bn-act
        x = self.pool0(x)
        x = self.conv0(x)
        x = self.act0(self.norm0(x))

        # conv-bn-act
        x = self.pool1(x)
        x = self.conv1(x)
        x = self.act1(self.norm1(x))

        # conv-bn-act
        x = self.pool2(x)
        x = self.conv2(x)
        x = self.act2(self.norm2(x))

        # transformer encoder layers
        x = torch.permute(x, (2, 0, 1))  # minibatch, dimension, length --> length, minibatch, dimension
        x = x + self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = torch.permute(x, (1, 2, 0))  # length, minibatch, dimension --> minibatch, dimension, length

        # conv-bn-act again
        x = self.pool3(x)
        x = self.conv3(x)
        x = self.act3(self.norm3(x))

        x = self.final_pool(x).reshape((N, -1))

        # fully-connected layers
        if self.use_age == "fc":
            x = torch.cat((x, age.reshape(-1, 1)), dim=1)

        if target_from_last == 0:
            x = self.fc_stage(x)
        else:
            if target_from_last > self.fc_stages:
                raise ValueError(
                    f"{self.__class__.__name__}.compute_feature_embedding(target_from_last) receives "
                    f"an integer equal to or smaller than fc_stages={self.fc_stages}."
                )

            for l in range(self.fc_stages - target_from_last):
                x = self.fc_stage[l](x)
        return x

    def forward(self, x, age):
        x = self.compute_feature_embedding(x, age)
        # return F.log_softmax(x, dim=1)
        return x
