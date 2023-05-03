"""
Modified from:
    - torchvision implementation of VGG (linked below)
    - https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
    - VGG paper: https://arxiv.org/abs/1409.1556
"""

from typing import List, Union, Dict

import torch
import torch.nn as nn

from .utils import program_conv_filters
from .activation import get_activation_class


vgg_layer_cfgs: Dict[str, List[Dict[str, int]]] = {
    "1D-VGG-11": [
        {"layers": 1, "channel_mul": 1},
        {"layers": 1, "channel_mul": 2},
        {"layers": 2, "channel_mul": 4},
        {"layers": 2, "channel_mul": 8},
        {"layers": 2, "channel_mul": 8},
    ],
    "1D-VGG-13": [
        {"layers": 2, "channel_mul": 1},
        {"layers": 2, "channel_mul": 2},
        {"layers": 2, "channel_mul": 4},
        {"layers": 2, "channel_mul": 8},
        {"layers": 2, "channel_mul": 8},
    ],
    "1D-VGG-16": [
        {"layers": 2, "channel_mul": 1},
        {"layers": 2, "channel_mul": 2},
        {"layers": 3, "channel_mul": 4},
        {"layers": 3, "channel_mul": 8},
        {"layers": 3, "channel_mul": 8},
    ],
    "1D-VGG-19": [
        {"layers": 2, "channel_mul": 1},
        {"layers": 2, "channel_mul": 2},
        {"layers": 4, "channel_mul": 4},
        {"layers": 4, "channel_mul": 8},
        {"layers": 4, "channel_mul": 8},
    ],
}


class VGG1D(nn.Module):
    def __init__(
        self,
        model: str,
        in_channels: int,
        out_dims: int,
        seq_length: int,
        use_age: str,
        base_channels: int = 64,
        dropout: float = 0.5,
        batch_norm: bool = False,
        fc_stages: int = 2,
        base_pool: str = "max",
        final_pool: str = "average",
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__()

        if model not in vgg_layer_cfgs.keys():
            raise ValueError(
                f"{self.__class__.__name__}.__init__(model) " f"receives one of [{vgg_layer_cfgs.keys()}]."
            )

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

        self.batch_norm = batch_norm
        self.nn_act = get_activation_class(activation, class_name=self.__class__.__name__)

        if base_pool == "average":
            self.base_pool = nn.AvgPool1d
        elif base_pool == "max":
            self.base_pool = nn.MaxPool1d

        layer_cfgs = vgg_layer_cfgs[model]
        conv_filter_list = []
        for i in layer_cfgs:  # to prevent shallow copying
            conv_filter_list.append({"kernel_size": 9})
        self.sequence_length = seq_length
        self.output_length = program_conv_filters(
            sequence_length=seq_length,
            conv_filter_list=conv_filter_list,
            output_lower_bound=4,
            output_upper_bound=8,
            class_name=self.__class__.__name__,
        )

        # convolution stage
        self.current_channels = in_channels
        self.conv_stage1 = self._make_conv_stage(conv_filter_list[0], layer_cfgs[0], base_channels)
        self.conv_stage2 = self._make_conv_stage(conv_filter_list[1], layer_cfgs[1], base_channels)
        self.conv_stage3 = self._make_conv_stage(conv_filter_list[2], layer_cfgs[2], base_channels)
        self.conv_stage4 = self._make_conv_stage(conv_filter_list[3], layer_cfgs[3], base_channels)
        self.conv_stage5 = self._make_conv_stage(conv_filter_list[4], layer_cfgs[4], base_channels)

        # pooling right before fully-connection
        if final_pool == "average":
            self.final_pool = nn.AdaptiveAvgPool1d(1)
        elif final_pool == "max":
            self.final_pool = nn.AdaptiveMaxPool1d(1)

        # fully-connected stage
        fc_stage: List[nn.Module] = []
        if self.use_age == "fc":
            self.current_channels = self.current_channels + 1

        for i in range(fc_stages - 1):
            if self.batch_norm:
                layer = nn.Sequential(
                    nn.Linear(self.current_channels, self.current_channels // 2, bias=False),
                    nn.Dropout(p=dropout),
                    nn.BatchNorm1d(self.current_channels // 2),
                    self.nn_act(),
                )
            else:
                layer = nn.Sequential(
                    nn.Linear(self.current_channels, self.current_channels // 2, bias=True),
                    nn.Dropout(p=dropout),
                    self.nn_act(),
                )
            self.current_channels = self.current_channels // 2
            fc_stage.append(layer)

        fc_stage.append(nn.Linear(self.current_channels, out_dims, bias=True))
        self.fc_stage = nn.Sequential(*fc_stage)

        self.reset_weights()

    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear,)):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def _make_conv_stage(self, conv_filter, cfg, base_channels):
        conv_layers: List[nn.Module] = []

        if conv_filter["pool"] > 1:
            conv_layers += [self.base_pool(conv_filter["pool"])]

        for k in range(cfg["layers"]):
            if k == 0:
                stride = conv_filter["stride"]
            else:
                stride = 1

            if self.batch_norm:
                conv_layers += [
                    nn.Conv1d(
                        in_channels=self.current_channels,
                        out_channels=cfg["channel_mul"] * base_channels,
                        kernel_size=conv_filter["kernel_size"],
                        padding=conv_filter["kernel_size"] // 2,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm1d(cfg["channel_mul"] * base_channels),
                    self.nn_act(),
                ]
            else:
                conv_layers += [
                    nn.Conv1d(
                        in_channels=self.current_channels,
                        out_channels=cfg["channel_mul"] * base_channels,
                        kernel_size=conv_filter["kernel_size"],
                        padding=conv_filter["kernel_size"] // 2,
                        stride=stride,
                        bias=True,
                    ),
                    self.nn_act(),
                ]

            self.current_channels = cfg["channel_mul"] * base_channels
        return nn.Sequential(*conv_layers)

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

        x = self.conv_stage1(x)
        x = self.conv_stage2(x)
        x = self.conv_stage3(x)
        x = self.conv_stage4(x)
        x = self.conv_stage5(x)

        x = self.final_pool(x)
        x = x.reshape((N, -1))

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
