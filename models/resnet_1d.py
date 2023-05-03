"""
1D modification of:
    - torchvision implementation of ResNet, ResNeXt, and Wide ResNet (linked below)
    - https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    - ResNet paper: https://arxiv.org/abs/1603.05027
    - ReNeXt paper: https://arxiv.org/abs/1611.05431
    - Wide ResNet paper: https://arxiv.org/abs/1605.07146
"""

from typing import Callable, Optional, Type, Union, List, Any

import torch
import torch.nn as nn

from .activation import get_activation_class
from .utils import program_conv_filters

# __all__ = []


class BasicBlock1D(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_channels: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = nn.ReLU,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_channels != 64:
            raise ValueError("BasicBlock1D only supports groups=1 and base_channels=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock1D")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.norm1 = norm_layer(out_channels)
        self.act1 = activation()

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.norm2 = norm_layer(out_channels)
        self.act2 = activation()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.act2(out)

        return out


class BottleneckBlock1D(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_channels: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = nn.ReLU,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(out_channels * (base_channels / 64.0)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=width, kernel_size=1, stride=1, bias=False)
        self.norm1 = norm_layer(width)
        self.act1 = activation()

        self.conv2 = nn.Conv1d(
            in_channels=width,
            out_channels=width,
            groups=groups,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=kernel_size // 2,
            bias=False,
        )
        self.norm2 = norm_layer(width)
        self.act2 = activation()

        self.conv3 = nn.Conv1d(
            in_channels=width, out_channels=out_channels * self.expansion, kernel_size=1, stride=1, bias=False
        )
        self.norm3 = norm_layer(out_channels * self.expansion)
        self.act3 = activation()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.act3(out)

        return out


class ResNet1D(nn.Module):
    def __init__(
        self,
        block: str,
        conv_layers: List[int],
        in_channels: int,
        out_dims: int,
        seq_length: int,
        base_channels: int,
        use_age: str,
        fc_stages: int,
        dropout: float = 0.1,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: str = "relu",
        base_pool: str = "max",
        final_pool: str = "average",
        **kwargs,
    ) -> None:
        super().__init__()

        if block not in ["basic", "bottleneck"]:
            raise ValueError(f"{self.__class__.__name__}.__init__(block) " f"receives one of ['basic', 'bottleneck'].")

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

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.zero_init_residual = zero_init_residual

        self.current_channels = base_channels
        self.groups = groups
        self.base_channels = width_per_group
        self.groups = groups

        if base_pool == "average":
            self.base_pool = nn.AvgPool1d
        elif base_pool == "max":
            self.base_pool = nn.MaxPool1d

        if block == "basic":
            block = BasicBlock1D
            conv_filter_list = [
                {"kernel_size": 41},
                {"kernel_size": 9},  # 9 or 17 to reflect the composition of 9conv and 9conv
                {"kernel_size": 9},  # 9 or 17 to reflect the composition of 9conv and 9conv
                {"kernel_size": 9},  # 9 or 17 to reflect the composition of 9conv and 9conv
                {"kernel_size": 9},  # 9 or 17 to reflect the composition of 9conv and 9conv
            ]
        else:  # bottleneck case
            block = BottleneckBlock1D
            conv_filter_list = [
                {"kernel_size": 41},
                {"kernel_size": 9},
                {"kernel_size": 9},
                {"kernel_size": 9},
                {"kernel_size": 9},
            ]

        self.sequence_length = seq_length
        self.output_length = program_conv_filters(
            sequence_length=seq_length,
            conv_filter_list=conv_filter_list,
            output_lower_bound=4,
            output_upper_bound=8,
            class_name=self.__class__.__name__,
        )

        cf = conv_filter_list[0]
        input_stage = []
        if cf["pool"] > 1:
            input_stage.append(self.base_pool(cf["pool"]))
        input_stage.append(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.current_channels,
                kernel_size=cf["kernel_size"],
                stride=cf["stride"],
                padding=cf["kernel_size"] // 2,
                bias=False,
            )
        )
        input_stage.append(norm_layer(self.current_channels))
        input_stage.append(self.nn_act())
        self.input_stage = nn.Sequential(*input_stage)

        cf = conv_filter_list[1]
        self.conv_stage1 = self._make_conv_stage(
            block,
            base_channels,
            conv_layers[0],
            cf["kernel_size"],
            stride=cf["stride"],
            pre_pool=cf["pool"],
            activation=self.nn_act,
        )
        cf = conv_filter_list[2]
        self.conv_stage2 = self._make_conv_stage(
            block,
            2 * base_channels,
            conv_layers[1],
            cf["kernel_size"],
            stride=cf["stride"],
            pre_pool=cf["pool"],
            activation=self.nn_act,
        )
        cf = conv_filter_list[3]
        self.conv_stage3 = self._make_conv_stage(
            block,
            4 * base_channels,
            conv_layers[2],
            cf["kernel_size"],
            stride=cf["stride"],
            pre_pool=cf["pool"],
            activation=self.nn_act,
        )
        cf = conv_filter_list[4]
        self.conv_stage4 = self._make_conv_stage(
            block,
            8 * base_channels,
            conv_layers[3],
            cf["kernel_size"],
            stride=cf["stride"],
            pre_pool=cf["pool"],
            activation=self.nn_act,
        )

        if final_pool == "average":
            self.final_pool = nn.AdaptiveAvgPool1d(1)
        elif final_pool == "max":
            self.final_pool = nn.AdaptiveMaxPool1d(1)

        fc_stage = []
        if self.use_age == "fc":
            self.current_channels = self.current_channels + 1

        for i in range(fc_stages - 1):
            layer = nn.Sequential(
                nn.Linear(self.current_channels, self.current_channels // 2, bias=False),
                nn.Dropout(p=dropout),
                norm_layer(self.current_channels // 2),
                self.nn_act(),
            )
            self.current_channels = self.current_channels // 2
            fc_stage.append(layer)
        fc_stage.append(nn.Linear(self.current_channels, out_dims))
        self.fc_stage = nn.Sequential(*fc_stage)

        self.reset_weights()

    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear,)):
                nn.init.xavier_normal_(m.weight)
            elif hasattr(m, "reset_parameters"):
                m.reset_parameters()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckBlock1D):
                    nn.init.constant_(m.norm3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_conv_stage(
        self,
        block: Type[Union[BasicBlock1D, BottleneckBlock1D]],
        channels: int,
        blocks: int,
        kernel_size: int,
        stride: int = 1,
        pre_pool: int = 1,
        activation=nn.ReLU,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.current_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.current_channels,
                    out_channels=channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(channels * block.expansion),
            )

        conv_layers = []

        if pre_pool > 1:
            conv_layers.append(self.base_pool(pre_pool))

        conv_layers.append(
            block(
                in_channels=self.current_channels,
                out_channels=channels,
                kernel_size=kernel_size,
                base_channels=self.base_channels,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                activation=activation,
            )
        )

        self.current_channels = channels * block.expansion
        for _ in range(1, blocks):
            conv_layers.append(
                block(
                    in_channels=self.current_channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    base_channels=self.base_channels,
                    groups=self.groups,
                    stride=1,
                    activation=activation,
                )
            )

        return nn.Sequential(*conv_layers)

    def get_output_length(self):
        return self.output_length

    def get_num_fc_stages(self):
        return self.fc_stages

    def compute_feature_embedding(self, x, age, target_from_last: int = 0):
        N, _, L = x.size()
        if self.use_age == "conv":
            age = age.reshape((N, 1, 1)).expand(N, 1, L)
            x = torch.cat((x, age), dim=1)

        x = self.input_stage(x)

        x = self.conv_stage1(x)
        x = self.conv_stage2(x)
        x = self.conv_stage3(x)
        x = self.conv_stage4(x)

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
