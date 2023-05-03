"""
Modified from:
    - torchvision implementation of ResNet, ResNeXt, and Wide ResNet (linked below)
    - https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    - ResNet paper: https://arxiv.org/abs/1603.05027
    - ReNeXt paper: https://arxiv.org/abs/1611.05431
    - Wide ResNet paper: https://arxiv.org/abs/1605.07146
"""

from typing import Callable, Optional, Type, Union, List, Any, Tuple

import torch
import torch.nn as nn

from .activation import get_activation_class
from .utils import program_conv_filters

__all__ = [
    "ResNet2D",
    "resnet18_2d",
    "resnet34_2d",
    "resnet50_2d",
    "resnet101_2d",
    "resnet152_2d",
    "resnext50_32x4d_2d",
    "resnext101_32x8d_2d",
    "wide_resnet50_2d",
    "wide_resnet101_2_2d",
    "BasicBlock2D",
    "Bottleneck2D",
]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock2D(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
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
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_channels != 64:
            raise ValueError("BasicBlock2D only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock2D")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.norm1 = norm_layer(out_channels)
        self.act1 = activation()

        self.conv2 = conv3x3(out_channels, out_channels)
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


class Bottleneck2D(nn.Module):
    # Bottleneck2D in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
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
            norm_layer = nn.BatchNorm2d
        width = int(out_channels * (base_channels / 64.0)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_channels, width)
        self.norm1 = norm_layer(width)
        self.act1 = activation()

        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.norm2 = norm_layer(width)
        self.act2 = activation()

        self.conv3 = conv1x1(width, out_channels * self.expansion)
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


class ResNet2D(nn.Module):
    def __init__(
        self,
        block: str,
        conv_layers: List[int],
        in_channels: int,
        out_dims: int,
        seq_len_2d: Tuple[int],
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
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.zero_init_residual = zero_init_residual

        self.current_channels = base_channels
        self.groups = groups
        self.base_channels = width_per_group

        if base_pool == "average":
            self.base_pool = nn.AvgPool1d
        elif base_pool == "max":
            self.base_pool = nn.MaxPool1d

        if block == "basic":
            block = BasicBlock2D
            conv_filter_list = [
                {"kernel_size": 7},
                {"kernel_size": 3},  # 3 or 5 to reflect the composition of 9conv and 9conv
                {"kernel_size": 3},  # 3 or 5 to reflect the composition of 9conv and 9conv
                {"kernel_size": 3},  # 3 or 5 to reflect the composition of 9conv and 9conv
                {"kernel_size": 3},  # 3 or 5 to reflect the composition of 9conv and 9conv
            ]
        else:  # bottleneck case
            block = Bottleneck2D
            conv_filter_list = [
                {"kernel_size": 7},
                {"kernel_size": 3},
                {"kernel_size": 3},
                {"kernel_size": 3},
                {"kernel_size": 3},
            ]

        self.seq_len_2d = seq_len_2d
        self.output_length = program_conv_filters(
            sequence_length=min(seq_len_2d),
            conv_filter_list=conv_filter_list,
            output_lower_bound=4,
            output_upper_bound=8,
            stride_to_pool_ratio=0.7,
            class_name=self.__class__.__name__,
        )

        cf = conv_filter_list[0]
        input_stage = []
        if cf["pool"] > 1:
            input_stage.append(self.base_pool(cf["pool"]))
        input_stage.append(
            nn.Conv2d(
                in_channels,
                self.current_channels,
                kernel_size=(cf["kernel_size"], cf["kernel_size"]),
                stride=(cf["stride"], cf["stride"]),
                padding=cf["kernel_size"] // 2,
                bias=False,
            )
        )
        input_stage.append(norm_layer(self.current_channels))
        input_stage.append(self.nn_act())
        self.input_stage = nn.Sequential(*input_stage)

        cf = conv_filter_list[1]
        self.conv_stage1 = self._make_conv_stage(
            block=block, planes=base_channels, blocks=conv_layers[0], stride=cf["stride"], activation=self.nn_act
        )
        cf = conv_filter_list[2]
        self.conv_stage2 = self._make_conv_stage(
            block=block, planes=2 * base_channels, blocks=conv_layers[1], stride=cf["stride"], activation=self.nn_act
        )
        cf = conv_filter_list[3]
        self.conv_stage3 = self._make_conv_stage(
            block=block, planes=4 * base_channels, blocks=conv_layers[2], stride=cf["stride"], activation=self.nn_act
        )
        cf = conv_filter_list[4]
        self.conv_stage4 = self._make_conv_stage(
            block=block, planes=8 * base_channels, blocks=conv_layers[3], stride=cf["stride"], activation=self.nn_act
        )

        if final_pool == "average":
            self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif final_pool == "max":
            self.final_pool = nn.AdaptiveMaxPool2d((1, 1))

        fc_stage = []
        if self.use_age == "fc":
            self.current_channels = self.current_channels + 1

        for i in range(fc_stages - 1):
            layer = nn.Sequential(
                nn.Linear(self.current_channels, self.current_channels // 2, bias=False),
                nn.Dropout(p=dropout),
                nn.BatchNorm1d(self.current_channels // 2),
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
            elif hasattr(m, "reset_parameters"):
                m.reset_parameters()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck2D):
                    nn.init.constant_(m.norm3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock2D):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_conv_stage(
        self,
        block: Type[Union[BasicBlock2D, Bottleneck2D]],
        planes: int,
        blocks: int,
        stride: int = 1,
        pre_pool: int = 1,
        activation=nn.ReLU,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.current_channels != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.current_channels, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        conv_layers = []

        if pre_pool > 1:
            conv_layers.append(self.base_pool(pre_pool))

        conv_layers.append(
            block(
                self.current_channels,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_channels,
                dilation=1,
                norm_layer=norm_layer,
                activation=activation,
            )
        )

        self.current_channels = planes * block.expansion
        for _ in range(1, blocks):
            conv_layers.append(
                block(
                    self.current_channels,
                    planes,
                    stride=1,
                    downsample=None,
                    groups=self.groups,
                    base_channels=self.base_channels,
                    dilation=1,
                    norm_layer=norm_layer,
                    activation=activation,
                )
            )

        return nn.Sequential(*conv_layers)

    def get_output_length(self):
        return self.output_length

    def get_num_fc_stages(self):
        return self.fc_stages

    def compute_feature_embedding(self, x, age, target_from_last: int = 0):
        if self.use_age == "conv":
            N, _, H, W = x.size()
            age = age.reshape((N, 1, 1, 1)).expand(N, 1, H, W)
            x = torch.cat((x, age), dim=1)

        x = self.input_stage(x)

        x = self.conv_stage1(x)
        x = self.conv_stage2(x)
        x = self.conv_stage3(x)
        x = self.conv_stage4(x)

        x = self.final_pool(x)
        x = torch.flatten(x, 1)

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

    def _forward_impl(self, x: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        x = self.compute_feature_embedding(x, age)
        return x

    def forward(self, x: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        x = self.compute_feature_embedding(x, age)
        return x


def _resnet_2d(
    arch: str,
    block: Type[Union[BasicBlock2D, Bottleneck2D]],
    conv_layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet2D:
    model = ResNet2D(block, conv_layers, **kwargs)
    return model


def resnet18_2d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_2d("resnet18_2d", BasicBlock2D, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34_2d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_2d("resnet34_2d", BasicBlock2D, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50_2d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_2d("resnet50_2d", Bottleneck2D, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101_2d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_2d("resnet101_2d", Bottleneck2D, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet152_2d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_2d("resnet152_2d", Bottleneck2D, [3, 8, 36, 3], pretrained, progress, **kwargs)


def resnext50_32x4d_2d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet_2d("resnext50_32x4d_2d", Bottleneck2D, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnext101_32x8d_2d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet_2d("resnext101_32x8d_2d", Bottleneck2D, [3, 4, 23, 3], pretrained, progress, **kwargs)


def wide_resnet50_2d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the Bottleneck2D number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet_2d("wide_resnet50_2d", Bottleneck2D, [3, 4, 6, 3], pretrained, progress, **kwargs)


def wide_resnet101_2_2d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the Bottleneck2D number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet_2d("wide_resnet101_2_2d", Bottleneck2D, [3, 4, 23, 3], pretrained, progress, **kwargs)
