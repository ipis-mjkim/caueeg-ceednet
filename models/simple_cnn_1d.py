import torch
import torch.nn as nn

from .utils import program_conv_filters
from .utils import make_pool_or_not
from .activation import get_activation_class
from .activation import get_activation_functional

# __all__ = []


class TinyCNN1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_dims: int,
        fc_stages: int,
        seq_length: int,
        use_age: str,
        base_channels: int = 64,
        dropout: float = 0.3,
        base_pool: str = "max",
        final_pool: str = "average",
        activation: str = "relu",
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

        if base_pool == "average":
            self.base_pool = nn.AvgPool1d
        elif base_pool == "max":
            self.base_pool = nn.MaxPool1d

        conv_filter_list = [
            {"kernel_size": 35},
            {"kernel_size": 9},
        ]
        self.sequence_length = seq_length
        self.output_length = program_conv_filters(
            sequence_length=seq_length,
            conv_filter_list=conv_filter_list,
            output_lower_bound=4,
            output_upper_bound=8,
            stride_to_pool_ratio=0.7,
            class_name=self.__class__.__name__,
        )

        cf = conv_filter_list[0]
        self.pool1 = make_pool_or_not(self.base_pool, cf["pool"])
        self.conv1 = nn.Conv1d(
            in_channels,
            base_channels,
            kernel_size=cf["kernel_size"],
            padding=cf["kernel_size"] // 2,
            stride=cf["stride"],
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(base_channels)

        cf = conv_filter_list[1]
        self.pool2 = make_pool_or_not(self.base_pool, cf["pool"])
        self.conv2 = nn.Conv1d(
            base_channels,
            base_channels,
            kernel_size=cf["kernel_size"],
            padding=cf["kernel_size"] // 2,
            stride=cf["stride"],
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(base_channels)
        current_channels = base_channels

        if final_pool == "average":
            self.final_pool = nn.AdaptiveAvgPool1d(1)
        elif final_pool == "max":
            self.final_pool = nn.AdaptiveMaxPool1d(1)

        fc_stage = []
        if self.use_age == "fc":
            current_channels = base_channels + 1

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

        # conv-bn-act-pool
        x = self.pool1(x)
        x = self.conv1(x)
        x = self.F_act(self.bn1(x))

        x = self.pool2(x)
        x = self.conv2(x)
        x = self.F_act(self.bn2(x))

        x = self.final_pool(x).reshape((N, -1))

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


class M5(nn.Module):
    def __init__(
        self,
        in_channels,
        out_dims,
        fc_stages,
        seq_length: int,
        use_age: str,
        base_channels=256,
        dropout: float = 0.3,
        base_pool: str = "max",
        final_pool: str = "average",
        activation: str = "relu",
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

        self.nn_act = get_activation_class(activation, class_name=self.__class__.__name__)
        self.F_act = get_activation_functional(activation, class_name=self.__class__.__name__)

        if base_pool == "average":
            self.base_pool = nn.AvgPool1d
        elif base_pool == "max":
            self.base_pool = nn.MaxPool1d

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
            stride_to_pool_ratio=0.7,
            class_name=self.__class__.__name__,
        )

        cf = conv_filter_list[0]
        self.pool1 = make_pool_or_not(self.base_pool, cf["pool"])
        self.conv1 = nn.Conv1d(
            in_channels,
            base_channels,
            kernel_size=cf["kernel_size"],
            padding=cf["kernel_size"] // 2,
            stride=cf["stride"],
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(base_channels)

        cf = conv_filter_list[1]
        self.pool2 = make_pool_or_not(self.base_pool, cf["pool"])
        self.conv2 = nn.Conv1d(
            base_channels,
            base_channels,
            kernel_size=cf["kernel_size"],
            padding=cf["kernel_size"] // 2,
            stride=cf["stride"],
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(base_channels)

        cf = conv_filter_list[2]
        self.pool3 = make_pool_or_not(self.base_pool, cf["pool"])
        self.conv3 = nn.Conv1d(
            base_channels,
            2 * base_channels,
            kernel_size=cf["kernel_size"],
            padding=cf["kernel_size"] // 2,
            stride=cf["stride"],
            bias=False,
        )
        self.bn3 = nn.BatchNorm1d(2 * base_channels)

        cf = conv_filter_list[3]
        self.pool4 = make_pool_or_not(self.base_pool, cf["pool"])
        self.conv4 = nn.Conv1d(
            2 * base_channels,
            2 * base_channels,
            kernel_size=cf["kernel_size"],
            padding=cf["kernel_size"] // 2,
            stride=cf["stride"],
            bias=False,
        )
        self.bn4 = nn.BatchNorm1d(2 * base_channels)

        cf = conv_filter_list[4]
        self.pool5 = make_pool_or_not(self.base_pool, cf["pool"])
        self.conv5 = nn.Conv1d(
            2 * base_channels,
            2 * base_channels,
            kernel_size=cf["kernel_size"],
            padding=cf["kernel_size"] // 2,
            stride=cf["stride"],
            bias=False,
        )
        self.bn5 = nn.BatchNorm1d(2 * base_channels)
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

        # conv-bn-act-pool
        x = self.pool1(x)
        x = self.conv1(x)
        x = self.F_act(self.bn1(x))

        x = self.pool2(x)
        x = self.conv2(x)
        x = self.F_act(self.bn2(x))

        x = self.pool3(x)
        x = self.conv3(x)
        x = self.F_act(self.bn3(x))

        x = self.pool4(x)
        x = self.conv4(x)
        x = self.F_act(self.bn4(x))

        x = self.pool5(x)
        x = self.conv5(x)
        x = self.F_act(self.bn5(x))

        x = self.final_pool(x).reshape((N, -1))

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
