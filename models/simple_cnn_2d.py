import numpy as np
import torch
import torch.nn as nn

from .activation import get_activation_class
from .activation import get_activation_functional


class IeracitanoCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_dims: int,
        fc_stages: int,
        seq_length: int,
        use_age: str,
        base_channels: int = 16,
        base_pool: str = "max",
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__()

        if use_age != "no":
            raise ValueError(f"{self.__class__.__name__}.__init__(use_age) accepts for only 'no'.")

        if fc_stages != 1:
            raise ValueError(f"{self.__class__.__name__}.__init__(fc_stages) accepts for only 1.")

        self.sequence_length = seq_length
        self.fc_stages = fc_stages

        self.nn_act = get_activation_class(activation, class_name=self.__class__.__name__)
        self.F_act = get_activation_functional(activation, class_name=self.__class__.__name__)

        if base_pool == "average":
            self.base_pool = nn.AvgPool2d
        elif base_pool == "max":
            self.base_pool = nn.MaxPool2d

        self.conv1 = nn.Conv2d(1, base_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.pool1 = self.base_pool((2, 2))

        fc_stage = []

        self.output_length = 160 // 2
        current_dim = 11520
        fc_stage.append(nn.Linear(current_dim, 300))
        fc_stage.append(nn.BatchNorm1d(300))
        fc_stage.append(self.nn_act())

        current_dim = 300
        fc_stage.append(nn.Linear(current_dim, out_dims))
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

        # preprocessing stage
        x = torch.abs(torch.fft.rfft(x)[:, :, :160])
        x = x.reshape(N, 1, C, 160)

        # conv-bn-act-pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.F_act(x)
        x = self.pool1(x)
        x = x.reshape(N, -1)

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
