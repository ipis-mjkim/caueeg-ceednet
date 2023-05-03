from typing import Tuple

import torch
import torch.nn as nn

# __all__ = []


class LinearClassifier(nn.Module):
    def __init__(self, in_channels: int, out_dims: int, seq_length: int, use_age: str, dropout: float = 0.3, **kwargs):
        super().__init__()

        if use_age not in ["fc", "conv", "no"]:
            raise ValueError(f"{self.__class__.__name__}.__init__(use_age) " f"receives one of ['fc', 'conv', 'no'].")

        self.use_age = use_age
        if self.use_age == "conv":
            in_channels += 1

        self.sequence_length = seq_length
        current_dims = seq_length * in_channels
        if self.use_age in ["fc", "conv"]:
            current_dims = current_dims + 1

        self.output_length = current_dims
        self.linear = nn.Linear(current_dims, out_dims)
        self.dropout = nn.Dropout(p=dropout)

        self.reset_weights()

    def reset_weights(self):
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def get_output_length(self):
        return self.output_length

    def forward(self, x, age):
        N, C, L = x.size()

        x = x.reshape((N, -1))

        if self.use_age in ["conv", "fc"]:
            x = torch.cat((x, age.reshape(-1, 1)), dim=1)

        x = self.linear(x)
        x = self.dropout(x)

        return x


class LinearClassifier2D(nn.Module):
    def __init__(
        self, in_channels: int, out_dims: int, seq_len_2d: Tuple[int], use_age: str, dropout: float = 0.3, **kwargs
    ):
        super().__init__()

        if use_age not in ["fc", "conv", "no"]:
            raise ValueError(f"{self.__class__.__name__}.__init__(use_age) " f"receives one of ['fc', 'conv', 'no'].")

        self.use_age = use_age
        if self.use_age == "conv":
            in_channels += 1

        self.seq_len_2d = seq_len_2d
        current_dims = seq_len_2d[0] * seq_len_2d[1] * in_channels
        if self.use_age in ["fc", "conv"]:
            current_dims = current_dims + 1

        self.output_length = current_dims
        self.linear = nn.Linear(current_dims, out_dims)
        self.dropout = nn.Dropout(p=dropout)

        self.reset_weights()

    def reset_weights(self):
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def get_output_length(self):
        return self.output_length

    def forward(self, x, age):
        N, C, H, W = x.size()

        x = x.reshape((N, -1))

        if self.use_age in ["conv", "fc"]:
            x = torch.cat((x, age.reshape(-1, 1)), dim=1)

        x = self.linear(x)
        x = self.dropout(x)

        return x
