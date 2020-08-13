from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F


class _LayerNorm(nn.Module):
    def __init__(self, num_features, img_size):
        """
        Normalizes over the entire image and scales + weights for each feature
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            (num_features, img_size, img_size), elementwise_affine=False, eps=1e-12
        )
        self.weight = torch.nn.Parameter(
            torch.ones(num_features).float().unsqueeze(-1).unsqueeze(-1),
            requires_grad=True,
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(num_features).float().unsqueeze(-1).unsqueeze(-1),
            requires_grad=True,
        )

    def forward(self, x):
        out = self.layer_norm(x)
        out = out * self.weight + self.bias
        return out


class _SamePad(nn.Module):
    """
    Pads equivalent to the behavior of tensorflow "SAME"
    """

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 2 and x.shape[2] % 2 == 0:
            return F.pad(x, (0, 1, 0, 1))
        return F.pad(x, (1, 1, 1, 1))


def _conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    out_size=None,
    activate=True,
    dropout=0.0,
):
    layers = OrderedDict()
    layers["pad"] = _SamePad(stride)
    layers["conv"] = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    if activate:
        if out_size is None:
            raise ValueError("Must provide out_size if activate is True")
        layers["relu"] = nn.LeakyReLU(0.2)
        layers["norm"] = _LayerNorm(out_channels, out_size)

    if dropout > 0.0:
        layers["dropout"] = nn.Dropout(dropout)
    return nn.Sequential(layers)


class _EncoderBlock(nn.Module):
    def __init__(
        self,
        pre_channels,
        in_channels,
        out_channels,
        num_layers,
        out_size,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.pre_conv = _conv2d(
            in_channels=pre_channels,
            out_channels=pre_channels,
            kernel_size=3,
            stride=2,
            activate=False,
        )

        self.conv0 = _conv2d(
            in_channels=in_channels + pre_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            out_size=out_size,
        )
        total_channels = in_channels + out_channels
        for i in range(1, num_layers):
            self.add_module(
                "conv%d" % i,
                _conv2d(
                    in_channels=total_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    out_size=out_size,
                ),
            )
            total_channels += out_channels
        self.add_module(
            "conv%d" % num_layers,
            _conv2d(
                in_channels=total_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                out_size=(out_size + 1) // 2,
                dropout=dropout_rate,
            ),
        )

    def forward(self, inp):
        pre_input, x = inp
        pre_input = self.pre_conv(pre_input)
        out = self.conv0(torch.cat([x, pre_input], 1))

        all_outputs = [x, out]
        for i in range(1, self.num_layers + 1):
            input_features = torch.cat(
                [all_outputs[-1], all_outputs[-2]] + all_outputs[:-2], 1
            )
            module = self._modules["conv%d" % i]
            out = module(input_features)
            all_outputs.append(out)
        return all_outputs[-2], all_outputs[-1]


class Discriminator(nn.Module):
    def __init__(self, dim, channels, dropout_rate=0.0, z_dim=100):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.channels = channels
        self.layer_sizes = [64, 64, 128, 128]
        self.num_inner_layers = 5

        # Number of times dimension is halved
        self.depth = len(self.layer_sizes)

        # dimension at each level of U-net
        self.dim_arr = [dim]
        for i in range(self.depth):
            self.dim_arr.append((self.dim_arr[-1] + 1) // 2)

        # Encoders
        self.encode0 = _conv2d(
            in_channels=self.channels,
            out_channels=self.layer_sizes[0],
            kernel_size=3,
            stride=2,
            out_size=self.dim_arr[1],
        )
        for i in range(1, self.depth):
            self.add_module(
                "encode%d" % i,
                _EncoderBlock(
                    pre_channels=self.channels if i == 1 else self.layer_sizes[i - 1],
                    in_channels=self.layer_sizes[i - 1],
                    out_channels=self.layer_sizes[i],
                    num_layers=self.num_inner_layers,
                    out_size=self.dim_arr[i],
                    dropout_rate=dropout_rate,
                ),
            )
        self.dense1 = nn.Linear(self.layer_sizes[-1], 1024)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dense2 = nn.Linear(self.layer_sizes[-1] * self.dim_arr[-1] ** 2 + 1024, 1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        out = [x, self.encode0(x)]
        for i in range(1, len(self.layer_sizes)):
            out = self._modules["encode%d" % i](out)
        out = out[1]

        out_mean = out.mean([2, 3])
        out_flat = torch.flatten(out, 1)

        out = self.dense1(out_mean)
        out = self.leaky_relu(out)
        out = self.dense2(torch.cat([out, out_flat], 1))

        return out
