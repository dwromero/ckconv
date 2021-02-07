import torch
import ckconv.nn
from torch.nn.utils import weight_norm


class CausalConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        bias: bool,
        dropout: float,
        weight_dropout: float,
    ):
        super().__init__()

        # CKConv layers
        self.conv1 = weight_norm(
            ckconv.nn.CausalConv1d(
                in_channels,
                hidden_channels,
                kernel_size,
                bias,
                weight_dropout,
            )
        )
        self.conv2 = weight_norm(
            ckconv.nn.CausalConv1d(
                hidden_channels,
                hidden_channels,
                kernel_size,
                bias,
                weight_dropout,
            )
        )
        # Norm layers
        # self.norm1 = ckconv.nn.LayerNorm(hidden_channels)
        # self.norm2 = ckconv.nn.LayerNorm(hidden_channels)

        # Dropout
        self.dp = torch.nn.Dropout(dropout)

        shortcut = []
        if in_channels != hidden_channels:
            shortcut.append(ckconv.nn.Linear1d(in_channels, hidden_channels))
        self.shortcut = torch.nn.Sequential(*shortcut)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.dp(torch.relu(self.conv1(x)))
        out = torch.relu(self.dp(torch.relu(self.conv2(out))) + shortcut)
        return out


# Big Filter CNN
class BFCNN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        num_blocks: int,
        bias: bool,
        dropout: float,
        weight_dropout: float,
        pool: bool,
    ):
        super(BFCNN, self).__init__()

        blocks = []
        for i in range(num_blocks):
            block_in_channels = in_channels if i == 0 else hidden_channels
            # dp = dropout if i != (num_blocks - 1) else 0.0
            blocks.append(
                CausalConvBlock(
                    block_in_channels,
                    hidden_channels,
                    kernel_size,
                    bias,
                    dropout,
                    weight_dropout,
                )
            )
            if pool:
                blocks.append(torch.nn.MaxPool1d(kernel_size=2))
        self.backbone = torch.nn.Sequential(*blocks)

    def forward(self, x):
        return self.backbone(x)


class seqImg_BFCNN(BFCNN):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        num_blocks: int,
        bias: bool,
        dropout: float,
        weight_dropout: float,
        pool: bool,
    ):
        super().__init__(
            in_channels,
            hidden_channels,
            kernel_size,
            num_blocks,
            bias,
            dropout,
            weight_dropout,
            pool,
        )

        self.finallyr = torch.nn.Linear(
            in_features=hidden_channels, out_features=out_channels
        )
        # Initialize finallyr
        self.finallyr.weight.data.normal_(
            mean=0.0,
            std=0.01,
        )
        self.finallyr.bias.data.fill_(value=0.0)

    def forward(self, x):
        out = self.backbone(x)
        out = self.finallyr(out[:, :, -1])
        return out
