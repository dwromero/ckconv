import torch
import ckconv.nn


class CKBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernelnet_hidden_channels: int,
        kernelnet_activation_function: str,
        kernelnet_norm_type: str,
        dim_linear: int,
        bias: bool,
        omega_0: bool,
        dropout: float,
        weight_dropout: float,
    ):
        """
        Creates a Residual Block with CKConvs as:
        ( Follows the Residual Block of Bai et. al., 2017 )

        input
         | ---------------|
         CKConv           |
         LayerNorm        |
         ReLU             |
         DropOut          |
         |                |
         CKConv           |
         LayerNorm        |
         ReLU             |
         DropOut          |
         + <--------------|
         |
         ReLU
         |
         output


        :param in_channels:  Number of channels in the input signal
        :param out_channels:  Number of output (and hidden) channels of the block.
        :param kernelnet_hidden_channels: Number of hidden units in the KernelNets of the CKConvs.
        :param kernelnet_activation_function: Activation function used in the KernelNets of the CKConvs.
        :param kernelnet_norm_type: Normalization type used in the KernelNets of the CKConvs (only for non-Sine KernelNets).
        :param dim_linear:  Spatial dimension of the input, e.g., for audio = 1, images = 2 (only 1 suported).
        :param bias:  If True, adds a learnable bias to the output.
        :param omega_0: Value of the omega_0 value of the KernelNets. (only for non-Sine KernelNets).
        :param dropout: Dropout rate of the block
        :param weight_dropout: Dropout rate applied to the sampled convolutional kernels.
        """
        super().__init__()

        # CKConv layers
        self.cconv1 = ckconv.nn.CKConv(
            in_channels,
            out_channels,
            kernelnet_hidden_channels,
            kernelnet_activation_function,
            kernelnet_norm_type,
            dim_linear,
            bias,
            omega_0,
            weight_dropout,
        )
        self.cconv2 = ckconv.nn.CKConv(
            out_channels,
            out_channels,
            kernelnet_hidden_channels,
            kernelnet_activation_function,
            kernelnet_norm_type,
            dim_linear,
            bias,
            omega_0,
            weight_dropout,
        )
        # Norm layers
        self.norm1 = ckconv.nn.LayerNorm(out_channels)
        self.norm2 = ckconv.nn.LayerNorm(out_channels)

        # Dropout
        self.dp = torch.nn.Dropout(dropout)

        shortcut = []
        if in_channels != out_channels:
            shortcut.append(ckconv.nn.Linear1d(in_channels, out_channels))
        self.shortcut = torch.nn.Sequential(*shortcut)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.dp(torch.relu(self.norm1(self.cconv1(x))))
        out = torch.relu(self.dp(torch.relu(self.norm2(self.cconv2(out)))) + shortcut)
        return out
