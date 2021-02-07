import torch
import torch.fft
import torch.nn
import ckconv.nn.functional as ckconv_f


class CausalConv1d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool,
        weight_dropout: float,
    ):
        """
        Applies a 1D convolution over an input signal of input_channels.

        :param in_channels: Number of channels in the input signal
        :param out_channels: Number of channels produced by the convolution
        :param kernel_size: Size of the convolving kernel
        :param bias: If True, adds a learnable bias to the output.
        :param weight_dropout: Dropout rate applied to the sampled convolutional kernels.
        """
        super().__init__()
        self.weight_dropout = weight_dropout
        self.w_dropout = torch.nn.Dropout(p=weight_dropout)

        self.weight = torch.nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size)
        )
        self.weight.data.normal_(0, 0.01)
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
            self.bias.data.fill_(value=0.0)

    def forward(self, x):

        # Dropout weight values if required
        if self.weight_dropout != 0.0:
            weight = self.w_dropout(self.weight)
        else:
            weight = self.weight

        # Perform causal convolution
        return ckconv_f.causal_fftconv(
            x,
            weight,
            self.bias,
            double_precision=False,
        )
