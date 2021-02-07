import copy
import torch
import torch.nn.functional as f
import torch.fft

from functools import partial
from typing import Tuple, Union, Iterable


def fft_conv_1d(
    signal: torch.Tensor,
    kernel: torch.Tensor,
    bias: torch.Tensor = None,
    padding: int = 0,
) -> torch.Tensor:
    """
    Args:
        signal: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Optional, Tensor) Bias tensor to add to the output.
        padding: (int) Number of zero samples to pad the input on the last dimension.

    Returns:
        (Tensor) Convolved tensor
    """
    # 1. Pad the input signal & kernel tensors
    signal = f.pad(signal, [padding, padding])
    kernel_padding = [0, signal.size(-1) - kernel.size(-1)]
    padded_kernel = f.pad(kernel, kernel_padding)  # .roll(shifts=-kernel.shape[-1]//2)

    # 2. Perform fourier convolution
    signal_fr = torch.fft.rfft(signal.double(), dim=-1)
    kernel_fr = torch.fft.rfft(padded_kernel.double(), dim=-1)

    # 3. Multiply the transformed matrices
    kernel_fr.imag *= -1.0
    output_fr = (signal_fr.unsqueeze(1) * kernel_fr.unsqueeze(0)).sum(2)
    # 3.1. Expand on the output and batch dim. respectively, do point-wise multiplication and sum.

    # 4. Compute inverse FFT, and remove extra padded values
    output = torch.fft.irfft(output_fr, dim=-1).float()
    output = output[:, :, : signal.size(-1) - kernel.size(-1) + 1]

    # 5. Optionally, add a bias term before returning.
    if bias is not None:
        output += bias.view(1, -1, 1)

    return output


# def complex_matmul(
#     a: torch.Tensor,
#     b: torch.Tensor,
# ) -> torch.Tensor:
#     """Multiplies two complex-valued tensors."""
#     # Scalar matrix multiplication of two tensors, over only the first two dimensions.
#     # Dimensions 3 and higher will have the same shape after multiplication.
#     scalar_matmul = partial(torch.einsum, "ab..., cb... -> ac...")
#
#     # Compute the real and imaginary parts independently, then manually insert them
#     # into the output Tensor.  This is fairly hacky but necessary for PyTorch 1.7.0,
#     # because Autograd is not enabled for complex matrix operations yet.  Not exactly
#     # idiomatic PyTorch code, but it should work for all future versions (>= 1.7.0).
#     real = scalar_matmul(a.real, b.real) - scalar_matmul(a.imag, b.imag)
#     imag = scalar_matmul(a.imag, b.real) + scalar_matmul(a.real, b.imag)
#     c = torch.zeros(real.shape, dtype=torch.complex128)
#     c.real, c.imag = real, imag
#
#     return c


def causal_conv(signal, kernel):
    if kernel.shape[-1] % 2 == 0:
        kernel = f.pad(kernel, [1, 0], value=0.0)
    pad = torch.nn.ConstantPad1d((kernel.shape[-1] - 1, 0), 0)
    pad_sig = pad(signal)
    return torch.nn.functional.conv1d(pad_sig, kernel, padding=0)


def causal_fftconv(
    signal: torch.Tensor,
    kernel: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """
    Args:
        signal: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Optional, Tensor) Bias tensor to add to the output.
        padding: (int) Number of zero samples to pad the input on the last dimension.

    Returns:
        (Tensor) Convolved tensor
    """
    # 1. Pad the input signal & kernel tensors
    if kernel.shape[-1] % 2 == 0:
        kernel = f.pad(kernel, [1, 0], value=0.0)
    pad = torch.nn.ConstantPad1d((kernel.shape[-1] - 1, 0), 0)
    pad_sig = pad(signal)
    kernel_padding = [0, pad_sig.size(-1) - kernel.size(-1)]
    padded_kernel = f.pad(kernel, kernel_padding)

    # 2. Perform fourier convolution
    signal_fr = torch.fft.rfft(pad_sig.double(), dim=-1)
    kernel_fr = torch.fft.rfft(padded_kernel.double(), dim=-1)

    # 3. Multiply the transformed matrices
    kernel_fr.imag *= -1.0
    output_fr = (signal_fr.unsqueeze(1) * kernel_fr.unsqueeze(0)).sum(2)
    # 3.1. Expand on the output and batch dim. respectively, do point-wise multiplication and sum.

    # 4. Compute inverse FFT, and remove extra padded values
    output = torch.fft.irfft(output_fr, dim=-1).float()
    output = output[:, :, : signal.shape[-1]]

    # 5. Optionally, add a bias term before returning.
    if bias is not None:
        output += bias.view(1, -1, 1)

    return output


if __name__ == "__main__":

    torch.manual_seed(1234)
    input_size = 28 * 28
    signal = torch.randn(3, 20, input_size)

    kernel_size = 500

    kernel1 = torch.nn.Parameter(torch.randn(20, 20, input_size))
    kernel2 = torch.nn.Parameter(torch.randn(20, 20, input_size))
    kernel3 = torch.nn.Parameter(torch.randn(20, 20, input_size))

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        y1 = torch.relu(causal_conv(signal, kernel1))
        y1 = y1 / torch.abs(y1).max()
        y2 = torch.relu(causal_conv(y1, kernel2))
        y2 = y2 / torch.abs(y2).max()
        y3 = causal_conv(y2, kernel3)

        y3.mean().backward()
    print(prof)

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        y1_fft = torch.relu(causal_fftconv(signal, kernel1))
        y1_fft = y1_fft / torch.abs(y1_fft).max()
        y2_fft = torch.relu(causal_fftconv(y1_fft, kernel2))
        y2_fft = y2_fft / torch.abs(y2_fft).max()
        y3_fft = causal_fftconv(y2_fft, kernel3)

        y3_fft.mean().backward()
    print(prof)

    y1 = torch.relu(causal_conv(signal, kernel1))
    y1 = y1 / torch.abs(y1).max()
    y2 = torch.relu(causal_conv(y1, kernel2))
    y2 = y2 / torch.abs(y2).max()
    y3 = causal_conv(y2, kernel3)

    y1_fft = torch.relu(causal_fftconv(signal, kernel1))
    y1_fft = y1_fft / torch.abs(y1_fft).max()
    y2_fft = torch.relu(causal_fftconv(y1_fft, kernel2))
    y2_fft = y2_fft / torch.abs(y2_fft).max()
    y3_fft = causal_fftconv(y2_fft, kernel3)

    abs_error = torch.abs(y1 - y1_fft)
    print(f"\nAbs Error Mean: {abs_error.mean():.3E}")
    print(f"Abs Error Std Dev: {abs_error.std():.3E}")

    abs_error = torch.abs(y2 - y2_fft)
    print(f"\nAbs Error Mean: {abs_error.mean():.3E}")
    print(f"Abs Error Std Dev: {abs_error.std():.3E}")

    abs_error = torch.abs(y3 - y3_fft)
    print(f"\nAbs Error Mean: {abs_error.mean():.3E}")
    print(f"Abs Error Std Dev: {abs_error.std():.3E}")

    y3.mean().backward()
    grad_kernel1_conv = copy.deepcopy(kernel1.grad.data)
    grad_kernel2_conv = copy.deepcopy(kernel2.grad.data)
    grad_kernel3_conv = copy.deepcopy(kernel3.grad.data)

    kernel1.grad.data.zero_()
    kernel2.grad.data.zero_()
    kernel3.grad.data.zero_()

    y3_fft.mean().backward()
    grad_kernel1_fftconv = copy.deepcopy(kernel1.grad.data)
    grad_kernel2_fftconv = copy.deepcopy(kernel2.grad.data)
    grad_kernel3_fftconv = copy.deepcopy(kernel3.grad.data)

    print("Gradients:")
    abs_error = torch.abs(grad_kernel1_conv - grad_kernel1_fftconv)
    print(f"\nAbs Error Mean: {abs_error.mean():.3E}")
    print(f"Abs Error Std Dev: {abs_error.std():.3E}")
    abs_error = torch.abs(grad_kernel2_conv - grad_kernel2_fftconv)
    print(f"\nAbs Error Mean: {abs_error.mean():.3E}")
    print(f"Abs Error Std Dev: {abs_error.std():.3E}")
    abs_error = torch.abs(grad_kernel3_conv - grad_kernel3_fftconv)
    print(f"\nAbs Error Mean: {abs_error.mean():.3E}")
    print(f"Abs Error Std Dev: {abs_error.std():.3E}")

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(y1_fft.detach().numpy()[0, 0, :])
    plt.plot(y1.detach().numpy()[0, 0, :])
    plt.show()

    plt.figure()
    plt.plot(y2_fft.detach().numpy()[0, -5, :])
    plt.plot(y2.detach().numpy()[0, -5, :])
    plt.show()

    plt.figure()
    plt.plot(y3_fft.detach().numpy()[0, -1, :])
    plt.plot(y3.detach().numpy()[0, -1, :])
    plt.show()

    # plt.figure()
    # plt.plot(y1_fft.detach().numpy()[0, 0, :] - y1.detach().numpy()[0, 0, :])
    # plt.show()
    #
    # plt.figure()
    # plt.plot(y2_fft.detach().numpy()[0, -5, :] - y2.detach().numpy()[0, -5, :])
    # plt.show()
    #
    # plt.figure()
    # plt.plot(y3_fft.detach().numpy()[0, -1, :] - y3.detach().numpy()[0, -1, :])
    # plt.show()
    #
    # plt.figure()
    # plt.plot(grad_kernel1_conv.detach().numpy()[0, 0, :] - grad_kernel1_fftconv.detach().numpy()[0, 0, :])
    # plt.show()
    #
    # plt.figure()
    # plt.plot(grad_kernel2_conv.detach().numpy()[0, 0, :] - grad_kernel2_fftconv.detach().numpy()[0, 0, :])
    # plt.show()
    #
    # plt.figure()
    # plt.plot(grad_kernel3_conv.detach().numpy()[0, 0, :] - grad_kernel3_fftconv.detach().numpy()[0, 0, :])
    # plt.show()
