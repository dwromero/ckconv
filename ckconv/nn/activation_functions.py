# torch
import torch
from ckconv.nn.misc import Expression


def Swish():
    """
    out = x * sigmoid(x)
    """
    return Expression(lambda x: x * torch.sigmoid(x))


def Sine():
    """
    out = sin(x)
    """
    return Expression(lambda x: torch.sin(x))
