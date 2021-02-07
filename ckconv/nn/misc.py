import torch


# From LieConv
class Expression(torch.nn.Module):
    def __init__(self, func):
        """
        Creates a torch.nn.Module that applies the function func.

        :param func: lambda function
        """
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def Multiply(
    omega_0: float,
):
    """
    out = omega_0 * x
    """
    return Expression(lambda x: omega_0 * x)


class MultiplyLearned(torch.nn.Module):
    def __init__(
        self,
        omega_0: float,
    ):
        """
        out = omega_0 * x, with a learned omega_0
        """
        super().__init__()
        self.omega_0 = torch.nn.Parameter(torch.Tensor(1))
        with torch.no_grad():
            self.omega_0.fill_(omega_0)

    def forward(self, x):
        return 100 * self.omega_0 * x
