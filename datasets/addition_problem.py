import torch
import numpy as np

from typing import Tuple


def data_generator(
    N: int,
    seq_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        N: # of data samples in the set
        seq_length: Length of the adding problem data
    """
    X_num = torch.rand([N, 1, seq_length])
    X_mask = torch.zeros([N, 1, seq_length])
    Y = torch.zeros([N, 1])
    for i in range(N):
        positions = np.random.choice(seq_length, size=2, replace=False)
        X_mask[i, 0, positions[0]] = 1
        X_mask[i, 0, positions[1]] = 1
        Y[i, 0] = X_num[i, 0, positions[0]] + X_num[i, 0, positions[1]]
    X = torch.cat((X_num, X_mask), dim=1)
    return X, Y


class AdditionProblem(torch.utils.data.TensorDataset):
    def __init__(
        self,
        partition: str,
        seq_length: int,
        **kwargs,
    ):
        """
        Creates a Addition Problem dataset.
        """
        if partition == "train":
            X, Y = data_generator(50000, seq_length)
        elif partition == "test":
            X, Y = data_generator(1000, seq_length)
        else:
            raise NotImplementedError(
                "The dataset partition {} does not exist".format(partition)
            )

        super(AdditionProblem, self).__init__(X, Y)
