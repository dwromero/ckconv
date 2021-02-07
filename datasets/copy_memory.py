import numpy as np
import torch
from torch.autograd import Variable


def data_generator(T, mem_length, b_size):
    """
    Generate data for the copying memory task
    :param T: The total blank time length
    :param mem_length: The length of the memory to be recalled
    :param b_size: The batch size
    :return: Input and target data tensor
    """
    seq = torch.from_numpy(np.random.randint(1, 9, size=(b_size, mem_length))).float()
    zeros = torch.zeros((b_size, T))
    marker = 9 * torch.ones((b_size, mem_length + 1))
    placeholders = torch.zeros((b_size, mem_length))

    X = torch.cat((seq, zeros[:, :-1], marker), 1)
    Y = torch.cat((placeholders, zeros, seq), 1).long()

    return X, Y


class CopyMemory(torch.utils.data.TensorDataset):  # TODO: Documentation
    def __init__(
        self,
        partition: str,
        seq_length: int,
        **kwargs,
    ):
        memory_size = kwargs["memory_size"]

        blank_length = seq_length
        # total_seq_length = blank_length + (
        #     2 * memory_size
        # )  # Total size of sequence is blank space + 2 times the size of the string to memorize.

        if partition == "train":
            dataset_size = 10000
            X, Y = data_generator(blank_length, memory_size, dataset_size)
        elif partition == "test":
            dataset_size = 1000
            X, Y = data_generator(blank_length, memory_size, dataset_size)
        else:
            raise NotImplementedError(
                "The dataset partition {} does not exist".format(partition)
            )

        super(CopyMemory, self).__init__(X, Y)
