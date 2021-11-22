"""
Adapted from https://github.com/locuslab/TCN/blob/master/TCN/
"""
import pickle
from collections import Counter
import os
import numpy as np
import torch
import pathlib
from .utils import load_data, save_data
import observations


class PennTreeBankChar(torch.utils.data.Dataset):
    def __init__(
            self,
            partition: int,
            seq_length: int,
            valid_seq_len: int,
            batch_size: int,
            **kwargs,
    ):
        self.seq_len = seq_length
        self.valid_seq_len = valid_seq_len
        self.batch_size = batch_size
        self.root = pathlib.Path("./data")
        self.base_loc = self.root / "penn"
        data_loc = self.base_loc / "preprocessed_data_char"

        if os.path.exists(data_loc):
            self.dictionary = pickle.load(open(str(data_loc / 'dictionary_char'), 'rb'))
        else:
            train, valid, test = self._process_data()

            if not os.path.exists(data_loc):
                os.mkdir(data_loc)
            pickle.dump(self.dictionary, open(str(data_loc / 'dictionary_char'), 'wb'))
            save_data(
                data_loc,
                train=train,
                valid=valid,
                test=test,
            )

        self.X, self.y = self.load_data(data_loc, partition)
        if partition == 'train':
            self.sampler = SequentialBatchSampler(self)
        else:
            self.sampler = SequentialBatchSampler(self, shuffle=False)
        super(PennTreeBankChar, self).__init__()

    def __getitem__(self, ind):
        b = ind // len(self.X[0])
        i = ind - b * len(self.X[0])
        return self.X[b][i], self.y[b][i]

    def __len__(self):
        return len(self.X[0]) * len(self.X)

    def create_seq(self, data, batch_size):
        nbatch = data.size(0) // batch_size
        data = data.narrow(0, 0, nbatch * batch_size).view(batch_size, -1)  ## crop tail
        x = []
        y = []
        L = data.shape[1]
        for i in range(0, L-1, self.valid_seq_len):
            if i + self.seq_len - self.valid_seq_len >= L - 1:
                continue
            end = min(i + self.seq_len, L - 1)
            x.append(data[:, i: end].contiguous())
            y.append(data[:, i+1: end+1].contiguous())
        return x, y

    def _process_data(self):
        self.dictionary = Dictionary()
        train, test, valid = getattr(observations, 'ptb')(self.base_loc)
        for c in train + ' ' + test + '' + valid:
            self.dictionary.add_word(c)
        self.dictionary.prep_dict()


        train = self._char_to_tensor(train)
        valid = self._char_to_tensor(valid)
        test = self._char_to_tensor(test)
        return train, valid, test

    def _char_to_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for i in range(len(string)):
            tensor[i] = self.dictionary.char2idx[string[i]]
        return tensor

    def load_data(self, data_loc, partition):
        tensors = load_data(data_loc)
        if partition == "train":
            data = tensors["train"]
        elif partition == "val":
            data = tensors["valid"]
        elif partition == "test":
            data = tensors["test"]
        else:
            raise NotImplementedError("the set {} is not implemented.".format(set))
        X, y = self.create_seq(data, self.batch_size)
        return X, y


class Dictionary(object):
    def __init__(self):
        self.char2idx = {}
        self.idx2char = []
        self.counter = Counter()

    def add_word(self, word):
        self.counter[word] += 1

    def prep_dict(self):
        for char in self.counter:
            if char not in self.char2idx:
                self.idx2char.append(char)
                self.char2idx[char] = len(self.idx2char) - 1

    def __len__(self):
        return len(self.idx2char)


class SequentialBatchSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, shuffle=True):
        super(SequentialBatchSampler, self).__init__(data_source)
        self.X = data_source.X
        if shuffle:
            self.sampler = torch.utils.data.SubsetRandomSampler(np.arange(len(self.X)))
        else:
            self.sampler = np.arange(len(self.X))
        self.batch_size = self.X[0].shape[0]

    def __iter__(self):
        for idx in self.sampler:
            batch = [idx * self.batch_size + j for j in range(self.batch_size)]
            yield batch

    def __len__(self):
        return len(self.X)
