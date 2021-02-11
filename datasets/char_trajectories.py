"""
Adapted from https://github.com/patrick-kidger/NeuralCDE/blob/758d3a7134e3a691013e5cc6b7f68f277e9e6b69/experiments/datasets/speech_commands.py
"""
import os
import pathlib
import urllib.request
import zipfile
from sktime.utils.data_io import load_from_tsfile_to_dataframe
import numpy as np
import torch
import collections as co
from .utils import normalise_data, split_data, load_data, save_data, subsample, pad


class CharTrajectories(torch.utils.data.TensorDataset):
    def __init__(
        self,
        partition: str,
        **kwargs,
    ):
        self.sampling_rate = kwargs["sr"]
        self.dropped_rate = kwargs["dropped_rate"]

        self.root = pathlib.Path("./data")
        data_loc = self.root / "UEA" / "CharacterTrajectories" / "processed_data"

        if self.dropped_rate != 0:
            data_loc = pathlib.Path(
                str(data_loc) + "_dropped{}".format(self.dropped_rate)
            )

        if os.path.exists(data_loc):
            pass
        else:
            self.download()
            if not os.path.exists(data_loc.parent):
                os.mkdir(data_loc.parent)
            if not os.path.exists(data_loc):
                os.mkdir(data_loc)
            train_X, val_X, test_X, train_y, val_y, test_y = self._process_data()
            save_data(
                data_loc,
                train_X=train_X,
                val_X=val_X,
                test_X=test_X,
                train_y=train_y,
                val_y=val_y,
                test_y=test_y,
            )

        X, y = self.load_data(data_loc, partition)

        X, y = subsample(X, y, self.sampling_rate)

        super(CharTrajectories, self).__init__(X, y)

    def download(self):
        root = self.root
        base_loc = root / "UEA"
        loc = base_loc / "Multivariate2018_ts.zip"
        if os.path.exists(loc):
            return
        if not os.path.exists(base_loc):
            os.mkdir(base_loc)
        urllib.request.urlretrieve(
            "http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip",
            str(loc),
        )

        with zipfile.ZipFile(loc, "r") as f:
            f.extractall(str(base_loc))

    def _process_data(self):
        root = self.root
        data_loc = (
            root
            / "UEA"
            / "Multivariate_ts"
            / "CharacterTrajectories"
            / "CharacterTrajectories"
        )

        train_X, train_y = load_from_tsfile_to_dataframe(str(data_loc) + "_TRAIN.ts")
        test_X, test_y = load_from_tsfile_to_dataframe(str(data_loc) + "_TEST.ts")
        train_X = train_X.to_numpy()
        test_X = test_X.to_numpy()
        X = np.concatenate((train_X, test_X), axis=0)
        y = np.concatenate((train_y, test_y), axis=0)

        lengths = torch.tensor([len(Xi[0]) for Xi in X])
        # final_index = lengths - 1
        maxlen = lengths.max()

        # Each channel is a pandas.core.series.Series object of length corresponding to the length of the time series
        X = torch.stack(
            [
                torch.stack([pad(channel, maxlen) for channel in batch], dim=0)
                for batch in X
            ],
            dim=0,
        )

        # Now fix the labels to be integers from 0 upwards
        targets = co.OrderedDict()
        counter = 0
        for yi in y:
            if yi not in targets:
                targets[yi] = counter
                counter += 1
        y = torch.tensor([targets[yi] for yi in y])

        # If dropped is different than zero, randomly drop that quantity of data from the dataset.
        if self.dropped_rate != 0:
            generator = torch.Generator().manual_seed(56789)
            X_removed = []
            for Xi in X:
                removed_points = (
                    torch.randperm(X.shape[-1], generator=generator)[
                        : int(X.shape[-1] * float(self.dropped_rate) / 100.0)
                    ]
                    .sort()
                    .values
                )
                Xi_removed = Xi.clone()
                Xi_removed[:, removed_points] = float("nan")
                X_removed.append(Xi_removed)
            X = torch.stack(X_removed, dim=0)

        # Normalize data
        X = normalise_data(X, y)

        # Once the data is normalized append times and mask values if required.
        if self.dropped_rate != 0:
            # Get mask of possitions that are deleted (Only first channel required
            # as all channels eliminated synchronously).
            mask_exists = (~torch.isnan(X[:, :1, :])).float()
            X = torch.where(~torch.isnan(X), X, torch.Tensor([0.0]))
            X = torch.cat([X, mask_exists], dim=1)

        train_X, val_X, test_X = split_data(X, y)
        train_y, val_y, test_y = split_data(y, y)

        return (
            train_X,
            val_X,
            test_X,
            train_y,
            val_y,
            test_y,
        )

    @staticmethod
    def load_data(data_loc, partition):

        tensors = load_data(data_loc)
        if partition == "train":
            X = tensors["train_X"]
            y = tensors["train_y"]
        elif partition == "val":
            X = tensors["val_X"]
            y = tensors["val_y"]
        elif partition == "test":
            X = tensors["test_X"]
            y = tensors["test_y"]
        else:
            raise NotImplementedError("the set {} is not implemented.".format(set))

        return X, y
