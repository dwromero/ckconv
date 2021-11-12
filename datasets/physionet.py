"""
Adapted from https://github.com/patrick-kidger/NeuralCDE/blob/758d3a7134e3a691013e5cc6b7f68f277e9e6b69/experiments/datasets/speech_commands.py
"""
import os
import pathlib
import urllib.request
import zipfile
import torch
import csv
import math
import torchaudio

from .utils import normalise_data, split_data, load_data, save_data, subsample


class PhysioNet(torch.utils.data.TensorDataset):
    def __init__(
        self,
        partition: int,
        **kwargs,
    ):
        self.root = pathlib.Path("./data")
        self.base_loc = self.root / "sepsis"
        data_loc = self.base_loc / "preprocessed_data"

        if os.path.exists(self.base_loc):
            pass
        else:
            self.download()
            train_X, val_X, test_X, train_y, val_y, test_y = self._process_data()
            if not os.path.exists(data_loc):
                os.mkdir(data_loc)
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

        super(PhysioNet, self).__init__(X, y)


    def download(self):
        loc_Azip = self.base_loc / 'training_setA.zip'
        loc_Bzip = self.base_loc / 'training_setB.zip'

        if not os.path.exists(loc_Azip):
            if not os.path.exists(self.base_loc):
                os.mkdir(self.base_loc)
            urllib.request.urlretrieve('https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip',
                                       str(loc_Azip))
            urllib.request.urlretrieve('https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip',
                                       str(loc_Bzip))

            with zipfile.ZipFile(loc_Azip, 'r') as f:
                f.extractall(str(self.base_loc))
            with zipfile.ZipFile(loc_Bzip, 'r') as f:
                f.extractall(str(self.base_loc))
            for folder in ('training', 'training_setB'):
                for filename in os.listdir(self.base_loc / folder):
                    if os.path.exists(self.base_loc / filename):
                        raise RuntimeError
                    os.rename(self.base_loc / folder / filename, self.base_loc / filename)

    def _process_data(self):
        X_times = []
        X_static = []
        y = []
        for filename in os.listdir(self.base_loc):
            if filename.endswith('.psv'):
                with open(self.base_loc / filename) as file:
                    time = []
                    label = 0.0
                    reader = csv.reader(file, delimiter='|')
                    reader = iter(reader)
                    next(reader)  # first line is headings
                    prev_iculos = 0
                    for line in reader:
                        assert len(line) == 41
                        *time_values, age, gender, unit1, unit2, hospadmtime, iculos, sepsislabel = line
                        iculos = int(iculos)
                        if iculos > 72:  # keep at most the first three days
                            break
                        for iculos_ in range(prev_iculos + 1, iculos):
                            time.append([float('nan') for value in time_values])
                        prev_iculos = iculos
                        time.append([float(value) for value in time_values])
                        label = max(label, float(sepsislabel))
                    unit1 = float(unit1)
                    unit2 = float(unit2)
                    unit1_obs = not math.isnan(unit1)
                    unit2_obs = not math.isnan(unit2)
                    if not unit1_obs:
                        unit1 = 0.
                    if not unit2_obs:
                        unit2 = 0.
                    hospadmtime = float(hospadmtime)
                    if math.isnan(hospadmtime):
                        hospadmtime = 0.  # this only happens for one record
                    static = [float(age), float(gender), unit1, unit2, hospadmtime]
                    static += [unit1_obs, unit2_obs]
                    if len(time) > 2:
                        X_times.append(time)
                        X_static.append(static)
                        y.append(label)
        final_indices = []
        for time in X_times:
            final_indices.append(len(time) - 1)
        maxlen = max(final_indices) + 1
        for time in X_times:
            for _ in range(maxlen - len(time)):
                time.append([float('nan') for value in time_values])

        X_times = torch.tensor(X_times)
        X_static = torch.tensor(X_static)
        y = torch.tensor(y).long()

        # Normalize data
        X_times = normalise_data(X_times, y)

        # Append extra channels together.
        augmented_X_times = []
        intensity = ~torch.isnan(X_times)  # of size (batch, stream, channels)
        intensity = intensity.to(X_times.dtype).cumsum(dim=1)
        augmented_X_times.append(intensity)
        augmented_X_times.append(X_times)
        X_times = torch.cat(augmented_X_times, dim=2)

        X_times = torch.where(~torch.isnan(X_times), X_times, torch.Tensor([0.0]))

        train_X_times, val_X_times, test_X = split_data(X_times, y)
        train_y, val_y, test_y = split_data(y, y)

        X_static_ = X_static[:, :-2]
        X_static_ = normalise_data(X_static_, y)
        X_static = torch.cat([X_static_, X_static[:, -2:]], dim=1).unsqueeze(1).repeat(1, X_times.shape[1], 1)

        train_X_static, val_X_static, test_X_static = split_data(X_static, y)

        # Concatenate
        train_X = torch.cat([train_X_times,  train_X_static], dim=-1).transpose(-2, -1)
        val_X = torch.cat([val_X_times, val_X_static], dim=-1).transpose(-2, -1)
        test_X = torch.cat([val_X_times, val_X_static], dim=-1).transpose(-2, -1)

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







