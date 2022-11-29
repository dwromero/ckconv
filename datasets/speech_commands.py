"""
Adapted from https://github.com/patrick-kidger/NeuralCDE/blob/758d3a7134e3a691013e5cc6b7f68f277e9e6b69/experiments/datasets/speech_commands.py
"""
import os
import pathlib
import urllib.request
import tarfile
import torch
import torchaudio

from .utils import normalise_data, split_data, load_data, save_data, subsample


class SpeechCommands(torch.utils.data.TensorDataset):
    def __init__(
        self,
        partition: int,
        **kwargs,
    ):
        mfcc = kwargs["mfcc"]
        sr = kwargs["sr"]
        self.dropped_rate = kwargs["dropped_rate"]

        self.root = pathlib.Path("./data")
        base_loc = self.root / "SpeechCommands" / "processed_data"

        if mfcc:
            data_loc = base_loc / "mfcc"
        else:
            data_loc = base_loc / "raw"

            if self.dropped_rate != 0:
                data_loc = pathlib.Path(
                    str(data_loc) + "_dropped{}".format(self.dropped_rate)
                )

        if os.path.exists(data_loc):
            pass
        else:
            self.download()
            train_X, val_X, test_X, train_y, val_y, test_y = self._process_data(mfcc)

            if not os.path.exists(base_loc):
                os.mkdir(base_loc)
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

        # Subsample
        if not mfcc:
            X, y = subsample(X, y, sr)

        super(SpeechCommands, self).__init__(X, y)

    def download(self):
        root = self.root
        base_loc = root / "SpeechCommands"
        loc = base_loc / "speech_commands.tar.gz"
        if os.path.exists(loc):
            return
        if not os.path.exists(root):
            os.mkdir(root)
        if not os.path.exists(base_loc):
            os.mkdir(base_loc)
        urllib.request.urlretrieve(
            "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz", loc
        )  # TODO: Add progress bar
        with tarfile.open(loc, "r") as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, base_loc)

    def _process_data(self, mfcc):
        base_loc = self.root / "SpeechCommands"
        X = torch.empty(34975, 16000, 1)
        y = torch.empty(34975, dtype=torch.long)

        batch_index = 0
        y_index = 0
        for foldername in (
            "yes",
            "no",
            "up",
            "down",
            "left",
            "right",
            "on",
            "off",
            "stop",
            "go",
        ):
            loc = base_loc / foldername
            for filename in os.listdir(loc):
                audio, _ = torchaudio.load_wav(
                    loc / filename, channels_first=False, normalization=False
                )  # for forward compatbility if they fix it
                audio = (
                    audio / 2 ** 15
                )  # Normalization argument doesn't seem to work so we do it manually.

                # A few samples are shorter than the full length; for simplicity we discard them.
                if len(audio) != 16000:
                    continue

                X[batch_index] = audio
                y[batch_index] = y_index
                batch_index += 1
            y_index += 1
        assert batch_index == 34975, "batch_index is {}".format(batch_index)

        # If MFCC, then we compute these coefficients.
        if mfcc:
            X = torchaudio.transforms.MFCC(
                log_mels=True, n_mfcc=20, melkwargs=dict(n_fft=200, n_mels=64)
            )(X.squeeze(-1)).detach()
            # X is of shape (batch=34975, channels=20, length=161)
        else:
            X = X.unsqueeze(1).squeeze(-1)
            # X is of shape (batch=34975, channels=1, length=16000)

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
            # Get mask of possitions that are deleted
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
