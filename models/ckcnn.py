import torch
import ckconv.nn


class CKCNN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_blocks: int,
        kernelnet_hidden_channels: int,
        kernelnet_activation_function: str,
        kernelnet_norm_type: str,
        dim_linear: int,
        bias: bool,
        omega_0: bool,
        dropout: float,
        weight_dropout: float,
        pool: bool,  # Always False in our experiments.
        out_channels=None,
    ):
        super(CKCNN, self).__init__()

        blocks = []
        for i in range(num_blocks):
            block_in_channels = in_channels if i == 0 else hidden_channels
            block_out_channels = hidden_channels
            if i == num_blocks-1 and out_channels is not None:
                block_out_channels = out_channels
            blocks.append(
                ckconv.nn.CKBlock(
                    block_in_channels,
                    block_out_channels,
                    kernelnet_hidden_channels,
                    kernelnet_activation_function,
                    kernelnet_norm_type,
                    dim_linear,
                    bias,
                    omega_0,
                    dropout,
                    weight_dropout,
                )
            )
            if pool:
                blocks.append(torch.nn.MaxPool1d(kernel_size=2))
        self.backbone = torch.nn.Sequential(*blocks)

    def forward(self, x):
        return self.backbone(x)


class CopyMemory_CKCNN(CKCNN):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_blocks: int,
        kernelnet_hidden_channels: int,
        kernelnet_activation_function: str,
        kernelnet_norm_type: str,
        dim_linear: int,
        bias: bool,
        omega_0: bool,
        dropout: float,
        weight_dropout: float,
        pool: bool,
    ):
        super().__init__(
            in_channels,
            hidden_channels,
            num_blocks,
            kernelnet_hidden_channels,
            kernelnet_activation_function,
            kernelnet_norm_type,
            dim_linear,
            bias,
            omega_0,
            dropout,
            weight_dropout,
            pool,
        )

        self.finallyr = torch.nn.Linear(in_features=hidden_channels, out_features=10)
        # Initialize finallyr
        self.finallyr.weight.data.normal_(
            mean=0.0,
            std=0.01,
        )
        self.finallyr.bias.data.fill_(value=0.0)

    def forward(self, x):
        out = self.backbone(x)
        out = self.finallyr(out.transpose(1, 2))
        return out


class AddProblem_CKCNN(CKCNN):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_blocks: int,
        kernelnet_hidden_channels: int,
        kernelnet_activation_function: str,
        kernelnet_norm_type: str,
        dim_linear: int,
        bias: bool,
        omega_0: bool,
        dropout: float,
        weight_dropout: float,
        pool: bool,
    ):
        super().__init__(
            in_channels,
            hidden_channels,
            num_blocks,
            kernelnet_hidden_channels,
            kernelnet_activation_function,
            kernelnet_norm_type,
            dim_linear,
            bias,
            omega_0,
            dropout,
            weight_dropout,
            pool,
        )

        self.finallyr = torch.nn.Linear(in_features=hidden_channels, out_features=1)
        # Initialize finallyr
        self.finallyr.weight.data.normal_(
            mean=0.0,
            std=0.01,
        )
        self.finallyr.bias.data.fill_(value=0.0)

    def forward(self, x):
        out = self.backbone(x)
        out = self.finallyr(out[:, :, -1])
        return out


class seqImg_CKCNN(CKCNN):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_blocks: int,
        kernelnet_hidden_channels: int,
        kernelnet_activation_function: str,
        kernelnet_norm_type: str,
        dim_linear: int,
        bias: bool,
        omega_0: bool,
        dropout: float,
        weight_dropout: float,
        pool: bool,
    ):
        super().__init__(
            in_channels,
            hidden_channels,
            num_blocks,
            kernelnet_hidden_channels,
            kernelnet_activation_function,
            kernelnet_norm_type,
            dim_linear,
            bias,
            omega_0,
            dropout,
            weight_dropout,
            pool,
        )

        self.finallyr = torch.nn.Linear(
            in_features=hidden_channels, out_features=out_channels
        )
        # Initialize finallyr
        self.finallyr.weight.data.normal_(
            mean=0.0,
            std=0.01,
        )
        self.finallyr.bias.data.fill_(value=0.0)

    def forward(self, x):
        out = self.backbone(x)
        out = self.finallyr(out[:, :, -1])
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out


class seqText_CKCNN(CKCNN):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_channels: int,
            num_blocks: int,
            kernelnet_hidden_channels: int,
            kernelnet_activation_function: str,
            kernelnet_norm_type: str,
            dim_linear: int,
            bias: bool,
            omega_0: bool,
            dropout: float,
            weight_dropout: float,
            pool: bool,
            # vocab_size: int,
            emb_dropout: float,
            tied_weights=True,
    ):
        super().__init__(
            in_channels,
            hidden_channels,
            num_blocks,
            kernelnet_hidden_channels,
            kernelnet_activation_function,
            kernelnet_norm_type,
            dim_linear,
            bias,
            omega_0,
            dropout,
            weight_dropout,
            pool,
            out_channels=in_channels
        )
        self.encoder = torch.nn.Embedding(out_channels, in_channels)
        self.drop = torch.nn.Dropout(emb_dropout)

        self.finallyr = torch.nn.Linear(
            in_features=in_channels, out_features=out_channels
        )
        self.train_len = None
        self.init_weights()
        if tied_weights:
            self.finallyr.weight = self.encoder.weight
            print("Weight tied")

    def init_weights(self):
        self.encoder.weight.data.normal_(0, 0.01)
        self.finallyr.weight.data.normal_(0, 0.01)
        self.finallyr.bias.data.fill_(value=0.0)

    def forward(self, x, return_emb=False):
        emb = self.prepare_input(x)
        y1 = self.backbone(emb)[:, :, self.train_len-x.shape[-1]:]  # MB x hid_size x seq_len
        out = self.finallyr(y1.transpose(1, 2))  # MB x seq_len x voc_size
        if return_emb:
            return out, y1.transpose(1, 2)
        return out

    def prepare_input(self, x):
        if self.train_len is None:
            self.train_len = x.shape[-1]
        emb = self.drop(self.encoder(x)).transpose(1, 2)  # MB x emb_size x seq_len
        if emb.shape[-1] < self.train_len:
            emb = torch.nn.functional.pad(emb, (self.train_len - emb.shape[-1], 0))
        return emb
