from torchvision import datasets, transforms


class MNIST(datasets.MNIST):
    def __init__(
        self,
        partition: int,
        **kwargs,
    ):
        if "root" in kwargs:
            root = kwargs["root"]
        else:
            root = "./data"

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        if partition == "train":
            train = True
        elif partition == "test":
            train = False
        else:
            raise NotImplementedError(
                "The dataset partition {} does not exist".format(partition)
            )

        super(MNIST, self).__init__(
            root=root, train=train, transform=transform, download=True
        )
