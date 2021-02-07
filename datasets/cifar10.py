from torchvision import datasets, transforms


class CIFAR10(datasets.CIFAR10):  # TODO: Documentation
    def __init__(
        self,
        partition: str,
        **kwargs,
    ):

        root = "./data"
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        if partition == "train":
            train = True
        elif partition == "test":
            train = False
        else:
            raise NotImplementedError(
                "The dataset partition {} does not exist".format(partition)
            )

        super().__init__(root=root, train=train, transform=transform, download=True)
