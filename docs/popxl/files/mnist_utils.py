# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import time
from contextlib import ContextDecorator
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import torch
import torchvision
import urllib.request
import pathlib


@dataclass
class Timer(ContextDecorator):
    desc: str
    _start_time: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        print(f"Timer '{self.desc}' took {elapsed_time:.3f} seconds")


def _download_mnist(datasets_path: pathlib.Path):
    mnist_files = [
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
    ]
    mnist_path = datasets_path / "MNIST" / "raw"
    if mnist_path.exists():
        print(f"Found existing MNIST dataset at {mnist_path}")
    else:
        print(f"Downloading MNIST dataset to {mnist_path}")
        mnist_path.mkdir(parents=True)
        root_url = (
            "https://graphcore-external-datasets.s3-eu-west-1.amazonaws.com/mnist/"
        )
        for file_str in mnist_files:
            mnist_file = str(mnist_path / file_str)
            urllib.request.urlretrieve(root_url + file_str, mnist_file)
            torchvision.datasets.utils.extract_archive(mnist_file, str(mnist_path))


# dataset begin
@Timer(desc="Getting and preparing MNIST data")
def get_mnist_data(
    datasets_dir: str,
    test_batch_size: int,
    batch_size: int,
    limit_nbatches: Optional[int],
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Get the training and testing data for MNIST.

    Args:
        datasets_dir (str): Path to find, else download, MNIST into.
        test_batch_size (int): The batch size for test.
        batch_size (int): The batch size for training.
        limit_nbatches (int): Limit the number of batches for training and testing.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: the data loaders for training data and test data.
    """
    datasets_path = pathlib.Path(datasets_dir).expanduser().resolve()
    _download_mnist(datasets_path)

    def maybe_limit_nbatches(dataset):
        if limit_nbatches:
            limit_nsamples = (batch_size + test_batch_size) * limit_nbatches
            indices = range(min(len(dataset), limit_nsamples))
            return torch.utils.data.Subset(dataset, indices)
        else:
            return dataset

    def mnist(train: bool):
        return maybe_limit_nbatches(
            torchvision.datasets.MNIST(
                datasets_path,
                train=train,
                download=False,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        # Mean and std computed on the training set.
                        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                    ]
                ),
            )
        )

    training_data = torch.utils.data.DataLoader(
        mnist(train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    validation_data = torch.utils.data.DataLoader(
        mnist(train=False),
        batch_size=test_batch_size,
        shuffle=True,
        drop_last=True,
    )
    return training_data, validation_data
