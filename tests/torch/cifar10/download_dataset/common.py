# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import sys

from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import urlparse

import c10datadir

CIFAR10_URL = 'http://s3-eu-west-1.amazonaws.com/graphcore-public-mirror/cifar-10-python.tar.gz'
CIFAR10_HASH = 'c58f30108f718f92721af3b95e74349a'


def already_downloaded_and_extracted():
    # check if the required files have been extracted, if not : extract 'em
    expected_files = [
        c10datadir.c10datadir / "cifar-10-batches-py" / x for x in [
            "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4",
            "data_batch_5", "test_batch", "batches.meta"
        ]
    ]

    return all(x.exists() for x in expected_files)


def report_skipped():
    sys.exit(2)


def tar_file_path():
    c10fname = Path(urlparse(CIFAR10_URL).path).name
    return str(c10datadir.c10datadir / c10fname)
