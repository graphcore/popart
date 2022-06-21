# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import tarfile

import c10datadir
import download_dataset.common

if download_dataset.common.already_downloaded_and_extracted():
    download_dataset.common.report_skipped()

print('Extracting CIFAR10 data...', flush=True)
download_dataset.common.tar_file_path()
data_tar = tarfile.open(download_dataset.common.tar_file_path())
data_tar.extractall(path=c10datadir.c10datadir)
data_tar.close()

print('Extracted, now remoiving tar.gz')
os.remove(download_dataset.common.tar_file_path())
