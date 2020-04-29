import hashlib
import sys

import download_dataset.common

if download_dataset.common.already_downloaded_and_extracted():
    download_dataset.common.report_skipped()

hasher = hashlib.md5()
with open(download_dataset.common.tar_file_path(), "rb") as f:
    hasher.update(f.read())

hash = hasher.hexdigest()

if hash == download_dataset.common.CIFAR10_HASH:
    print(f"{download_dataset.common.tar_file_path()} has correct hash")
else:
    print(f"{download_dataset.common.tar_file_path()} has invalid hash")
    sys.exit(1)
