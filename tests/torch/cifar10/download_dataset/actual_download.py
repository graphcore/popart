import os

from urllib.request import urlretrieve

import download_dataset.common

if download_dataset.common.already_downloaded_and_extracted():
    download_dataset.common.report_skipped()

# check if the tarball has been downloaded, if not : download it
if not os.path.exists(download_dataset.common.tar_file_path()):
    print('Attempting to download CIFAR10 dataset...', flush=True)
    urlretrieve(download_dataset.common.CIFAR10_URL,
                download_dataset.common.tar_file_path())
else:
    print('CIFAR10 dataset already downloaded', flush=True)
    download_dataset.common.report_skipped()
