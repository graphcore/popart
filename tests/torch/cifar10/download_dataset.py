from pathlib import Path
import tempfile
from torchvision import transforms, datasets
from urllib.request import urlretrieve
from urllib.parse import urlparse
import tarfile
import c10datadir

if not c10datadir.c10datadir.exists():
    print(f'Creating directory {c10datadir.c10datadir}', flush=True)
    c10datadir.c10datadir.mkdir()

else:
    print(f'Using existing directory {c10datadir.c10datadir}', flush=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# check if the tarball has been downloaded, if not : download it
cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
c10fname = Path(urlparse(cifar10_url).path).name
c10tar = c10datadir.c10datadir / c10fname
if not c10tar.exists():
    print('Attempting to download CIFAR10 dataset...', flush=True)
    urlretrieve(cifar10_url, c10tar)
else:
    print('CIFAR10 dataset already downloaded', flush=True)

# check if the required files have been extracted, if not : extract 'em
expected_files = [
    c10datadir.c10datadir / "cifar-10-batches-py" / x for x in [
        "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4",
        "data_batch_5", "test_batch", "batches.meta"
    ]
]
expected_files_all_present = all(x.exists() for x in expected_files)
if expected_files_all_present:
    print('CIFAR10 dataset already extracted', flush=True)
else:
    print('Extracting CIFAR10 data...', flush=True)
    data_tar = tarfile.open(c10tar)
    data_tar.extractall(path=c10datadir.c10datadir)

print('Verifying CIFAR10 data...', flush=True)
trainset = datasets.CIFAR10(root=c10datadir.c10datadir,
                            train=True,
                            download=False,
                            transform=transform)
