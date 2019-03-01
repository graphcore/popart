from pathlib import Path
import tempfile
from torchvision import transforms, datasets
from urllib.request import urlretrieve
from urllib.parse import urlparse
import tarfile

tmpdir = tempfile.gettempdir()
c10datadir = Path(tmpdir) / 'cifar10data'
if not c10datadir.exists():
    print(f'Creating directory {c10datadir}', flush=True)
    c10datadir.mkdir()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print('Attempting to download CIFAR10 dataset...', flush=True)
cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
c10fname = Path(urlparse(cifar10_url).path).name
c10tar = c10datadir / c10fname
if not c10tar.exists():
    urlretrieve(cifar10_url, c10tar)
else:
    print('Data already downloaded', flush=True)

print('Extracting CIFAR10 data...', flush=True)
data_tar = tarfile.open(c10tar)
data_tar.extractall(path=c10datadir)

print('Verifying CIFAR10 data...', flush=True)
trainset = datasets.CIFAR10(
    root=c10datadir, train=True, download=False, transform=transform)
