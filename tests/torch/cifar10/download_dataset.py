from pathlib import Path
import tempfile
from torchvision import transforms, datasets
import sys

tmpdir = tempfile.gettempdir()
c10datadir = Path(tmpdir) / 'cifar10data'
if not c10datadir.exists():
    print(f'Creating directory {c10datadir}')
    sys.stdout.flush()
    c10datadir.mkdir()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print('Attempting to download CIFAR10 dataset...')
sys.stdout.flush()
trainset = datasets.CIFAR10(
    root=c10datadir, train=True, download=True, transform=transform)
