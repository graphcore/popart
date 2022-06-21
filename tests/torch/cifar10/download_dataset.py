# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
from torchvision import transforms, datasets

import c10datadir

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print('Verifying CIFAR10 data...', flush=True)
trainset = datasets.CIFAR10(root=c10datadir.c10datadir,
                            train=True,
                            download=False,
                            transform=transform)
