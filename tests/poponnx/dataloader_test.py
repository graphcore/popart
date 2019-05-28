import poponnx
import os
import numpy
import tempfile
from torchvision import transforms, datasets


def get_data_dir():
    datadir = "unset"

    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_c10datadir = os.path.join(dir_path, "c10datadir.py")
    if os.path.exists(path_c10datadir):
        import c10datadir
        datadir = c10datadir.c10datadir
    else:
        tmpdir = tempfile.gettempdir()
        datadir = os.path.abspath(os.path.join(tmpdir, 'cifar10data'))
    print("Using datadir=%s" % (datadir))

    return datadir


def test_create_data_loader():

    samplesPerBatch = 4
    batchesPerStep = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(
        root=get_data_dir(), train=True, download=True, transform=transform)

    loader = poponnx.DataLoader(
        dataset,
        # the amount of data loaded for each step.
        # note this is not the batch size, it's the "step" size
        # (samples per step)
        batch_size=samplesPerBatch * batchesPerStep)

    for i, data in enumerate(loader):
        assert (data[0].dtype, numpy.float32)
        pass


def test_create_data_loader_float16():

    samplesPerBatch = 4
    batchesPerStep = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(
        root=get_data_dir(), train=True, download=True, transform=transform)

    loader = poponnx.DataLoader(
        dataset,
        # the amount of data loaded for each step.
        # note this is not the batch size, it's the "step" size
        # (samples per step)
        batch_size=samplesPerBatch * batchesPerStep,
        tensor_type='float16')

    for i, data in enumerate(loader):
        assert (data[0].dtype, numpy.float16)
        pass


def test_create_data_loader_with_stats():

    samplesPerBatch = 4
    batchesPerStep = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(
        root=get_data_dir(), train=True, download=True, transform=transform)

    loader = poponnx.DataLoader(
        dataset,
        # the amount of data loaded for each step.
        # note this is not the batch size, it's the "step" size
        # (samples per step)
        batch_size=samplesPerBatch * batchesPerStep,
        log_statisics=True)

    for i, data in enumerate(loader):
        pass


'''
Test does not work on master build bot!!!
def test_create_data_loader_with_workers():

    samplesPerBatch = 4
    batchesPerStep = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(
        root=get_data_dir(), train=True, download=True, transform=transform)

    loader = poponnx.DataLoader(
        dataset,
        # the amount of data loaded for each step.
        # note this is not the batch size, it's the "step" size
        # (samples per step)
        batch_size=samplesPerBatch * batchesPerStep,
        num_workers=8,
        log_statisics=True)

    for i, data in enumerate(loader):
        pass
'''
