import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import popart
import popart.torch
import time
from resnet_main import resnet18
import torch.nn as nn
import torch.optim as optim

batches_per_step = 100
batch_size = 4

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size * batches_per_step,
    shuffle=True,
    num_workers=0,
    drop_last=True)


def main():
    net = resnet18(pretrained=False, progress=True, num_classes=10)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    inputs, labels = iter(trainloader).next()

    opts = popart.SessionOptions()
    opts.enableVirtualGraphs = True
    opts.virtualGraphMode = popart.VirtualGraphMode.Auto

    start = time.process_time()
    # Pass all the pytorch stuff to the session
    torchSession = popart.torch.TrainingSession(
        torchModel=net,
        inputs=inputs,
        targets=labels,
        optimizer=optimizer,
        losses=criterion,
        batch_size=batch_size,
        batches_per_step=batches_per_step,
        deviceInfo=popart.DeviceManager().acquireAvailableDevice(4),
        userOptions=opts)
    print("Converting pytorch model took {:.2f}s".format(time.process_time() -
                                                         start))

    anchors = torchSession.initAnchorArrays()

    torchSession.prepareDevice()
    torchSession.optimizerFromHost()
    torchSession.weightsFromHost()

    for epoch in range(10):  # loop over the dataset multiple times
        start_time = time.time()

        running_loss = 0.0
        running_accuracy = 0

        for i, data in enumerate(trainloader):

            torchSession.run(inputs, labels)
            running_loss += np.mean(anchors["loss_0"])

            progress = 20 * (
                i + 1) * batch_size * batches_per_step // len(trainset)
            print('\repoch {} [{}{}]  '.format(epoch + 1, progress * '.',
                                               (20 - progress) * ' '),
                  end='')

            results = np.argmax(
                anchors['output_0'].reshape(
                    [batches_per_step * batch_size, 10]), 1)
            num_correct = np.sum(results == anchors['target_0'].reshape(
                [batches_per_step * batch_size]))
            running_accuracy += num_correct
        print("Accuracy: {}%".format(running_accuracy * 100 / len(trainset)))

        end_time = time.time()
        print('loss: {:.2f}'.format(running_loss / (i + 1)))
        print("Images per second: {:.0f}".format(
            len(trainset) / (end_time - start_time)))
    print('Finished Training')


if __name__ == '__main__':
    main()
