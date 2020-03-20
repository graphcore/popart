# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import time

import popart
import popart.torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

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

# Test phase
testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size * batches_per_step,
    shuffle=False,
    num_workers=0,
    drop_last=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x


def main():
    net = Net()

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    inputs, labels = iter(trainloader).next()

    opts = popart.SessionOptions()

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
        deviceInfo=popart.DeviceManager().acquireAvailableDevice(1),
        userOptions=opts)
    print("Converting pytorch model took {:.2f}s".format(time.process_time() -
                                                         start))

    # Prepare for training.
    anchors = torchSession.initAnchorArrays()

    print("Compiling model...")
    torchSession.prepareDevice()
    torchSession.optimizerFromHost()
    torchSession.weightsFromHost()

    for epoch in range(10):  # loop over the dataset multiple times
        start_time = time.time()

        running_loss = 0.0
        running_accuracy = 0
        print("#" * 20, "Train phase:", "#" * 20)
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

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

        # Save the model with weights
        torchSession.modelToHost("torchModel.onnx")

        # Pytorch currently doesn't support importing from onnx:
        # https://github.com/pytorch/pytorch/issues/21683
        # And pytorch->onnx->caffe2 is broken:
        # https://github.com/onnx/onnx/issues/2463
        # So we import into popart session and infer.
        # Alternatively, use any other ONNX compatible runtime.

        builder = popart.Builder("torchModel.onnx")

        inferenceSession = popart.InferenceSession(
            fnModel=builder.getModelProto(),
            dataFeed=popart.DataFlow(
                batches_per_step,
                {"output_0": popart.AnchorReturnType("ALL")}),
            deviceInfo=popart.DeviceManager().acquireAvailableDevice(1))

        print("Compiling test model...")
        inferenceSession.prepareDevice()
        inferenceAnchors = inferenceSession.initAnchorArrays()
        print("#" * 20, "Test phase:", "#" * 20)
        test_accuracy = 0
        for j, data in enumerate(testloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            stepio = popart.PyStepIO({"input_0": inputs.data.numpy()},
                                     inferenceAnchors)

            inferenceSession.run(stepio)

            progress = 20 * (j +
                             1) * batch_size * batches_per_step // len(testset)
            print('\rtest epoch {} [{}{}]  '.format(epoch + 1, progress * '.',
                                                    (20 - progress) * ' '),
                  end='')

            results = np.argmax(
                inferenceAnchors['output_0'].reshape(
                    [batches_per_step * batch_size, 10]), 1)
            num_correct = np.sum(results == labels.data.numpy().reshape(
                [batches_per_step * batch_size]))
            test_accuracy += num_correct

        print("Accuracy: {}%".format(test_accuracy * 100 / len(testset)))
    print('Finished Training')


if __name__ == '__main__':
    main()
