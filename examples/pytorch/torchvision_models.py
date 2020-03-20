# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import time
import argparse
import os

import popart
import popart.torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as model


def _get_torch_type(np_type):
    return {
        np.float16: 'torch.HalfTensor',
        np.float32: 'torch.FloatTensor'
    }[np_type]


def get_dataset(opts):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Train
    trainset = torchvision.datasets.CIFAR10(root='/tmp/data',
                                            train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=opts.batch_size * opts.batches_per_step,
        shuffle=True,
        num_workers=0,
        drop_last=True)

    # Test
    testset = torchvision.datasets.CIFAR10(root='/tmp/data',
                                           train=False,
                                           download=True,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=opts.batch_size * opts.batches_per_step,
        shuffle=False,
        num_workers=0,
        drop_last=True)

    return trainset, testset, trainloader, testloader


def get_device(num_ipus, simulation=False):

    deviceManager = popart.DeviceManager()
    if simulation:
        print("Creating ipu sim")
        ipu_options = {
            "compileIPUCode": True,
            'numIPUs': num_ipus,
            "tilesPerIPU": 1216
        }
        device = deviceManager.createIpuModelDevice(ipu_options)
        if device is None:
            raise OSError("Failed to acquire IPU.")
    else:
        print("Aquiring IPU")
        device = deviceManager.acquireAvailableDevice(num_ipus)
        if device is None:
            raise OSError("Failed to acquire IPU.")
        else:
            print("Acquired IPU: {}".format(device))

    return device


def get_options(opts):

    # Create a session to compile and execute the graph
    options = popart.SessionOptions()

    options.engineOptions = {"debug.allowOutOfMemory": "true"}

    # Enable the reporting of variables in the summary report
    options.reportOptions = {'showVarStorage': 'true'}

    if opts.fp_exceptions:
        # Enable exception on floating point errors
        options.enableFloatingPointChecks = True

    if opts.prng:
        options.enableStochasticRounding = True

    # Need to disable constant weights so they can be set before
    # executing the inference session
    options.constantWeights = False

    # Enable recomputation
    if opts.recompute:
        options.autoRecomputation = popart.RecomputationType.Standard

    # Enable auto-sharding
    if opts.num_ipus > 1 and opts.num_ipus > opts.replication_factor:
        options.enableVirtualGraphs = True
        options.virtualGraphMode = popart.VirtualGraphMode.Auto

    # Enable pipelining
    if opts.pipeline:
        options.enablePipelining = True

    if (opts.replication_factor > 1):
        options.enableReplicatedGraphs = True
        options.replicatedGraphCount = opts.replication_factor

        # Enable merge updates
        options.mergeVarUpdate = popart.MergeVarUpdateType.AutoLoose
        options.mergeVarUpdateMemThreshold = 6000000

    return options


def train_process(opts):
    net = getattr(model, opts.model_name)(
        pretrained=False,
        progress=True,
        num_classes=10 if opts.dataset == "CIFAR-10" else 1000)

    # Models are missing a softmax layer to work with our NllLoss,
    # so we just add one on.
    net = nn.Sequential(net, nn.Softmax(dim=1))

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=opts.learning_rate)

    trainset, testset, trainloader, testloader = get_dataset(opts)

    inputs, labels = iter(trainloader).next()

    sessionOpts = get_options(opts)

    start = time.time()
    # Pass all the pytorch stuff to the session
    torchSession = popart.torch.TrainingSession(
        torchModel=net,
        inputs=inputs,
        targets=labels,  # .type(_get_torch_type(opts.precision))
        optimizer=optimizer,
        losses=criterion,
        batch_size=opts.batch_size,
        batches_per_step=opts.batches_per_step,
        deviceInfo=get_device(opts.num_ipus, opts.simulation),
        userOptions=sessionOpts)
    print("Converting pytorch model took {:.2f}s".format(time.time() - start))

    # Prepare for training.
    start = time.time()
    print("Compiling model...")
    anchors = torchSession.initAnchorArrays()

    torchSession.prepareDevice()
    torchSession.optimizerFromHost()
    torchSession.weightsFromHost()
    print("Compiling popart model took {:.2f}s".format(time.time() - start))
    for epoch in range(opts.epochs):  # loop over the dataset multiple times
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
                i +
                1) * opts.batch_size * opts.batches_per_step // len(trainset)
            print('\repoch {} [{}{}]  '.format(epoch + 1, progress * '.',
                                               (20 - progress) * ' '),
                  end='')

            results = np.argmax(
                anchors['output_0'].reshape([
                    opts.batches_per_step * opts.batch_size,
                    10 if opts.dataset == "CIFAR-10" else 1000
                ]), 1)
            num_correct = np.sum(results == anchors['target_0'].reshape(
                [opts.batches_per_step * opts.batch_size]))
            running_accuracy += num_correct
        print("Accuracy: {}%".format(running_accuracy * 100 / len(trainset)))

        end_time = time.time()
        print('loss: {:.2f}'.format(running_loss / (i + 1)))
        print("Images per second: {:.0f}".format(
            len(trainset) / (end_time - start_time)))

        if not opts.no_validation:
            # Save the model with weights
            onnx_path = os.path.join(opts.data_dir, opts.onnx_model_name)
            torchSession.modelToHost(onnx_path)

            inferenceOpts = get_options(opts)
            inferenceOpts.constantWeights = False

            # Pytorch currently doesn't support importing from onnx:
            # https://github.com/pytorch/pytorch/issues/21683
            # And pytorch->onnx->caffe2 is broken:
            # https://github.com/onnx/onnx/issues/2463
            # So we import into popart session and infer.
            # Alternatively, use any other ONNX compatible runtime.
            builder = popart.Builder(onnx_path)
            inferenceSession = popart.InferenceSession(
                fnModel=builder.getModelProto(),
                dataFeed=popart.DataFlow(
                    opts.batches_per_step,
                    {"output_0": popart.AnchorReturnType("ALL")}),
                deviceInfo=get_device(opts.num_ipus, opts.simulation),
                userOptions=inferenceOpts)

            print("Compiling test model...")
            inferenceSession.prepareDevice()

            inferenceSession.weightsFromHost()
            inferenceAnchors = inferenceSession.initAnchorArrays()
            print("#" * 20, "Test phase:", "#" * 20)
            test_accuracy = 0
            for j, data in enumerate(testloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                stepio = popart.PyStepIO({"input_0": inputs.data.numpy()},
                                         inferenceAnchors)

                inferenceSession.run(stepio)

                progress = 20 * (
                    j + 1
                ) * opts.batch_size * opts.batches_per_step // len(testset)
                print('\rtest epoch {} [{}{}]  '.format(
                    epoch + 1, progress * '.', (20 - progress) * ' '),
                      end='')

                results = np.argmax(
                    inferenceAnchors['output_0'].reshape(
                        [opts.batches_per_step * opts.batch_size, 10]), 1)
                num_correct = np.sum(results == labels.data.numpy().reshape(
                    [opts.batches_per_step * opts.batch_size]))
                test_accuracy += num_correct
            inferenceSession = None
            print("Accuracy: {}%".format(test_accuracy * 100 / len(testset)))

    print('Finished Training')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='TorchVision training in Popart',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # -------------- DATASET ------------------
    group = parser.add_argument_group('Dataset')
    group.add_argument('--dataset',
                       type=str,
                       choices=["IMAGENET", "CIFAR-10"],
                       default="CIFAR-10",
                       help="Choose which dataset to run on")
    group.add_argument('--data-dir',
                       type=str,
                       default="/tmp/data/",
                       help="Path to data")
    group.add_argument('--onnx-model-name',
                       type=str,
                       default="torchModel.onnx",
                       help="ONNX model save name")

    # -------------- MODEL ------------------
    group = parser.add_argument_group('Model')
    group.add_argument('--model-name',
                       type=str,
                       default="resnet18",
                       help='Model name e.g. resnet18',
                       choices=["resnet18", "resnet34", "resnet50"])
    group.add_argument('--batch-size',
                       type=int,
                       default=4,
                       help='Set batch size. '
                       'This must be a multiple of the replication-factor')
    group.add_argument('--precision',
                       choices=['16', '32'],
                       default="16",
                       help="Setting of float datatype 16 or 32")
    group.add_argument('--prng',
                       action="store_true",
                       default=True,
                       help="Enable Stochastic Rounding")
    group.add_argument('--no-prng',
                       action="store_false",
                       dest='prng',
                       default=False,
                       help="Disable Stochastic Rounding")
    group.add_argument('--fp-exceptions',
                       action="store_true",
                       default=False,
                       help="Enable floating point exception")

    # -------------- TRAINING ------------------
    group = parser.add_argument_group('Training')

    group.add_argument(
        '--base-learning-rate',
        type=int,
        default=-6,
        help="Base learning rate exponent (2**N). blr = lr /  bs")

    group.add_argument(
        '--learning-rate-decay',
        type=str,
        default="1,0.1,0.01",
        help="Learning rate decay schedule. Comma Separated ('1,0.1,0.01')")
    group.add_argument('--learning-rate-schedule',
                       type=str,
                       default="0.5,0.75",
                       help="Learning rate drop points (proportional). "
                       "Comma Separated ('0.5,0.75')")
    group.add_argument('--epochs',
                       type=int,
                       default=30,
                       help="Number of training epochs")
    group.add_argument('--no-validation',
                       action="store_true",
                       help="Dont do any validation runs.")
    group.add_argument('--valid-per-epoch',
                       type=float,
                       default=1,
                       help="Validation steps per epoch.")
    group.add_argument('--steps-per-log',
                       type=int,
                       default=1,
                       help="Log statistics every N steps.")
    group.add_argument(
        '--weight-decay',
        type=float,
        default=0,
        help="Value for weight decay bias, setting to 0 removes weight decay.")
    group.add_argument('--num-ipus',
                       type=int,
                       default=1,
                       help="Number of IPU's")
    group.add_argument(
        '--replication-factor',
        type=int,
        default=1,
        help="Number of times to replicate the graph to perform data parallel"
        " training. Must be a factor of the number of IPUs")
    group.add_argument(
        '--recompute',
        action="store_true",
        default=False,
        help="Enable recomputations of activations in backward pass")
    group.add_argument('--pipeline',
                       action="store_true",
                       default=False,
                       help="Pipeline the model over IPUs")
    group.add_argument(
        '--batches-per-step',
        type=int,
        default=100,
        help="How many minibatches to perform on the device before returning"
        "to the host.")
    group.add_argument('--simulation',
                       action="store_true",
                       help="Run the program on the IPU Model")

    args = parser.parse_args()

    args.learning_rate_decay = list(
        map(float, args.learning_rate_decay.split(',')))
    args.learning_rate_schedule = list(
        map(float, args.learning_rate_schedule.split(',')))
    args.learning_rate = (2**args.base_learning_rate) * (args.batch_size)

    if ((args.batch_size % args.replication_factor) != 0):
        raise Exception("Invalid Argument : Batch size ({}) must be a "
                        "multiple of replication factor ({})".format(
                            args.batch_size, args.replication_factor))

    if ((args.num_ipus % args.replication_factor) != 0):
        raise Exception("Invalid Argument : Number of IPUs ({}) must be a "
                        "multiple of replication factor ({})".format(
                            args.num_ipus, args.replication_factor))

    # The number of samples that the device will process currently
    args.samples_per_device = (int)(args.batch_size / args.replication_factor)

    # Display Options.
    log_str = ("{model_name} Training.\n"
               " Dataset {dataset}\n"
               " Num IPUs {num_ipus}\n"
               " Precision {precision}\n"
               " Stochastic Rounding {prng}\n"
               " Floating Point Exceptions {fp_exceptions}\n"
               "Training Graph.\n"
               " Batch Size {batch_size}.\n"
               " Batches Per Step {batches_per_step}.\n"
               " Replication Factor {replication_factor}.\n"
               " Epochs {epochs}\n"
               " Weight Decay {weight_decay}\n"
               " Base Learning Rate 2^{base_learning_rate}\n"
               " Learning Rate {learning_rate}\n"
               " Learning Rate Schedule {learning_rate_schedule}\n"
               "Validation Graph.\n")

    print(log_str.format(**vars(args)))

    args.train = True
    args.precision = np.float16 if args.precision == '16' else np.float32

    # Detemine if we are current with the gc-profile tool
    args.gc_profile_log_dir = os.environ.get('GC_PROFILE_LOG_DIR', None)

    train_process(args)
