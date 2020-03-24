import numpy as np
import os
from collections import namedtuple

# import the PopART Horovod extension
import horovod.popart as hvd
import popart

Session = namedtuple('Session', ['session', 'anchors'])
batch_size = 1
IN_SHAPE = 784
OUT_SHAPE = 10


def create_model():
    builder = popart.Builder()
    dtype = np.float32

    np.random.seed(42)
    input_shape = popart.TensorInfo(dtype, [batch_size, IN_SHAPE])
    x = builder.addInputTensor(input_shape)
    init_weights = np.random.normal(0, 1, [IN_SHAPE, OUT_SHAPE]).astype(dtype)
    w = builder.addInitializedInputTensor(init_weights)
    init_biases = np.random.normal(0, 1, [OUT_SHAPE]).astype(dtype)
    b = builder.addInitializedInputTensor(init_biases)
    h = builder.aiOnnx.matmul([x, w])
    a = builder.aiOnnx.add([h, b])

    output = a
    builder.addOutputTensor(output)
    probs = builder.aiOnnx.softmax([output])

    label_shape = popart.TensorInfo("INT32", [batch_size])
    label = builder.addInputTensor(label_shape)

    loss = popart.NllLoss(probs, label, "nllLossVal")

    proto = builder.getModelProto()

    return builder, proto, x, label, output, loss


def get_device(simulation=True):
    num_ipus = 1
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


def init_session(proto, loss, dataFlow, userOpts, device):
    # Create a session to compile and execute the graph
    optimizer = popart.SGD({"defaultLearningRate": (0.1, False)})
    session = popart.TrainingSession(fnModel=proto,
                                     losses=[loss],
                                     deviceInfo=device,
                                     optimizer=optimizer,
                                     dataFeed=dataFlow,
                                     userOptions=userOpts)

    session.prepareDevice()
    session.setRandomSeed(42)

    # Create buffers to receive results from the execution
    anchors = session.initAnchorArrays()

    return Session(session, anchors), optimizer


def train():
    builder, proto, data_in, labels_in, output, loss, = create_model()

    batches_per_step = 32
    anchor_desc = {
        output: popart.AnchorReturnType("ALL"),
        loss.output(0): popart.AnchorReturnType("ALL")
    }
    dataFlow = popart.DataFlow(batches_per_step, anchor_desc)

    userOpts = popart.SessionOptions()
    device = get_device()

    # Enable host side AllReduce operations in the graph
    userOpts.hostAllReduce = True
    training, optimizer = init_session(proto, loss, dataFlow, userOpts, device)
    if userOpts.hostAllReduce:
        hvd.init()

        distributed_optimizer = hvd.DistributedOptimizer(
            optimizer, training.session, userOpts)

        # Broadcast weights to all the other processes
        hvd.broadcast_weights(training.session, root_rank=0)

    training.session.weightsFromHost()
    training.session.optimizerFromHost()

    # Synthetic data
    data = np.random.normal(size=(batches_per_step, batch_size, 784)).astype(
        np.float32)
    labels = np.zeros((batches_per_step, batch_size, 1)).astype(np.int32)

    num_training_steps = 10

    for _ in range(num_training_steps):
        stepio = popart.PyStepIO({
            data_in: data,
            labels_in: labels
        }, training.anchors)
        training.session.run(stepio)


train()
