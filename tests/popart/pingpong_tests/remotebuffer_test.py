import numpy as np
import popart
import pytest
import re
from collections import namedtuple

# importing test_session requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from test_session import PopartTestSession

Session = namedtuple('Session', ['session', 'anchors'])


def create_model(batch_size,
                 multsize,
                 num_multiplications,
                 init_weights,
                 dtype=np.float16):
    builder = popart.Builder()
    input_shape = popart.TensorInfo(
        'FLOAT' if dtype == np.float32 else 'FLOAT16',
        [batch_size, 1, multsize])
    input_data = builder.addInputTensor(input_shape)
    x = input_data
    for i in range(num_multiplications):
        with builder.pingPongPhase(i):
            with builder.nameScope(str(i)):
                W = builder.addInitializedInputTensor(init_weights[i])
                print('Weight tensor:', W)
                x = builder.aiOnnx.matmul([x, W])
    output = x
    builder.addOutputTensor(output)
    diff = builder.aiOnnx.sub([input_data, output])
    loss = popart.L1Loss(diff, 'lossval', 1.0)
    proto = builder.getModelProto()
    return proto, input_data, output, loss


def get_device(sim=True, device_id=None, num_ipus=1, pingpong=False):
    print('device:', device_id)
    # Select a device
    deviceManager = popart.DeviceManager()
    if sim:
        options = {
            'compileIPUCode': True,
            'numIPUs': num_ipus,
            'tilesPerIPU': 1216
        }
        device = deviceManager.createIpuModelDevice(options)
    else:
        if (device_id is not None):
            device = deviceManager.acquireDeviceById(
                int(device_id), popart.SyncPattern.PingPong
                if pingpong else popart.SyncPattern.Full)
        else:
            pattern = (popart.SyncPattern.PingPong
                       if pingpong else popart.SyncPattern.Full)
            device = tu.acquire_ipu(numIPUs=num_ipus,
                                    tilesPerIPU=1216,
                                    pattern=pattern)
            assert device
        if device is None:
            raise Exception('Failed to acquire IPU. Exiting.')
    return device


def init_session(proto, loss, dataFlow, userOpts, device, training=True):
    # Create a session to compile and execute the graph
    session = popart.TrainingSession(fnModel=proto,
                                     losses=[loss],
                                     deviceInfo=device,
                                     optimizer=popart.SGD({
                                         'defaultLearningRate': (1e-5, False),
                                     }),
                                     dataFeed=dataFlow,
                                     userOptions=userOpts)

    session.prepareDevice()

    # Create buffers to receive results from the execution
    anchors = session.initAnchorArrays()
    return Session(session, anchors)


class DataSet:
    def __init__(self,
                 batch_size,
                 batches_per_step,
                 multsize,
                 dtype=np.float16):
        self.data = np.random.normal(0, 1, [100, 1, multsize]).astype(dtype)
        self.num_examples = len(self.data)
        self.batch_size = batch_size
        self.multsize = multsize
        self.batches_per_step = min(batches_per_step,
                                    self.num_examples // self.batch_size)
        self.inputs_per_step = self.batch_size * self.batches_per_step
        self.steps_per_epoch = self.num_examples // self.inputs_per_step

    def __getitem__(self, key):
        input_begin = key * self.inputs_per_step
        input_end = input_begin + self.inputs_per_step
        data = self.data[input_begin:input_end]
        data = data.reshape(
            [self.batches_per_step, self.batch_size, 1, self.multsize])
        return data

    def __iter__(self):
        return (self[j] for j in range(self.steps_per_epoch))

    def __len__(self):
        return self.steps_per_epoch


# TODO: Fixme when Poplar is ready.
@pytest.mark.skip(reason="Enable when Poplar backend is ready")
def test_remote_buffer():
    np.random.seed(1234)
    """
    In this test we check that the remote buffer content stays intact.
    """
    batch_size = 32
    batches_per_step = 1
    multsize = 8
    nummult = 4
    dtype = np.float32

    init_weights = [
        (np.eye(multsize, multsize) +
         np.random.normal(0.0, 1e-6, [multsize, multsize])).astype(dtype)
        for i in range(nummult)
    ]

    training_set = DataSet(batch_size, batches_per_step, multsize, dtype)

    print('Creating ONNX model.')
    proto, data_in, output, loss = create_model(batch_size, multsize, nummult,
                                                init_weights, dtype)

    # Describe how to run the model
    anchor_desc = {
        output: popart.AnchorReturnType('ALL'),
        loss.output(0): popart.AnchorReturnType('ALL')
    }
    dataFlow = popart.DataFlow(batches_per_step, anchor_desc)

    # Options
    userOpts = popart.SessionOptions()
    userOpts.enableOutlining = False
    userOpts.outlineThreshold = -np.inf
    userOpts.enableOutliningCopyCostPruning = False
    userOpts.virtualGraphMode = popart.VirtualGraphMode.PingPong
    userOpts.pingPongPhases = nummult
    userOpts.constantWeights = False

    device = get_device(False, None, 2, True)

    training = init_session(proto,
                            loss,
                            dataFlow,
                            userOpts,
                            device,
                            training=True)

    training.session.weightsFromHost()
    training.session.optimizerFromHost()
    training.session.weightsToHost()
    weights = {}
    for i in range(nummult):
        weights[str(i) + "/init_input"] = np.empty([multsize, multsize], dtype)
    weightsio = popart.PyWeightsIO(weights)
    training.session.readWeights(weightsio)

    for i in range(nummult):
        assert (np.allclose(init_weights[i],
                            weights[str(i) + "/init_input"],
                            rtol=0.0,
                            atol=0.0))

    for i in range(100):
        for data in training_set:
            stepio = popart.PyStepIO({data_in: data}, training.anchors)
            training.session.run(stepio)
            training.session.weightsToHost()

    training.session.readWeights(weightsio)

    for i in range(nummult):
        assert (np.allclose(init_weights[i],
                            weights[str(i) + "/init_input"],
                            rtol=0.0,
                            atol=0.0))
