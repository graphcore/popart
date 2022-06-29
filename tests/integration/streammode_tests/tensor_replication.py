# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart

BATCHES_PER_STEP = 7
BATCH_SIZE = 16
CHANNELS = 2
DATA_LEN = 3
# Learning rate 1 for easy comparison.
LEARNING_RATE = 1.0

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def return_options(anchorDict):
    opts = popart.SessionOptions()

    if anchorDict["Pipelining"]:
        opts.enablePipelining = True

    if anchorDict["AccumulationFactor"] > 1:
        opts.enableGradientAccumulation = True
        opts.accumulationFactor = anchorDict["AccumulationFactor"]

    ipus = 1
    if anchorDict["Pipelining"] == False:
        ipus = 1 * anchorDict["ReplicationFactor"]
    else:
        opts.virtualGraphMode = popart.VirtualGraphMode.Auto
        ipus = 2 * anchorDict["ReplicationFactor"]

    if anchorDict["ReplicationFactor"] > 1:
        opts.replicatedGraphCount = anchorDict["ReplicationFactor"]
        opts.enableReplicatedGraphs = True
        device = tu.create_test_device(numIpus=ipus)
        if tu.ipu_available(ipus):
            device = tu.create_test_device(numIpus=ipus)
        else:
            print("No IPUS available for test options.")
            return None, None
    else:
        device = tu.create_test_device(numIpus=ipus)
    print("device: ", device)
    return opts, device


@tu.requires_ipu
def test_tensor_replication():
    """
    This test will verify that a broadcasted input tensor
    is the same on all replicas.
    """
    anchorDict = {
        "ReplicationFactor": 2,
        # Accl factor must divide batch size
        "AccumulationFactor": 4,
        "Pipelining": False,  #True
        "ReturnType": "ALL"
    }
    label_array = np.ones([BATCH_SIZE]).astype(np.int32)

    micro_batch_size = BATCH_SIZE // (anchorDict["AccumulationFactor"] *
                                      anchorDict["ReplicationFactor"])

    builder = popart.Builder()
    input_shape = [micro_batch_size, CHANNELS, DATA_LEN, DATA_LEN]

    data_shape = popart.TensorInfo("FLOAT", input_shape)
    lbl_shape = popart.TensorInfo("INT32", [micro_batch_size])
    w = builder.addInitializedInputTensor(
        np.random.random_sample(input_shape).astype(np.float32))

    settings = popart.InputSettings(popart.ReplicatedStreamMode.Broadcast)

    ip = builder.addInputTensor(data_shape, settings)
    lb = builder.addInputTensor(lbl_shape)

    a = builder.aiOnnx.matmul([ip, w])
    o = builder.reshape_const(
        builder.aiOnnx, [a],
        [micro_batch_size, CHANNELS * DATA_LEN * DATA_LEN])
    o = builder.aiOnnx.relu([o])
    o = builder.aiOnnx.softmax([o])
    nll = builder.aiGraphcore.nllloss([o, lb])

    GRAD = popart.reservedGradientPrefix() + w
    ACCL = popart.reservedAccumPrefix() + w
    art = popart.AnchorReturnType("All")
    data_flow = popart.DataFlow(BATCHES_PER_STEP, {
        o: art,
        a: art,
        ip: art,
        w: art,
        GRAD: art,
        ACCL: art
    })

    opts, device = return_options(anchorDict)
    with device as device:
        if device is None:
            pytest.skip("Test needs to run on IPU, but none are available")

        session = popart.TrainingSession(
            fnModel=builder.getModelProto(),
            dataFlow=data_flow,
            loss=nll,
            optimizer=popart.ConstSGD(LEARNING_RATE),
            userOptions=opts,
            deviceInfo=device)

        session.prepareDevice()

        if anchorDict[
                "ReplicationFactor"] > 1 and settings.replicatedStreamMode(
                ) != popart.ReplicatedStreamMode.Broadcast:
            input_shape = [anchorDict["ReplicationFactor"]] + input_shape
            label_array = label_array.reshape(
                [anchorDict["ReplicationFactor"], -1])
        if anchorDict["AccumulationFactor"] > 1:
            input_shape = [anchorDict["AccumulationFactor"]] + input_shape
            label_array = label_array.reshape(
                [anchorDict["AccumulationFactor"], -1])
        if BATCHES_PER_STEP > 1:
            input_shape = [BATCHES_PER_STEP] + input_shape
            label_array = np.repeat(label_array[np.newaxis], BATCHES_PER_STEP,
                                    0)

        anchors = session.initAnchorArrays()
        in_array = np.random.random_sample(input_shape).astype(np.float32)

        stepio = popart.PyStepIO({ip: in_array, lb: label_array}, anchors)
        session.weightsFromHost()

        session.run(stepio)

        #Compare the inputs
        for batch in range(anchors[ip].shape[0]):
            in_0 = anchors[ip][batch, -1, 0, :, :, :, :]
            in_1 = anchors[ip][batch, -1, 1, :, :, :, :]

            mx = anchors[ip].shape[len(anchors[ip].shape) - 1]
            _ = mx if mx <= 5 else 5
            assert np.allclose(in_0, in_1, equal_nan=False)
