# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
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
def test_anchor_output():
    """
    Test a specific example's output of weights and accumulated gradient.
    This should catch any slicing issues.
    """
    anchorDict = {
        "ReplicationFactor": 2,
        # Accl factor must divide batch size
        "AccumulationFactor": 4,
        "Pipelining": True,
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

    ip = builder.addInputTensor(data_shape)
    lb = builder.addInputTensor(lbl_shape)

    a = builder.aiOnnx.matmul([ip, w])
    o = builder.reshape_const(
        builder.aiOnnx, [a],
        [micro_batch_size, CHANNELS * DATA_LEN * DATA_LEN])
    o = builder.aiOnnx.relu([o])
    o = builder.aiOnnx.softmax([o])
    nll = builder.aiGraphcore.nllloss([o, lb])

    GRAD = popart.reservedGradientPrefix() + w
    ACCL = popart.reservedAcclToAccumulatorPrefix(
    ) + popart.reservedGradientPrefix() + w
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

    if device is None:
        pytest.skip("Test needs to run on IPU, but none are available")

    session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                     dataFlow=data_flow,
                                     losses=[popart.IdentityLoss(nll, "loss")],
                                     optimizer=popart.ConstSGD(LEARNING_RATE),
                                     userOptions=opts,
                                     deviceInfo=device)

    session.prepareDevice()

    if anchorDict["ReplicationFactor"] > 1:
        input_shape = [anchorDict["ReplicationFactor"]] + input_shape
        label_array = label_array.reshape(
            [anchorDict["ReplicationFactor"], -1])
    if anchorDict["AccumulationFactor"] > 1:
        input_shape = [anchorDict["AccumulationFactor"]] + input_shape
        label_array = label_array.reshape(
            [anchorDict["AccumulationFactor"], -1])
    if BATCHES_PER_STEP > 1:
        input_shape = [BATCHES_PER_STEP] + input_shape
        label_array = np.repeat(label_array[np.newaxis], BATCHES_PER_STEP, 0)

    anchors = session.initAnchorArrays()
    in_array = np.random.random_sample(input_shape).astype(np.float32)

    stepio = popart.PyStepIO({ip: in_array, lb: label_array}, anchors)
    session.weightsFromHost()

    session.run(stepio)

    # Returned anchors will be of shape
    # [bps, grad_accl_factor, repl_factor, micro_batch_size, channels, data_len, data_len]
    for batch in range(anchors[w].shape[0]):
        for replica in range(anchors[w].shape[1]):
            # Weights should not change over the gradient accumulation
            # dimension - only after gradAccl steps.
            assert np.allclose(anchors[w][batch, 0, :, :, :, :, :],
                               anchors[w][batch, replica, :, :, :, :, :])

    # Check that the accumulated gradient plus the weights for the current batch
    # equals the weights for the next batch.
    # We will need to multiply by this adjustment factor as with most
    # implementations of replication.
    adj = 1 / anchorDict["ReplicationFactor"]
    # Batch loop
    for batch in range(anchors[w].shape[0] - 1):
        calc_weight = {}
        # Replica loop.
        for replica in range(anchors[w].shape[2]):
            # For each replica in each batch, take the relevant replica's
            #  last weight tensor in the accumulation loop minus
            # the sum of the accumulated gradients across replicas
            calc_weight[replica] = anchors[w][batch, -1, replica, :, :, :, :] - \
                 adj * np.sum(anchors[ACCL][batch, -1, :, :, :, :, :], axis=0)
            # Then compare against the last weight tensor of the next batch,
            # for the relevant replica. These should match.
            assert np.allclose(calc_weight[replica],
                               anchors[w][batch + 1, -1, replica, :, :, :, :])
