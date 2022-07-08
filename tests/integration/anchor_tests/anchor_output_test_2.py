# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart

BATCHES_PER_STEP = 7
REPL_FACTOR = 4
BATCH_SIZE = 16
CHANNELS = 2
DATA_LEN = 3
# Learning rate 1 for easy comparison.
LEARNING_RATE = 1.0e-3

np.random.seed = 0

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def return_options(anchorDict):
    opts = popart.SessionOptions()

    if anchorDict["Pipelining"]:
        opts.enablePipelining = True

    ipus = 1
    if anchorDict["Pipelining"] is False:
        ipus = 1 * anchorDict["ReplicationFactor"]
    else:
        opts.virtualGraphMode = popart.VirtualGraphMode.Auto
        ipus = 2 * anchorDict["ReplicationFactor"]

    if anchorDict["ReplicationFactor"] > 1:
        opts.replicatedGraphCount = anchorDict["ReplicationFactor"]
        opts.enableReplicatedGraphs = True
    device = tu.create_test_device(numIpus=ipus)
    print("device: ", device)
    return opts, device


@tu.requires_ipu
def test_anchor_output():
    micro_batch_size = BATCH_SIZE // REPL_FACTOR
    data_shape = [CHANNELS, DATA_LEN, DATA_LEN]
    label_array = np.random.randint(
        0, 100, [BATCHES_PER_STEP, REPL_FACTOR, micro_batch_size]
    ).astype(np.int32)
    in_array = np.random.random_sample(
        [BATCHES_PER_STEP, REPL_FACTOR, micro_batch_size, *data_shape]
    ).astype(np.float32)
    input_shape = [micro_batch_size, CHANNELS, DATA_LEN, DATA_LEN]
    weight_array = np.random.random_sample(input_shape).astype(np.float32)

    def return_anchors(prefetch=True):
        """
        Test a specific example's output of weights and accumulated gradient.
        This should catch any slicing issues.
        """

        builder = popart.Builder()

        data_shape = popart.TensorInfo("FLOAT", input_shape)
        lbl_shape = popart.TensorInfo("INT32", [micro_batch_size])
        w = builder.addInitializedInputTensor(weight_array)

        ip = builder.addInputTensor(data_shape)
        lb = builder.addInputTensor(lbl_shape)

        a = builder.aiOnnx.add([ip, w])
        o = builder.reshape_const(
            builder.aiOnnx, [a], [micro_batch_size, CHANNELS * DATA_LEN * DATA_LEN]
        )
        o = builder.aiOnnx.relu([o])
        o = builder.aiOnnx.softmax([o])
        builder.addOutputTensor(o)

        art = popart.AnchorReturnType("ALL")
        data_flow = popart.DataFlow(
            BATCHES_PER_STEP,
            {
                o: art,
                ip: art,
                w: art,
                popart.reservedGradientPrefix() + o: art,
                popart.reservedGradientPrefix() + w: art,
            },
        )

        opts, deviceContext = return_options(
            {"ReplicationFactor": REPL_FACTOR, "Pipelining": False, "ReturnType": "ALL"}
        )
        opts.enablePrefetchDatastreams = prefetch
        opts.enableStochasticRounding = False

        with deviceContext as device:
            if device is None:
                pytest.skip("Test needs to run on IPU, but none are available")
            nlll = builder.aiGraphcore.nllloss([o, lb])
            session = popart.TrainingSession(
                builder.getModelProto(),
                deviceInfo=device,
                dataFlow=data_flow,
                loss=nlll,
                optimizer=popart.ConstSGD(LEARNING_RATE),
                userOptions=opts,
            )

            session.prepareDevice()
            session.setRandomSeed(0)

            anchors = session.initAnchorArrays()

            stepio = popart.PyStepIO({ip: in_array, lb: label_array}, anchors)
            session.weightsFromHost()

            session.run(stepio)
            return anchors

        in1 = return_anchors(True)
        in2 = return_anchors(False)
        for key in in1:
            print(key)
            print(np.max(np.abs(in1[key] - in2[key])))
            assert np.allclose(in1[key], in2[key], equal_nan=False)
