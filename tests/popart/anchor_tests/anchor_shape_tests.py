import numpy as np
import pytest
import popart
import itertools

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

# Co-prime numbers to avoid `accidentally correct` answers.
BATCHES_PER_STEP = 7
BATCH_SIZE = 8
CHANNELS = 2
DATA_LEN = 5
ANCHOR_TYPES = {
    "ReplicationFactor": [1],  # TODO: Enable replication once T12001 done.
    # Exception: Accl factor must divide batch size
    "AccumulationFactor": [4, 1],
    "Pipelining": [True, False],
    "ReturnType": ["FINAL", "ALL"]
}
# Learning rate 1 for easy comparison.
LEARNING_RATE = 1.0
# Strings for the anchors.
INPUT = "input"
WEIGHTS = "init_input"
ACTIVATION = "Reshape:0"
GRADIENT = popart.reservedGradientPrefix() + WEIGHTS
ACCL = popart.reservedAcclToAccumulatorPrefix(
) + popart.reservedGradientPrefix() + WEIGHTS


def dict_product(d):
    keys = d.keys()
    for element in itertools.product(*d.values()):
        yield dict(zip(keys, element))


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
        # TODO: use the tu.requires_ipu decorator
        if tu.ipu_available(ipus):
            device = tu.acquire_ipu(ipus)
        else:
            print("No IPUS available for test options.")
            return None, None
    else:
        device = popart.DeviceManager().createIpuModelDevice({'numIPUs': ipus})
    print("device: ", device)
    return opts, device


def return_anchors(anchorDict, label_array):
    """ Creates a model and returns the anchors given the provided options.
    """
    print(anchorDict)

    micro_batch_size = BATCH_SIZE // (anchorDict["AccumulationFactor"] *
                                      anchorDict["ReplicationFactor"])

    builder = popart.Builder()
    input_shape = [micro_batch_size, CHANNELS, DATA_LEN, DATA_LEN]

    data_shape = popart.TensorInfo("FLOAT", input_shape)
    lbl_shape = popart.TensorInfo("INT32", [micro_batch_size])

    ip = builder.addInputTensor(data_shape)
    lb = builder.addInputTensor(lbl_shape)

    w = builder.addInitializedInputTensor(
        np.ones([DATA_LEN, DATA_LEN], np.float32))
    b = builder.addInitializedInputTensor(np.ones([DATA_LEN], np.float32))
    o = builder.aiOnnx.matmul([ip, w])
    o = builder.aiOnnx.add([o, b])
    o = builder.reshape_const(
        builder.aiOnnx, [o],
        [micro_batch_size, CHANNELS * DATA_LEN * DATA_LEN])
    builder.addOutputTensor(o)

    anchors = {}
    anchors[o] = popart.AnchorReturnType(anchorDict["ReturnType"])
    anchors[GRADIENT] = popart.AnchorReturnType(anchorDict["ReturnType"])
    anchors[WEIGHTS] = popart.AnchorReturnType(anchorDict["ReturnType"])

    if anchorDict["AccumulationFactor"] > 1:
        anchors[ACCL] = popart.AnchorReturnType(anchorDict["ReturnType"])

    data_flow = popart.DataFlow(BATCHES_PER_STEP, anchors)

    opts, device = return_options(anchorDict)

    if device is None:
        return None

    session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                     dataFeed=data_flow,
                                     losses=[popart.NllLoss(o, lb, "loss")],
                                     optimizer=popart.SGD({
                                         "defaultLearningRate":
                                         (LEARNING_RATE, True),
                                         "defaultMomentum": (0.0, True),
                                         "defaultWeightDecay": (0.0, False),
                                         "defaultDampening": (0.0, True)
                                     }),
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

    inference_stepio = popart.PyStepIO(
        {
            ip: np.ones(input_shape, np.float32),
            lb: label_array
        }, anchors)
    session.weightsFromHost()
    session.optimizerFromHost()
    session.run(inference_stepio)

    return anchors


def test_all_anchor_returns():
    """
    Iterate through all combinations of options and test against
    the expected shapes:
    """
    for d in dict_product(ANCHOR_TYPES):
        micro_batch_size = BATCH_SIZE // (d["AccumulationFactor"] *
                                          d["ReplicationFactor"])

        label_array = np.zeros([BATCH_SIZE]).astype(np.int32)
        # Make a label array of increasing integers, for easy comparing outputs.
        i = 0
        with np.nditer(label_array, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = i
                i += 1
        # Get the anchors for this combo of options:
        dicts = return_anchors(d, label_array)
        expected_shapes = {
            # Weights
            WEIGHTS: [DATA_LEN, DATA_LEN],
            # Activation
            ACTIVATION: [micro_batch_size, CHANNELS * DATA_LEN * DATA_LEN],
            # Gradient.
            GRADIENT: [DATA_LEN, DATA_LEN],
            # Accl.
            ACCL: [DATA_LEN, DATA_LEN]
        }

        # Add in BPS if all batches are requested. (AnchorReturnType("ALL"))
        # Add in a replication dimension if needed.
        if d["ReplicationFactor"] > 1:
            for k in expected_shapes:
                expected_shapes[k] = [d["ReplicationFactor"]
                                      ] + expected_shapes[k]

        if d["AccumulationFactor"] > 1 and d["ReturnType"] is not "FINAL":
            for k in expected_shapes:  #[WEIGHTS, ACTIVATION, GRADIENT]:
                expected_shapes[k] = [d["AccumulationFactor"]
                                      ] + expected_shapes[k]

        if d["ReturnType"] == "ALL":
            # Then add BPS
            for k in expected_shapes:
                expected_shapes[k] = [BATCHES_PER_STEP] + expected_shapes[k]

        if dicts is not None:
            for a in dicts:
                print(a, ":")
                print("Actual : {}, Expected: {}".format(
                    list(dicts[a].shape), expected_shapes[a]))
                assert list(dicts[a].shape) == expected_shapes[a]
                print("CORRECT")
        else:
            # Case where invalid options are requested.
            print("Invalid test options")
