import numpy as np
import popart
import pytest

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu
from operators_test.op_tester import op_tester


# Replication 1, in some tests, but still requires an IPU as prefetching is only
# enabled on IPUs.
def get_model(batches_per_step, replication_factor, batch_size, channels,
              data_len, synthetic_data):

    micro_batch_size = batch_size // (replication_factor)

    builder = popart.Builder()
    input_shape = [micro_batch_size, channels, data_len, data_len]

    data_shape = popart.TensorInfo("FLOAT", input_shape)
    lbl_shape = popart.TensorInfo("INT32", [micro_batch_size])
    w = builder.addInitializedInputTensor(
        np.random.random_sample(input_shape).astype(np.float32))

    ip = builder.addInputTensor(data_shape, "input")
    lb = builder.addInputTensor(lbl_shape, "label")

    a = builder.aiOnnx.matmul([ip, w])
    o = builder.reshape_const(
        builder.aiOnnx, [a],
        [micro_batch_size, channels * data_len * data_len])
    o = builder.aiOnnx.relu([o])
    o = builder.aiOnnx.softmax([o])
    builder.addOutputTensor(o)

    art = popart.AnchorReturnType("ALL")
    data_flow = popart.DataFlow(batches_per_step, {ip: art, lb: art})

    opts = popart.SessionOptions()
    ipus = 1
    if replication_factor > 1:
        opts.replicatedGraphCount = replication_factor
        opts.enableReplicatedGraphs = True
        ipus *= replication_factor
    device = popart.DeviceManager().acquireAvailableDevice(ipus)

    if synthetic_data:
        opts.ignoreData = True

    if device is None:
        pytest.skip("Test needs to run on IPU, but none are available")

    session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                     dataFeed=data_flow,
                                     losses=[popart.NllLoss(o, lb, "loss")],
                                     optimizer=popart.ConstSGD(1.0),
                                     userOptions=opts,
                                     deviceInfo=device)

    session.prepareDevice()

    label_shape = [batch_size]

    if replication_factor > 1:
        input_shape = [replication_factor] + input_shape
        label_shape = [replication_factor] + label_shape
    if batches_per_step > 1:
        input_shape = [batches_per_step] + input_shape
        label_shape = [batches_per_step] + label_shape

    anchors = session.initAnchorArrays()

    return session, anchors, input_shape, label_shape


def run_model(session, anchors, in_array, label_array):
    stepio = popart.PyStepIO({
        "input": in_array,
        "label": label_array
    }, anchors)
    session.weightsFromHost()

    session.run(stepio)

    return anchors["input"], anchors["label"]


def run_test(batches_per_step, replication_factor, batch_size, channels,
             data_len, steps):
    micro_batch_size = batch_size // replication_factor

    session, anchors, input_shape, label_shape = get_model(
        batches_per_step=batches_per_step,
        replication_factor=replication_factor,
        batch_size=batch_size,
        channels=channels,
        data_len=data_len,
        synthetic_data=False)

    for step in range(steps):
        print("Step:", step)
        in_array = np.random.random_sample(input_shape).astype(np.float32)
        label_array = np.random.randint(low=0, high=20,
                                        size=label_shape).astype(np.int32)
        # Only provide one session.run's worth of data.
        in_anchor, label_anchor = run_model(session, anchors, in_array,
                                            label_array)

        # Returned anchors will be of shape
        # [bps, micro_batch_size, channels, data_len, data_len]
        for batch in range(batches_per_step):
            print("Batch:", batch)
            # If bps == 1, the dimension doesn't exist, so we pass none to ignore
            # that dimension.
            if batches_per_step == 1: batch = None
            for mini_batch in range(micro_batch_size):
                # Inputs check:
                assert np.allclose(in_array[batch, mini_batch, :, :, :],
                                   in_anchor[batch, mini_batch, :, :, :])
                # Labels check
                assert np.allclose(label_array[batch, mini_batch],
                                   label_anchor[batch, mini_batch])


# Batch size > 1
@tu.requires_ipu
def test_prefetch_0():
    args = dict(batches_per_step=1,
                replication_factor=1,
                batch_size=8,
                channels=2,
                data_len=2,
                steps=5)
    run_test(**args)


# BPS > 1
@tu.requires_ipu
def test_prefetch_1():
    args = dict(batches_per_step=7,
                replication_factor=1,
                batch_size=1,
                channels=2,
                data_len=3,
                steps=5)
    run_test(**args)


# BPS > 1, Batch size > 1
@tu.requires_ipu
def test_prefetch_2():
    args = dict(batches_per_step=7,
                replication_factor=1,
                batch_size=16,
                channels=2,
                data_len=3,
                steps=5)
    run_test(**args)


# Run a test with synthetic data. We don't expect the anchors to match the input
# data, which is ignored. If these do match, it means a stream has been erroneously connected.
# But we do expect the model to compile and run.
def run_synthetic_test(batches_per_step, replication_factor, batch_size,
                       channels, data_len, steps):
    micro_batch_size = batch_size // replication_factor

    session, anchors, input_shape, label_shape = get_model(
        batches_per_step=batches_per_step,
        replication_factor=replication_factor,
        batch_size=batch_size,
        channels=channels,
        data_len=data_len,
        synthetic_data=True)

    for step in range(steps):
        print("Step:", step)
        in_array = np.random.random_sample(input_shape).astype(np.float32)
        label_array = np.random.randint(low=0, high=20,
                                        size=label_shape).astype(np.int32)
        # Only provide one session.run's worth of data.
        in_anchor, label_anchor = run_model(session, anchors, in_array,
                                            label_array)

        # Returned anchors will be of shape
        # [bps, micro_batch_size, channels, data_len, data_len]
        for batch in range(batches_per_step):
            print("Batch:", batch)
            # If bps == 1, the dimension doesn't exist, so we pass none to ignore
            # that dimension.
            if batches_per_step == 1: batch = None
            for mini_batch in range(micro_batch_size):
                # Inputs check:
                assert not np.allclose(in_array[batch, mini_batch, :, :, :],
                                       in_anchor[batch, mini_batch, :, :, :])
            # Labels check. Comparing individual mini batches you occasionally
            # get a single mini batch matching. So we just check the entire batch.
            assert not np.allclose(label_array[batch], label_anchor[batch])


# BPS > 1, Batch size > 1, synthetic data.
@tu.requires_ipu
def test_prefetch_synthetic():
    args = dict(batches_per_step=7,
                replication_factor=1,
                batch_size=16,
                channels=2,
                data_len=3,
                steps=5)
    run_synthetic_test(**args)
