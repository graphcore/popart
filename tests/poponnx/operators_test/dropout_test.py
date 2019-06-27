import numpy as np
import poponnx
import pytest
import torch
import itertools
from op_tester import op_tester


# Check that dropout is equal to an identity function in inference
def test_dropout_testing(op_tester):
    d1 = np.random.rand(2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        (o, ) = builder.aiOnnx.dropout([i1], 1, 0.5, "test_dropout")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        t1 = torch.tensor(d1)
        return [t1]

    # Without identity pattern...
    op_tester.run(init_builder, reference, 'infer')

    # ... and with identity pattern
    op_tester.passes = ['OpToIdentity', 'DropoutGradOp']
    op_tester.run(init_builder, reference, 'infer')


# Verify when in training mode that dropout cannot be used with !=1 outputs.
def test_dropout_training1(op_tester):
    d1 = np.random.rand(10).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        [o1, o2] = builder.aiOnnx.dropout([i1], num_outputs=2)
        builder.addOutputTensor(o1)
        return [o1, o2]

    def reference(ref_data):
        dropout = torch.nn.Dropout()
        out = dropout(torch.tensor(d1))
        return [out]

    op_tester.passes = ['DropoutGradOp']
    with pytest.raises(poponnx.poponnx_exception) as e_info:
        op_tester.run(init_builder, reference, 'train')

    assert (e_info.value.args[0].find(
        "In Poponnx the Dropout op only supports a single output tensor") !=
            -1)


# Verify that poponnx errors our properly when the DropoutGradOp is
# not applied
def test_dropout_training2(op_tester):
    d1 = np.random.rand(10).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        [o1] = builder.aiOnnx.dropout([i1], num_outputs=1)
        builder.addOutputTensor(o1)
        return [o1, poponnx.reservedGradientPrefix() + i1]

    def reference(ref_data):
        dropout = torch.nn.Dropout()
        out = dropout(torch.tensor(d1))
        return [out]

    # 'DropoutGradOp' pattern not applied
    with pytest.raises(poponnx.poponnx_exception) as e_info:
        op_tester.run(init_builder, reference, 'train')

    assert (e_info.value.args[0].startswith(
        "DropoutGradOp should be optimised out"))


# We cannot directly test poponnx dropout vs pytorch here, because they have
# different (and not user-controllable) random seeds. But there are a
# number of tests we can perform to make sure that dropout behaves as
# expected in training:


# 1. Check the non-masked elements of dropout are scaled in the same way as in
#    pytorch
# 2. Check that the expected ratio of inputs to dropout are masked
def test_dropout_training3():
    dsize = 10000  # large input size to make statistical assumptions accurate
    ratio = 0.2
    session, ip, out, d__ip, anchors = get_dropout_session(
        dsize=dsize, ratio=ratio)

    # Ensure inputs in range [1.0, 2.0] to ensure comparing with 0 is valid
    ip_data = np.random.random_sample(dsize).astype(np.float32) + 1
    stepio = poponnx.PyStepIO({ip: ip_data}, anchors)

    session.run(stepio)

    t1 = torch.tensor(ip_data)
    dropout = torch.nn.Dropout(p=ratio)
    torchOut = dropout(t1).numpy()

    # 1.
    for onnxEl, torchEl in zip(anchors[out], torchOut):
        if onnxEl != 0 and torchEl != 0:
            assert np.isclose(onnxEl, torchEl)

    # 2.
    onnxDropoutProportion = np.count_nonzero(anchors[out]) / dsize
    torchDropoutProportion = np.count_nonzero(torchOut) / dsize
    assert (np.isclose(
        onnxDropoutProportion, torchDropoutProportion, atol=0.05))
    assert (np.isclose(onnxDropoutProportion, 1 - ratio, atol=0.05))


## TODO T8803 : requires hardware or a sim device
# Check that the forward activations and backwards gradients are
# masked in the same way
def test_dropout_training4():
    dsize = 10
    ratio = 0.2
    builder = poponnx.Builder()
    ip = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [dsize, dsize]))
    d__ip = poponnx.reservedGradientPrefix() + ip
    [d1] = builder.aiOnnx.dropout([ip], num_outputs=1, ratio=ratio)

    # Matmul to change the layout -- ensures we are testing the dependency
    # of random mask on the layout of the 'reference' dropout tensor
    w = builder.addInitializedInputTensor(np.ones([dsize, dsize], np.float32))
    out = builder.aiOnnx.matmul([d1, w])
    builder.addOutputTensor(out)

    device = poponnx.DeviceManager().acquireAvailableDevice()
    if device is None:
        pytest.skip("Test needs to run on IPU, but none are available")

    session, anchors = get_session(
        anchorIds=[d1, d__ip],
        proto=builder.getModelProto(),
        device=device,
        output=out)

    # Ensure inputs in range [1.0, 2.0] to ensure comparing with 0 is valid
    ip_data = np.random.random_sample((dsize, dsize)).astype(np.float32) + 1
    stepio = poponnx.PyStepIO({ip: ip_data}, anchors)

    session.run(stepio)

    for fwdEl, bwdEl in zip(
            np.ndarray.flatten(anchors[d1]),
            np.ndarray.flatten(anchors[d__ip])):
        if fwdEl == 0:
            assert bwdEl == 0
        if bwdEl != 0:
            assert fwdEl != 0


# Test for repeatable randomness (i.e. if you run the same training session
# on the same input data, the same random mask is applied)
def test_dropout_training5():
    dsize = 100
    session, ip, out, d__ip, anchors = get_dropout_session(dsize=dsize)

    ip_data = np.random.random_sample(dsize).astype(np.float32)
    stepio = poponnx.PyStepIO({ip: ip_data}, anchors)

    session.run(stepio)
    run1_out = anchors[out]

    session.run(stepio)
    run2_out = anchors[out]

    assert (np.array_equal(run1_out, run2_out))


## TODO T8803 : requires hardware or a sim device
# Test that the dropout mask is different round each repeat loop
# when repeat-count (batches-per-step) > 1
def test_dropout_training6():
    dsize = 100
    bps = 2
    session, ip, out, d__ip, anchors = get_dropout_session(
        dsize=dsize, bps=bps, use_ipu=True)

    # Same data for each batch
    ip_data_bps1 = np.random.random_sample(dsize).astype(np.float32)
    ip_data = np.array([
        ip_data_bps1,
    ] * bps)

    stepio = poponnx.PyStepIO({ip: ip_data}, anchors)

    session.run(stepio)
    assert (np.array_equal(anchors[out][0], anchors[out][1]) is not True)


## TODO T8803 : requires hardware or a sim device
# Test that two dropout ops with the same input, in the same graph,
# have a different mask
def test_dropout_training7():
    dsize = 100
    ratio = 0.2
    builder = poponnx.Builder()
    ip = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [dsize]))
    d__ip = poponnx.reservedGradientPrefix() + ip
    [d1] = builder.aiOnnx.dropout([ip], num_outputs=1, ratio=ratio)
    [d2] = builder.aiOnnx.dropout([ip], num_outputs=1, ratio=ratio)
    out = builder.aiOnnx.add([d1, d2])
    builder.addOutputTensor(out)

    device = poponnx.DeviceManager().acquireAvailableDevice()
    if device is None:
        pytest.skip("Test needs to run on IPU, but none are available")

    session, anchors = get_session(
        anchorIds=[d1, d2],
        proto=builder.getModelProto(),
        device=device,
        output=out)

    # Same data for each batch
    ip_data = np.random.random_sample(dsize).astype(np.float32)
    stepio = poponnx.PyStepIO({ip: ip_data}, anchors)

    session.run(stepio)
    assert (np.array_equal(anchors[d1], anchors[d2]) is not True)


def test_dropout_training_replicated(op_tester):
    ip_data = np.random.random_sample(100).astype(np.float32)
    replication_factor = 4

    def init_builder(builder):
        i1 = builder.addInputTensor(ip_data)
        [o] = builder.aiOnnx.dropout([i1], num_outputs=1, ratio=0.2)
        builder.addOutputTensor(o)

        return [o]

    def check_result(ref_data):
        o = ref_data.getOutputTensor(0)
        # np.sign on the output of dropout returns an array of zeros and ones
        o = np.sign(o)

        for ai, bi in itertools.combinations(
            [i for i in range(replication_factor)], 2):
            print(f'Checking o[{ai}] is not equal to o[{bi}]')
            a = o[ai]
            b = o[bi]
            assert not np.allclose(a, b)

        return [None]

    device = poponnx.DeviceManager().acquireAvailableDevice(replication_factor)
    if device is None:
        pytest.skip("Test needs to run on IPU, but none are available")

    op_tester.options.enableReplicatedGraphs = True
    op_tester.options.replicatedGraphCount = replication_factor
    op_tester.device = device
    op_tester.numIPUs = replication_factor
    session = op_tester.run(init_builder, check_result, 'train')


def get_dropout_session(dsize=100,
                        ratio=0.2,
                        bps=1,
                        use_ipu=False,
                        num_layers=1):
    builder = poponnx.Builder()
    ip = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [dsize]))
    d__ip = poponnx.reservedGradientPrefix() + ip
    out = ip
    for numl in range(num_layers):
        [out] = builder.aiOnnx.dropout([out], num_outputs=1, ratio=ratio)
    builder.addOutputTensor(out)

    if use_ipu is True:
        device = poponnx.DeviceManager().acquireAvailableDevice()
        if device is None:
            pytest.skip("Test needs to run on IPU, but none are available")
    else:
        device = poponnx.DeviceManager().createCpuDevice()

    session, anchors = get_session(
        anchorIds=[out, ip, d__ip],
        proto=builder.getModelProto(),
        device=device,
        output=out,
        bps=bps)

    return session, ip, out, d__ip, anchors


def get_session(anchorIds, proto, device, output, bps=1):
    dfAnchors = {}
    for anchorId in anchorIds:
        dfAnchors.update({anchorId: poponnx.AnchorReturnType("ALL")})

    session = poponnx.TrainingSession(
        fnModel=proto,
        dataFeed=poponnx.DataFlow(bps, dfAnchors),
        optimizer=poponnx.ConstSGD(0.1),
        losses=[poponnx.L1Loss(output, "l1LossVal", 0.1)],
        passes=poponnx.Patterns(poponnx.PatternsLevel.ALL),
        deviceInfo=device)

    session.prepareDevice()
    session.weightsFromHost()
    anchors = session.initAnchorArrays()

    return session, anchors
