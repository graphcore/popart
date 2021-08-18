# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest
import torch
import itertools
from op_tester import op_tester

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


# Check that dropout is equal to an identity function in inference
@tu.requires_ipu_model
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
    op_tester.setPatterns(['OpToIdentity'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


# Check dropout with a ratio 0.0 (i.e. identity)
@tu.requires_ipu_model
def test_dropout_ratio0_testing(op_tester):
    d1 = np.random.rand(2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        (o, ) = builder.aiOnnx.dropout([i1], 1, 0.0, "test_dropout")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        t1 = torch.tensor(d1)
        return [t1]

    # Without identity pattern...
    op_tester.run(init_builder, reference, 'train')

    # ... and with identity pattern
    op_tester.setPatterns(['OpToIdentity'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


# Verify that the mask returned is correct when requesting 2 outputs.
# Manually calculate the result of the dropout using numpy and compare
@tu.requires_ipu_model
def test_dropout_training1():
    dsize = 10
    ratio = 0.2
    d1 = np.random.rand(dsize).astype(np.float32)

    builder = popart.Builder()
    ip = builder.addInputTensor(popart.TensorInfo("FLOAT", [dsize]))
    d__ip = popart.TensorId(popart.reservedGradientPrefix() + ip)

    [o1, o2] = builder.aiOnnx.dropout([ip], num_outputs=2, ratio=ratio)
    out = builder.aiGraphcore.identityloss([o1])
    builder.addOutputTensor(o1)
    builder.addOutputTensor(o2)

    session, anchors = get_session(anchorIds=[o1, o2, ip, d__ip],
                                   proto=builder.getModelProto(),
                                   device=tu.create_test_device(),
                                   loss=out)

    stepio = popart.PyStepIO({ip: d1}, anchors)
    session.run(stepio)

    # d1 * mask * (1/(1-ratio)) should give the same answer as popart implementation
    reference = d1 * anchors[o2] * (1 / (1 - ratio))

    assert (np.isclose(anchors[o1], reference)).all()


# We cannot directly test popart dropout vs pytorch here, because they have
# different (and not user-controllable) random seeds. But there are a
# number of tests we can perform to make sure that dropout behaves as
# expected in training:


# 1. Check the non-masked elements of dropout are scaled in the same way as in
#    pytorch
# 2. Check that the expected ratio of inputs to dropout are masked
@tu.requires_ipu_model
def test_dropout_training3():
    dsize = 10000  # large input size to make statistical assumptions accurate
    ratio = 0.2
    session, ip, out, d__ip, anchors = get_dropout_session(dsize=dsize,
                                                           ratio=ratio)

    # Ensure inputs in range [1.0, 2.0] to ensure comparing with 0 is valid
    ip_data = np.random.random_sample(dsize).astype(np.float32) + 1
    stepio = popart.PyStepIO({ip: ip_data}, anchors)

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
    assert (np.isclose(onnxDropoutProportion,
                       torchDropoutProportion,
                       atol=0.05))
    assert (np.isclose(onnxDropoutProportion, 1 - ratio, atol=0.05))


# Check that the forward activations and backwards gradients are
# masked in the same way
@tu.requires_ipu
def test_dropout_training4():
    dsize = 10
    ratio = 0.2
    builder = popart.Builder()
    ip = builder.addInputTensor(popart.TensorInfo("FLOAT", [dsize, dsize]))
    d__ip = popart.TensorId(popart.reservedGradientPrefix() + ip)
    [d1] = builder.aiOnnx.dropout([ip], num_outputs=1, ratio=ratio)

    # Matmul to change the layout -- ensures we are testing the dependency
    # of random mask on the layout of the 'reference' dropout tensor
    w = builder.addInitializedInputTensor(np.ones([dsize, dsize], np.float32))
    out = builder.aiOnnx.matmul([d1, w])
    out = builder.aiGraphcore.identityloss([out])
    builder.addOutputTensor(out)

    device = tu.create_test_device()

    session, anchors = get_session(anchorIds=[d1, d__ip],
                                   proto=builder.getModelProto(),
                                   device=device,
                                   loss=out)

    # Ensure inputs in range [1.0, 2.0] to ensure comparing with 0 is valid
    ip_data = np.random.random_sample((dsize, dsize)).astype(np.float32) + 1
    stepio = popart.PyStepIO({ip: ip_data}, anchors)

    session.run(stepio)

    for fwdEl, bwdEl in zip(np.ndarray.flatten(anchors[d1]),
                            np.ndarray.flatten(anchors[d__ip])):
        if fwdEl == 0:
            assert bwdEl == 0
        if bwdEl != 0:
            assert fwdEl != 0


# Test that a different mask is used every time session.run is called.
@tu.requires_ipu
def test_dropout_training_randomness():
    dsize = 100
    session, ip, out, d__ip, anchors = get_dropout_session(dsize=dsize,
                                                           use_ipu=True)

    ip_data = np.random.random_sample(dsize).astype(np.float32)
    stepio = popart.PyStepIO({ip: ip_data}, anchors)

    session.setRandomSeed(0)

    session.run(stepio)
    # need to copy the anchor as the next call to run will overwrite the data
    run1_out = np.copy(anchors[out])

    session.run(stepio)
    run2_out = np.copy(anchors[out])

    assert not np.array_equal(run1_out, run2_out)


# Test for repeatable randomness (i.e. if you run the same training session
# on the same input data, with the same seed the same, random mask is applied)
@tu.requires_ipu
def test_dropout_training_set_seed():
    dsize = 100
    session, ip, out, d__ip, anchors = get_dropout_session(dsize=dsize,
                                                           use_ipu=True)

    ip_data = np.random.random_sample(dsize).astype(np.float32)
    stepio = popart.PyStepIO({ip: ip_data}, anchors)

    session.setRandomSeed(7)
    session.run(stepio)
    # need to copy the anchor as the next call to run will overwrite the data
    run1_out = np.copy(anchors[out])

    session.setRandomSeed(7)
    session.run(stepio)
    run2_out = np.copy(anchors[out])

    assert (np.array_equal(run1_out, run2_out))


# Test that the dropout mask is different round each repeat loop
# when repeat-count (batches-per-step) > 1
@tu.requires_ipu
def test_dropout_training6():
    dsize = 100
    bps = 2
    session, ip, out, d__ip, anchors = get_dropout_session(dsize=dsize,
                                                           bps=bps,
                                                           use_ipu=True)

    # Same data for each batch
    ip_data_bps1 = np.random.random_sample(dsize).astype(np.float32)
    ip_data = np.array([
        ip_data_bps1,
    ] * bps)

    stepio = popart.PyStepIO({ip: ip_data}, anchors)

    session.run(stepio)
    assert (np.array_equal(anchors[out][0], anchors[out][1]) is not True)


# Test that two dropout ops with the same input, in the same graph,
# have a different mask
@tu.requires_ipu
def test_dropout_training7():
    dsize = 100
    ratio = 0.2
    builder = popart.Builder()
    ip = builder.addInputTensor(popart.TensorInfo("FLOAT", [dsize]))
    d__ip = popart.TensorId(popart.reservedGradientPrefix() + ip)
    [d1] = builder.aiOnnx.dropout([ip], num_outputs=1, ratio=ratio)
    [d2] = builder.aiOnnx.dropout([ip], num_outputs=1, ratio=ratio)
    out = builder.aiOnnx.add([d1, d2])
    out = builder.aiGraphcore.identityloss([out])
    builder.addOutputTensor(out)

    if tu.ipu_available():
        device = tu.create_test_device()
    else:
        pytest.skip("Test needs to run on IPU, but none are available")

    session, anchors = get_session(anchorIds=[d1, d2],
                                   proto=builder.getModelProto(),
                                   device=device,
                                   loss=out)

    # Same data for each batch
    ip_data = np.random.random_sample(dsize).astype(np.float32)
    stepio = popart.PyStepIO({ip: ip_data}, anchors)

    session.run(stepio)
    assert (np.array_equal(anchors[d1], anchors[d2]) is not True)


# Verify that popart errors out properly when ratio is not in (0,1)
@tu.requires_ipu_model
def test_dropout_training8(op_tester):
    d1 = np.random.rand(10).astype(np.float32)
    ratio = 100 * np.random.rand(1)[0] + 1

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        [o1] = builder.aiOnnx.dropout([i1], num_outputs=1, ratio=ratio)
        builder.addOutputTensor(o1)
        return [o1, popart.TensorId(popart.reservedGradientPrefix() + i1)]

    def reference(ref_data):
        dropout = torch.nn.Dropout()
        out = dropout(torch.tensor(d1))
        return [out]

    # This should trigger the ratio error.
    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'train')

    assert (e_info.value.args[0].endswith(
        "Please use a value in the interval [0,1)"))


# check we get a different dropout from each replicated graph
@tu.requires_ipu
def test_dropout_training_replicated():
    replication_factor = 4
    dsize = 10
    session, ip, out, d__ip, anchors = get_replicated_dropout_session(
        dsize=dsize,
        num_layers=1,
        ratio=0.3,
        replication_factor=replication_factor)

    ip_data = np.ones([replication_factor, dsize], dtype=np.float32)
    stepio = popart.PyStepIO({ip: ip_data}, anchors)

    session.run(stepio)

    o = anchors[out]
    for ai, bi in itertools.combinations(
        [i for i in range(replication_factor)], 2):
        a = o[ai]
        b = o[bi]
        assert not np.allclose(a, b)


# Check set seed when using replicated graphs
@tu.requires_ipu
def test_dropout_training_replicated_repeatable():
    replication_factor = 4
    dsize = 10
    session, ip, out, d__ip, anchors = get_replicated_dropout_session(
        dsize=dsize,
        num_layers=1,
        ratio=0.3,
        replication_factor=replication_factor)

    ip_data = np.ones([replication_factor, dsize], dtype=np.float32)
    stepio = popart.PyStepIO({ip: ip_data}, anchors)

    session.setRandomSeed(7)
    session.run(stepio)
    ref_out = np.copy(anchors[out])
    ref_d__ip = np.copy(anchors[d__ip])

    # Another call should produce different results
    session.run(stepio)
    assert not np.array_equal(ref_out, anchors[out])
    assert not np.array_equal(ref_d__ip, anchors[d__ip])

    # Resetting the seed should give the same results as the first run
    session.setRandomSeed(7)
    session.run(stepio)
    assert np.array_equal(ref_out, anchors[out])
    assert np.array_equal(ref_d__ip, anchors[d__ip])


# Check that all micro batches use different seeds when using multiple
# batches per step and replicated graphs.
@tu.requires_ipu
def test_replicated_with_multiple_batches_per_step():
    replication_factor = 4
    dsize = 100
    batches_per_step = 2
    session, ip, out, d__ip, anchors = get_replicated_dropout_session(
        dsize=dsize,
        num_layers=1,
        ratio=0.3,
        replication_factor=replication_factor,
        batches_per_step=batches_per_step)

    ip_data = np.ones([batches_per_step, replication_factor, dsize],
                      dtype=np.float32)
    stepio = popart.PyStepIO({ip: ip_data}, anchors)

    session.run(stepio)
    ref_out = np.copy(anchors[out])

    # Another call should produce different results
    session.run(stepio)

    o = anchors[out]
    micro_batches = []
    for batch_index in range(batches_per_step):
        for replication_index in range(replication_factor):
            x = o[batch_index][replication_index]
            micro_batches.append(x)

    # Check that none of the micro batch results are the same
    for ai, bi in itertools.combinations(
        [i for i in range(len(micro_batches))], 2):
        a = micro_batches[ai]
        b = micro_batches[bi]
        assert not np.allclose(a, b)


def get_replicated_dropout_session(replication_factor=4,
                                   dsize=10,
                                   num_layers=1,
                                   ratio=0.3,
                                   batches_per_step=1,
                                   seed=0):
    builder = popart.Builder()
    ip = builder.addInputTensor(popart.TensorInfo("FLOAT", [dsize]))
    d__ip = popart.TensorId(popart.reservedGradientPrefix() + ip)
    out = ip
    for layer in range(num_layers):
        [out] = builder.aiOnnx.dropout([out], num_outputs=1, ratio=ratio)
    loss = builder.aiGraphcore.identityloss([out])
    builder.addOutputTensor(loss)

    device = tu.create_test_device(replication_factor)

    dfAnchors = [out, ip, d__ip]
    dfAnchors = {i: popart.AnchorReturnType("All") for i in dfAnchors}

    opts = popart.SessionOptions()
    opts.enableReplicatedGraphs = True
    opts.replicatedGraphCount = replication_factor

    session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                     dataFlow=popart.DataFlow(
                                         batches_per_step, dfAnchors),
                                     optimizer=popart.ConstSGD(0.1),
                                     loss=loss,
                                     userOptions=opts,
                                     deviceInfo=device)

    session.prepareDevice()
    session.setRandomSeed(seed)
    session.weightsFromHost()
    anchors = session.initAnchorArrays()
    return session, ip, out, d__ip, anchors


def get_dropout_session(dsize=100,
                        ratio=0.2,
                        bps=1,
                        use_ipu=False,
                        num_layers=1,
                        seed=0):
    builder = popart.Builder()
    ip = builder.addInputTensor(popart.TensorInfo("FLOAT", [dsize]))
    d__ip = popart.TensorId(popart.reservedGradientPrefix() + ip)
    out = ip
    for numl in range(num_layers):
        [out] = builder.aiOnnx.dropout([out], num_outputs=1, ratio=ratio)
    loss = builder.aiGraphcore.identityloss([out])
    builder.addOutputTensor(loss)

    device = tu.create_test_device()

    session, anchors = get_session(anchorIds=[out, ip, d__ip],
                                   proto=builder.getModelProto(),
                                   device=device,
                                   loss=loss,
                                   bps=bps,
                                   seed=seed)

    return session, ip, out, d__ip, anchors


def get_session(anchorIds, proto, device, loss, bps=1, seed=0):
    dfAnchors = {}
    for anchorId in anchorIds:
        dfAnchors.update({anchorId: popart.AnchorReturnType("All")})

    session = popart.TrainingSession(fnModel=proto,
                                     dataFlow=popart.DataFlow(bps, dfAnchors),
                                     optimizer=popart.ConstSGD(0.1),
                                     loss=loss,
                                     patterns=popart.Patterns(
                                         popart.PatternsLevel.All),
                                     deviceInfo=device)

    session.prepareDevice()
    session.setRandomSeed(seed)
    session.weightsFromHost()
    anchors = session.initAnchorArrays()
    return session, anchors
