# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart

# importing test_session and test_util requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


@tu.requires_ipu
def test_pipelined_dropout():
    # The test can be run without pipelining for debugging.
    def test(do_pipelining, do_sharding):
        dsize = 10
        ratio = 0.5
        if do_sharding:
            ipus = 4
        else:
            ipus = 1
        layers = 4
        batches_per_step = 7

        # Ensure inputs in range [1.0, 2.0] to ensure comparing with 0 is valid
        ip_shape = [dsize]
        ip_data = np.full([batches_per_step] + ip_shape, 1).astype(np.float32)

        dropouts = []
        dropoutGrads = []
        dropoutInputs = []
        dropoutOutputs = []

        builder = popart.Builder()
        ip = builder.addInputTensor(popart.TensorInfo("FLOAT", ip_shape))

        def add_layer(layer_input, vgraph_num):
            # This is to get the output of the dropout in the bwd pass.
            # D_next_layer_in also includes the gradient of the AddOp.
            identity0 = builder.aiOnnx.identity([layer_input])
            if do_sharding:
                builder.virtualGraph(identity0, vgraph_num)

            [dropout0] = builder.aiOnnx.dropout([identity0],
                                                num_outputs=1,
                                                ratio=ratio)
            if do_sharding:
                builder.virtualGraph(dropout0, vgraph_num)

            # the input to the forward pass dropout
            dropoutInputs.append(identity0)
            # the input to the backward pass dropout
            dropoutInputs.append(popart.reservedGradientPrefix() + dropout0)
            # the output of the backward pass dropout
            dropoutGrads.append(popart.reservedGradientPrefix() + identity0)
            # the output of the forward pass dropout
            dropouts.append(dropout0)

            # This ensures the all input elements to the dropouts, in both
            # the forward and backward passes, will be non-zero.
            add0 = builder.aiOnnx.add([layer_input, dropout0])
            if do_sharding:
                builder.virtualGraph(add0, vgraph_num)

            return add0

        # construct a graph of `layers` number of layers
        # with each layer on a different IPU.
        next_layer_in = ip
        for vgraph in range(layers):
            next_layer_in = add_layer(next_layer_in, vgraph)
        out = next_layer_in
        loss = builder.aiGraphcore.identityloss([out])
        builder.virtualGraph(loss, layers - 1)
        builder.addOutputTensor(loss)

        device = tu.create_test_device(numIpus=ipus)

        dfAnchors = {}
        for x in dropouts + dropoutGrads + dropoutInputs:
            dfAnchors[x] = popart.AnchorReturnType("All")

        dataFlow = popart.DataFlow(batches_per_step, dfAnchors)

        userOptions = popart.SessionOptions()
        if do_sharding:
            userOptions.virtualGraphMode = popart.VirtualGraphMode.Manual
        userOptions.enablePipelining = do_pipelining

        # Inplacing can differ between pipelining & non-pipelining,
        # which can cause the random behaviour to change
        # TODO: T32086
        patterns = popart.Patterns(popart.PatternsLevel.Default)
        patterns.InPlace = False

        session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                         dataFlow=dataFlow,
                                         optimizer=popart.ConstSGD(0.1),
                                         loss=loss,
                                         userOptions=userOptions,
                                         patterns=patterns,
                                         deviceInfo=device)

        session.prepareDevice()
        session.weightsFromHost()
        anchors = session.initAnchorArrays()
        session.setRandomSeed(0)

        stepio = popart.PyStepIO({ip: ip_data}, anchors)

        session.run(stepio)

        print(anchors.keys())

        # Check that none of the elements of the dropout inputs are zero
        for tid in dropoutInputs:
            x = anchors[tid]
            print(f'{tid}: {x}')
            zero = np.zeros(x.shape)
            assert not np.any(np.equal(x, zero)), \
                   f'Some elements of dropout input {tid} are zero'

        print()

        # For each dropout, check that the masked out elements are the same
        # in the forward and backward passes.
        for fwdId, bwdId in zip(dropouts, dropoutGrads):
            print(f'{fwdId}:\n{np.sign(anchors[fwdId])}')
            print(f'{bwdId}:\n{np.sign(anchors[bwdId])}')
            lhs = np.sign(anchors[fwdId])
            rhs = np.sign(anchors[bwdId])
            assert np.array_equal(lhs, rhs), \
                   f'{fwdId} and {bwdId} did not use the same dropout mask'
            print()

        return anchors


@tu.requires_ipu
def test_pipelined_recomputed_dropout():
    dsize = 10
    ratio = 0.5
    ipus = 4
    layers = 4
    batches_per_step = 7

    # Ensure inputs in range [1.0, 2.0] to ensure comparing with 0 is valid
    ip_shape = [dsize]
    ip_data = np.full([batches_per_step] + ip_shape, 1).astype(np.float32)

    dropouts = []
    dropoutGrads = []
    dropoutInputs = []
    dropoutOutputs = []

    builder = popart.Builder()
    ip = builder.addInputTensor(popart.TensorInfo("FLOAT", ip_shape))

    def add_layer(layer_input, vgraph_num):
        # This is to get the output of the dropout in the bwd pass.
        # D_next_layer_in also includes the gradient of the AddOp.
        identity0 = builder.aiOnnx.identity([layer_input])
        builder.virtualGraph(identity0, vgraph_num)

        [dropout0] = builder.aiOnnx.dropout([identity0],
                                            num_outputs=1,
                                            ratio=ratio)
        builder.virtualGraph(dropout0, vgraph_num)

        # the input to the forward pass dropout
        dropoutInputs.append(identity0)
        # the input to the backward pass dropout
        dropoutInputs.append(popart.reservedGradientPrefix() + dropout0)
        # the output of the backward pass dropout
        dropoutGrads.append(popart.reservedGradientPrefix() + identity0)
        # the output of the forward pass dropout
        dropouts.append(dropout0)

        relu0 = builder.aiOnnx.relu([dropout0])
        builder.virtualGraph(relu0, vgraph_num)

        # This ensures the all input elements to the dropouts, in both
        # the forward and backward passes, will be non-zero.
        add0 = builder.aiOnnx.add([layer_input, dropout0])
        builder.virtualGraph(add0, vgraph_num)

        return add0

    # construct a graph of `layers` number of layers
    # with each layer on a different IPU.
    next_layer_in = ip
    for vgraph in range(layers):
        next_layer_in = add_layer(next_layer_in, vgraph)
    out = next_layer_in

    device = tu.create_test_device(numIpus=ipus)

    dfAnchors = {}
    for x in dropouts + dropoutGrads + dropoutInputs:
        dfAnchors[x] = popart.AnchorReturnType("All")

    dataFlow = popart.DataFlow(batches_per_step, dfAnchors)

    loss = builder.aiGraphcore.identityloss([out])
    builder.virtualGraph(loss, layers - 1)

    userOptions = popart.SessionOptions()
    userOptions.virtualGraphMode = popart.VirtualGraphMode.Manual
    userOptions.enablePipelining = True
    userOptions.autoRecomputation = popart.RecomputationType.Pipeline

    # Inplacing can differ between pipelining & non-pipelining,
    # which can cause the random behaviour to change
    # TODO: T32086
    patterns = popart.Patterns(popart.PatternsLevel.Default)
    patterns.InPlace = False

    session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                     dataFlow=dataFlow,
                                     optimizer=popart.ConstSGD(0.1),
                                     loss=loss,
                                     userOptions=userOptions,
                                     patterns=patterns,
                                     deviceInfo=device)

    session.prepareDevice()
    session.weightsFromHost()
    anchors = session.initAnchorArrays()
    session.setRandomSeed(0)

    stepio = popart.PyStepIO({ip: ip_data}, anchors)

    session.run(stepio)

    print(anchors.keys())

    # Check that none of the elements of the dropout inputs are zero
    for tid in dropoutInputs:
        x = anchors[tid]
        print(f'{tid}: {x}')
        zero = np.zeros(x.shape)
        assert not np.any(np.equal(x, zero)), \
               f'Some elements of dropout input {tid} are zero'

    print()

    # For each dropout, check that the masked out elements are the same
    # in the forward and backward passes.
    for fwdId, bwdId in zip(dropouts, dropoutGrads):
        print(f'{fwdId}:\n{np.sign(anchors[fwdId])}')
        print(f'{bwdId}:\n{np.sign(anchors[bwdId])}')
        lhs = np.sign(anchors[fwdId])
        rhs = np.sign(anchors[bwdId])
        assert np.array_equal(lhs, rhs), \
               f'{fwdId} and {bwdId} did not use the same dropout mask'
        print()
