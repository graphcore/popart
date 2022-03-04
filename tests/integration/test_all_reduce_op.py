# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart
import popxl
import popxl.ops as ops
import numpy as np

import test_util as tu


def run_ir(ir, h2d_streams, d2h_ids, n_ipus):
    ir_ = ir._pb_ir

    dataFlow = popart.DataFlow(batchesPerStep=1,
                               anchorTensors={
                                   d2h_id: popart.AnchorReturnType("All")
                                   for d2h_id in d2h_ids
                               })
    ir_.setDataFlow(dataFlow)

    opts = ir_.getSessionOptions()
    opts.useHostCopyOps = True
    opts.enableExplicitMainLoops = True
    opts.aliasZeroCopy = True
    opts.explicitRecomputation = True

    ir_.removeIsolatedGraphs()

    # Runs the 'allreducetoidentitypattern' among others
    for g in ir_.getAllGraphs():
        ir_.applyPreAliasPatterns(g)

    ir_.updateVertices()

    with tu.create_test_device(numIpus=n_ipus) as device:
        session = popart.InferenceSession.fromIr(ir=ir_, deviceInfo=device)

        session.prepareDevice()

        anchors = session.initAnchorArrays()

        stepio = popart.PyStepIO(inputs=h2d_streams, outputs=anchors)

        session.weightsFromHost()
        session.run(stepio)

        y_host = [anchors[d2h_id] for d2h_id in d2h_ids]

    return y_host


def host_load(shape, dtype: popxl.dtype, name: str):
    x_h2d = popxl.h2d_stream(shape, dtype, name=f"{name}_stream")
    return x_h2d, ops.host_load(x_h2d, name)


@tu.requires_ipu
def test_all_reduce_op():
    n_ipus = 4
    ipus = list(range(n_ipus))

    inputs = np.arange(n_ipus * 2 * 3, dtype='float32').reshape((n_ipus, 2, 3))

    ir = popxl.Ir()
    main = ir.main_graph
    with main:

        x = []
        h2d_streams = {}
        for ipu in ipus:
            with popxl.ipu(ipu):
                x_h2d_i, x_i = host_load(inputs[ipu].shape,
                                         popxl.float32,
                                         name=f'x_{ipu}')
                h2d_streams[x_h2d_i.tensor_id] = inputs[ipu]
                x += [x_i]

        y = ops.collectives.all_reduce(x, ipus=ipus, op='add')

        y_d2h_ids = []
        for ipu in ipus:
            with popxl.ipu(ipu):
                y_i = y[ipu]
                y_d2h = popxl.d2h_stream(y_i.shape,
                                         y_i.dtype,
                                         name=f"y_{ipu}_stream")
                ops.host_store(y_d2h, y_i)
                y_d2h_ids += [y_d2h.tensor_id]

    y_host = run_ir(ir, h2d_streams, y_d2h_ids, n_ipus)

    # Outputs should be sum of inputs
    target = inputs.sum(axis=0)
    assert len(y_host) == n_ipus
    for i, y_host_i in enumerate(y_host):
        np.testing.assert_equal(target, y_host_i)


@tu.requires_ipu
def test_all_reduce_op_backwards():
    n_ipus = 4
    ipus = list(range(n_ipus))

    inputs = np.arange(n_ipus * 2 * 3, dtype='float32').reshape((n_ipus, 2, 3))

    ir = popxl.Ir()
    main = ir.main_graph
    with main:

        # Inputs for backwards
        x = []
        h2d_streams = {}
        for ipu in ipus:
            with popxl.ipu(ipu):
                x_h2d_i, x_i = host_load(inputs[ipu].shape,
                                         popxl.float32,
                                         name=f'x_{ipu}')
                h2d_streams[x_h2d_i.tensor_id] = inputs[ipu]
                x += [x_i]

        # Create graph
        def all_reduce_graph_func(*xs):
            return ops.collectives.all_reduce(xs, ipus=ipus, op='add')

        all_reduce_graph = ir.create_graph(all_reduce_graph_func, *x)

    # Auto diff
    all_reduce_graph_grad_info = popxl.transforms.autodiff(all_reduce_graph)
    all_reduce_graph_grad = all_reduce_graph_grad_info.graph

    with main:
        # call backwards
        y = ops.call(all_reduce_graph_grad, *x)

        # host store results
        y_d2h_ids = []
        for ipu in ipus:
            with popxl.ipu(ipu):
                y_i = y[ipu]
                y_d2h = popxl.d2h_stream(y_i.shape,
                                         y_i.dtype,
                                         name=f"y_{ipu}_stream")
                ops.host_store(y_d2h, y_i)
                y_d2h_ids += [y_d2h.tensor_id]

    y_host = run_ir(ir, h2d_streams, y_d2h_ids, n_ipus)

    # Outputs should be sum of inputs
    target = inputs.sum(axis=0)
    assert len(y_host) == n_ipus
    for i, y_host_i in enumerate(y_host):
        np.testing.assert_equal(target, y_host_i)


@tu.requires_ipu
def test_all_reduce_identical_inputs_op():
    n_ipus = 4
    ipus = list(range(n_ipus))

    inputs = np.arange(n_ipus * 2 * 3, dtype='float32').reshape((n_ipus, 2, 3))

    ir = popxl.Ir()
    main = ir.main_graph
    with main:

        x = []
        h2d_streams = {}
        for ipu in ipus:
            with popxl.ipu(ipu):
                x_h2d_i, x_i = host_load(inputs[ipu].shape,
                                         popxl.float32,
                                         name=f'x_{ipu}')
                h2d_streams[x_h2d_i.tensor_id] = inputs[ipu]
                x += [x_i]

        y = ops.collectives.all_reduce_identical_inputs(x, ipus=ipus, op='add')

        y_d2h_ids = []
        for ipu in ipus:
            with popxl.ipu(ipu):
                y_i = y[ipu]
                y_d2h = popxl.d2h_stream(y_i.shape,
                                         y_i.dtype,
                                         name=f"y_{ipu}_stream")
                ops.host_store(y_d2h, y_i)
                y_d2h_ids += [y_d2h.tensor_id]

    y_host = run_ir(ir, h2d_streams, y_d2h_ids, n_ipus)

    # Outputs should be identical to inputs
    assert len(y_host) == n_ipus
    for i, y_host_i in enumerate(y_host):
        np.testing.assert_equal(inputs[i], y_host_i)


@tu.requires_ipu
def test_all_reduce_identical_inputs_op_backwards():
    n_ipus = 4
    ipus = list(range(n_ipus))

    inputs = np.arange(n_ipus * 2 * 3, dtype='float32').reshape((n_ipus, 2, 3))

    ir = popxl.Ir()
    main = ir.main_graph
    with main:

        # Inputs for backwards
        x = []
        h2d_streams = {}
        for ipu in ipus:
            with popxl.ipu(ipu):
                x_h2d_i, x_i = host_load(inputs[ipu].shape,
                                         popxl.float32,
                                         name=f'x_{ipu}')
                h2d_streams[x_h2d_i.tensor_id] = inputs[ipu]
                x += [x_i]

        # Create graph
        def all_reduce_ifi_graph_func(*xs):
            return ops.collectives.all_reduce_identical_inputs(xs,
                                                               ipus=ipus,
                                                               op='add')

        all_reduce_ifi_graph = ir.create_graph(all_reduce_ifi_graph_func, *x)

    # Auto diff
    all_reduce_graph_grad_info = popxl.transforms.autodiff(
        all_reduce_ifi_graph)
    all_reduce_graph_grad = all_reduce_graph_grad_info.graph

    with main:
        # call backwards
        y = ops.call(all_reduce_graph_grad, *x)

        # host store results
        y_d2h_ids = []
        for ipu in ipus:
            with popxl.ipu(ipu):
                y_i = y[ipu]
                y_d2h = popxl.d2h_stream(y_i.shape,
                                         y_i.dtype,
                                         name=f"y_{ipu}_stream")
                ops.host_store(y_d2h, y_i)
                y_d2h_ids += [y_d2h.tensor_id]

    y_host = run_ir(ir, h2d_streams, y_d2h_ids, n_ipus)

    # Outputs should be sum of inputs
    target = inputs.sum(axis=0)
    assert len(y_host) == n_ipus
    for i, y_host_i in enumerate(y_host):
        np.testing.assert_equal(target, y_host_i)


@tu.requires_ipu
def test_all_reduce_identical_grad_inputs_op():
    n_ipus = 4
    ipus = list(range(n_ipus))

    inputs = np.arange(n_ipus * 2 * 3, dtype='float32').reshape((n_ipus, 2, 3))

    ir = popxl.Ir()
    main = ir.main_graph
    with main:

        x = []
        h2d_streams = {}
        for ipu in ipus:
            with popxl.ipu(ipu):
                x_h2d_i, x_i = host_load(inputs[ipu].shape,
                                         popxl.float32,
                                         name=f'x_{ipu}')
                h2d_streams[x_h2d_i.tensor_id] = inputs[ipu]
                x += [x_i]

        y = ops.collectives.all_reduce_identical_grad_inputs(x,
                                                             ipus=ipus,
                                                             op='add')

        y_d2h_ids = []
        for ipu in ipus:
            with popxl.ipu(ipu):
                y_i = y[ipu]
                y_d2h = popxl.d2h_stream(y_i.shape,
                                         y_i.dtype,
                                         name=f"y_{ipu}_stream")
                ops.host_store(y_d2h, y_i)
                y_d2h_ids += [y_d2h.tensor_id]

    y_host = run_ir(ir, h2d_streams, y_d2h_ids, n_ipus)

    # Outputs should be sum of inputs
    target = inputs.sum(axis=0)
    assert len(y_host) == n_ipus
    for i, y_host_i in enumerate(y_host):
        np.testing.assert_equal(target, y_host_i)


@tu.requires_ipu
def test_all_reduce_identical_grad_inputs_op_backwards():
    n_ipus = 4
    ipus = list(range(n_ipus))

    inputs = np.arange(n_ipus * 2 * 3, dtype='float32').reshape((n_ipus, 2, 3))

    ir = popxl.Ir()
    main = ir.main_graph
    with main:

        # Inputs for backwards
        x = []
        h2d_streams = {}
        for ipu in ipus:
            with popxl.ipu(ipu):
                x_h2d_i, x_i = host_load(inputs[ipu].shape,
                                         popxl.float32,
                                         name=f'x_{ipu}')
                h2d_streams[x_h2d_i.tensor_id] = inputs[ipu]
                x += [x_i]

        # Create graph
        def all_reduce_ibi_graph_func(*xs):
            return ops.collectives.all_reduce_identical_grad_inputs(xs,
                                                                    ipus=ipus,
                                                                    op='add')

        all_reduce_ibi_graph = ir.create_graph(all_reduce_ibi_graph_func, *x)

    # Auto diff
    all_reduce_graph_grad_info = popxl.transforms.autodiff(
        all_reduce_ibi_graph)
    all_reduce_graph_grad = all_reduce_graph_grad_info.graph

    with main:
        # call backwards
        y = ops.call(all_reduce_graph_grad, *x)

        # host store results
        y_d2h_ids = []
        for ipu in ipus:
            with popxl.ipu(ipu):
                y_i = y[ipu]
                y_d2h = popxl.d2h_stream(y_i.shape,
                                         y_i.dtype,
                                         name=f"y_{ipu}_stream")
                ops.host_store(y_d2h, y_i)
                y_d2h_ids += [y_d2h.tensor_id]

    y_host = run_ir(ir, h2d_streams, y_d2h_ids, n_ipus)

    # Outputs should be identical to inputs
    assert len(y_host) == n_ipus
    for i, y_host_i in enumerate(y_host):
        np.testing.assert_equal(inputs[i], y_host_i)
