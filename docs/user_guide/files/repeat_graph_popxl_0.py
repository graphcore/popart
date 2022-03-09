# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
'''
The intention of this example is to show how to call the same
subgraph from a Python function with a repeat operation.
'''
import numpy as np
import popxl
import popxl.ops as ops
import popart

# Creating a model with popxl
ir = popxl.Ir()
main = ir.main_graph


# Op begin
def increment_fn(x: popxl.Tensor, value: popxl.Tensor):
    return x + value


with main:
    # host load
    x = popxl.variable(np.ones([2, 2], np.float32), name="x")
    value = popxl.variable(np.ones(x.shape, x.dtype.as_numpy()), name="value")

    # create graph
    increment_graph = ir.create_graph(increment_fn, x, value)

    # call graph in a loop
    o, = ops.repeat(increment_graph, 2, x, value)
    # Op end
    # host store
    o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="output_stream")
    ops.host_store(o_d2h, o)

dataFlow = popart.DataFlow(
    batchesPerStep=1,
    anchorTensors={o_d2h.tensor_id: popart.AnchorReturnType("All")})

ir = ir._pb_ir
ir.setDataFlow(dataFlow)
opts = ir.getSessionOptions()
opts.useHostCopyOps = True
opts.enableExplicitMainLoops = True
ir.updateVertices()

device = popart.DeviceManager().createCpuDevice()
session = popart.InferenceSession.fromIr(ir=ir, deviceInfo=device)
session.prepareDevice()
anchors = session.initAnchorArrays()

# run the model
stepio = popart.PyStepIO({}, anchors)
session.weightsFromHost()
session.run(stepio)

assert (anchors[o_d2h.tensor_id] == [[3, 3], [3, 3]]).all()
