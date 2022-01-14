# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
'''
The intention of this example is to show how to create multiple
subgraphs from the same function.
'''

import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
import popart

# Creating a model with popart.ir
ir = pir.Ir()
main = ir.main_graph()


# Op begin
def matmul_fn(x: pir.Tensor, w: pir.Tensor):
    return x @ w


with main:
    # host load
    input = pir.h2d_stream([2, 2], pir.float32, name="input_stream")
    x = ops.host_load(input, "x")

    w1 = pir.variable(np.ones(x.shape, x.dtype.as_numpy()), name="w1")
    w2 = pir.variable(np.ones(x.shape[-1], x.dtype.as_numpy()), name="w2")

    # create two graphs
    matmul_graph1 = ir.create_graph(matmul_fn, x, w1)
    matmul_graph2 = ir.create_graph(matmul_fn, x, w2)

    # Op end

    # call graph
    o = ops.call(matmul_graph1, x, w1)
    o = ops.call(matmul_graph2, o, w2)

    # host store
    o_d2h = pir.d2h_stream(o.shape, o.dtype, name="output_stream")
    ops.host_store(o_d2h, o)

dataFlow = popart.DataFlow(
    batchesPerStep=1,
    anchorTensors={o_d2h.tensor_id(): popart.AnchorReturnType("All")})

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

# Generate some random input data
inputs = {input.tensor_id(): np.random.rand(2, 2).astype(np.float32)}

# run the model
stepio = popart.PyStepIO(inputs, anchors)
session.weightsFromHost()
session.run(stepio)

print(f"Input is {inputs[input.tensor_id()]}")
print(f"Result is {anchors[o_d2h.tensor_id()]}")
