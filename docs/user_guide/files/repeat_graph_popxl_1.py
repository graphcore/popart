# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
'''
The intention of this example is to show how to call the same
subgraph with repeat op and connect the inputs of the subgraph
with the caller graph by using argument `inputs_dict`.
'''

import numpy as np
import popxl
import popxl.ops as ops
import popart
from typing import Tuple

# Creating a model with popxl
ir = popxl.Ir()
main = ir.main_graph


# Op begin
class Linear(popxl.Module):
    def __init__(self):
        self.W: popxl.Tensor = None
        self.b: popxl.Tensor = None

    def build(self, x: popxl.Tensor, out_features: int,
              bias: bool = True) -> Tuple[popxl.Tensor, ...]:
        self.W = popxl.graph_input((x.shape[-1], out_features), popxl.float32,
                                   "W")
        y = x @ self.W
        if bias:
            self.b = popxl.graph_input((out_features, ), popxl.float32, "b")
            y = y + self.b
        return y


with main:
    # host load
    x = popxl.variable(np.ones([2, 2], np.float32), name="x")
    W = popxl.variable(np.ones([2, 2], np.float32), name="W")
    b = popxl.variable(np.ones([2], np.float32), name="b")

    # create graph
    linear = Linear()
    linear_graph = ir.create_graph(linear, x, out_features=2)

    # call graph in a loop
    # the x, W, b will be copied to the input of the `linear_graph` before the first iteration
    # the outputs of each iteration will be copied to the inputs of the next iteration
    # The outputs of the last iteration serve as the output of the `repeat` op
    o, = ops.repeat(linear_graph, 2, x, inputs_dict={linear.W: W, linear.b: b})
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

assert (anchors[o_d2h.tensor_id] == [[7, 7], [7, 7]]).all()
