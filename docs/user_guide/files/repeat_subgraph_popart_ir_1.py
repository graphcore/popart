# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
'''
The intention of this example is to show how to call the same
subgraph with repeat op and connect the inputs of the subgraph
with the caller graph by using argument `subgraph_in_to_parent_in`.
'''

import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
import popart
from typing import Tuple

# Creating a model with popart.ir
ir = pir.Ir()
main = ir.main_graph()


# Op begin
class Linear(pir.Module):
    def __init__(self):
        self.W: pir.Tensor = None
        self.b: pir.Tensor = None

    def build(self, x: pir.Tensor, out_features: int,
              bias: bool = True) -> Tuple[pir.Tensor, ...]:
        self.W = pir.subgraph_input((x.shape[-1], out_features), pir.float32,
                                    "W")
        y = x @ self.W
        if bias:
            self.b = pir.subgraph_input((out_features, ), pir.float32, "b")
            y = y + self.b
        # self.W and self.b are also returned to match the inputs of the corresponding subgraph
        return y, self.W, self.b


with main:
    # host load
    x = pir.variable(np.ones([2, 2], np.float32), name="x")
    W = pir.variable(np.ones([2, 2], np.float32), name="W")
    b = pir.variable(np.ones([2], np.float32), name="b")

    # create graph
    linear = Linear()
    linear_graph = ir.create_graph(linear, x, out_features=2)

    # call graph in a loop
    # the x, W, b will be copied to the input of the `linear_graph` before the first iteration
    # the outputs of each iteration will be copied to the inputs of the next iteration
    # The outputs of the last iteration serve as the output of the `repeat` op
    o, _, _ = ops.repeat(linear_graph,
                         2,
                         x,
                         subgraph_in_to_parent_in={
                             linear.W: W,
                             linear.b: b
                         })
    # Op end
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
ir.setIsPrepared()

device = popart.DeviceManager().createCpuDevice()
session = popart.InferenceSession.fromIr(ir=ir, deviceInfo=device)
session.prepareDevice()
anchors = session.initAnchorArrays()

# run the model
stepio = popart.PyStepIO({}, anchors)
session.weightsFromHost()
session.run(stepio)

assert (anchors[o_d2h.tensor_id()] == [[7, 7], [7, 7]]).all()
