# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
'''
The intention of this example is to show how to use
call_with_info op.
'''

import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
import popart

# Creating a model with popart.ir
ir = pir.Ir()
main = ir.main_graph()


# Op begin
def increment_fn(x: pir.Tensor):
    value = pir.subgraph_input(x.shape, x.dtype, "value")
    # inplace increment of the input tensor
    ops.var_updates.copy_var_update_(x, x + value)


with main, pir.in_sequence():
    x = pir.variable(1)
    value1 = pir.constant(1)

    # create graph
    increment_graph = ir.create_graph(increment_fn, x)
    # call graph
    info = ops.call_with_info(increment_graph, x, value1)
    info.set_op_input_modified(x)
    # host store
    o_d2h = pir.d2h_stream(x.shape, x.dtype, name="output_stream")
    ops.host_store(o_d2h, x)
    # Op end

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

# run the model
stepio = popart.PyStepIO({}, anchors)
session.weightsFromHost()
session.run(stepio)

print(f"Result is {anchors[o_d2h.tensor_id()]}")