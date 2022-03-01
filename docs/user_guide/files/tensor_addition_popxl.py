# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
'''
The intention of this example is to show the three
different types of tensors in popxl.
'''

import popxl
import popxl.ops as ops
import popart

# Creating a model with popxl
ir = popxl.Ir()
main = ir.main_graph
# Op begin
with main:
    a = popxl.variable(3, dtype=popxl.int8, name="variable_a")
    b = popxl.constant(1, dtype=popxl.int8, name="constant_b")

    # addition
    o = a + b
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

print(f"Result is {anchors[o_d2h.tensor_id]}")
