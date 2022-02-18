# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
'''
The intention of this example is to show how to use
popart.ir context manager in_sequence
'''

import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
import popart

# Creating a model with popart.ir
ir = pir.Ir()
main = ir.main_graph()
# Op begin
with main:
    x = pir.variable(1, pir.float32)
    b = pir.constant(5, pir.float32)

    with pir.in_sequence():
        ops.copy_var_update(x, b)
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
ir.setIsPrepared()

device = popart.DeviceManager().createCpuDevice()
session = popart.InferenceSession.fromIr(ir=ir, deviceInfo=device)
session.prepareDevice()
anchors = session.initAnchorArrays()

# run the model
stepio = popart.PyStepIO({}, anchors)
session.weightsFromHost()
session.run(stepio)

print(f"Result is {anchors[o_d2h.tensor_id()]}")
