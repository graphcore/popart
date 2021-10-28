# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
import popart

# Creating a model with popart.ir
ir = pir.Ir()
main = ir.main_graph()
with main:
    # host load
    input0 = pir.h2d_stream([1], pir.float32, name="input0_stream")
    a = ops.host_load(input0, "a")
    input1 = pir.h2d_stream([1], pir.float32, name="input1_stream")
    b = ops.host_load(input1, "b")

    # addition
    o = ops.add(a, b)

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

# Generate some random input data
data_a = np.random.rand(1).astype(np.float32)
data_b = np.random.rand(1).astype(np.float32)
inputs = {input0.tensor_id(): data_a, input1.tensor_id(): data_b}

# run the model
stepio = popart.PyStepIO(inputs, anchors)
session.run(stepio)

print(f"Input a is {data_a}")
print(f"Input b is {data_b}")
print(f"Result is {anchors[o_d2h.tensor_id()]}")
