# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart
import numpy as np
import test_util as tu


# Before D59260, this test would fail with the error:
#   E popart_core.popart_exception: Regions are of different rank (1 vs. 2) in sub
# This was due to a bug in the unwinding code for MaxOpx.
# All this test needs to do to pass is not throw an error.
def test_min_max_unwinding():
    builder = popart.Builder()

    input_tensor = builder.addInputTensor(popart.TensorInfo("INT32", [2, 4]))

    max_val_tensor = builder.aiOnnx.constant(np.array([19]).astype(np.int32))
    min_val_tensor = builder.aiOnnx.constant(np.array([0]).astype(np.int32))
    max_out = builder.aiOnnx.max([input_tensor, max_val_tensor])
    min_out = builder.aiOnnx.min([max_out, min_val_tensor])
    t0 = builder.aiOnnx.constant(np.array([1]).astype(np.int32))
    gather_out = builder.aiOnnx.gather([min_out, t0])

    o = gather_out

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    sess = popart.InferenceSession(builder.getModelProto(), dataFlow,
                                   tu.create_test_device())
    anchors = sess.initAnchorArrays()
    sess.prepareDevice()
