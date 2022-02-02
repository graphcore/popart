# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
import popart
import popart.ir as pir
import popart.ir.ops as ops
import numpy as np

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


@pytest.mark.parametrize('tiedgather', [False, True])
@pytest.mark.parametrize('uint', [False, True])
def test_gather_out_of_range(tiedgather, uint):
    """Tests wether the output of gather/tiedgather produces zeros for out of range (OOR) indices."""
    n_samples = 10
    sample_max = (2**32) // 2 - 1
    weight_size = 20
    np.random.seed(1984)

    ir = pir.Ir()
    main = ir.main_graph()
    with main, pir.in_sequence():
        if uint:
            input_data = np.random.randint(weight_size + 1, sample_max,
                                           (n_samples, )).astype(np.uint32)
        else:
            input_data = np.concatenate([
                np.random.randint(-sample_max, -1, (n_samples // 2, )),
                np.random.randint(weight_size + 1, sample_max,
                                  (n_samples // 2, ))
            ]).astype(np.int32)
        weight_data = np.random.rand(4, weight_size).astype(np.float32)

        input_ = pir.variable(input_data)
        weight = pir.variable(weight_data)

        op = ops.tied_gather if tiedgather else ops.gather
        y_zero_false = op(weight, input_, zero_OOR=False)
        y_zero_true = op(weight, input_, zero_OOR=True)

        d2hs = []
        for t, name in ((y_zero_false, 'y_zero_false'), (y_zero_true,
                                                         'y_zero_true')):
            y_d2h = pir.d2h_stream(t.shape, t.dtype, name=f"{name}_stream")
            ops.host_store(y_d2h, t)
            d2hs += [y_d2h]

    ## Run the program
    ir = ir._pb_ir  # Internal ir

    dataFlow = popart.DataFlow(batchesPerStep=1,
                               anchorTensors={
                                   y_d2h.tensor_id():
                                   popart.AnchorReturnType("All")
                                   for y_d2h in d2hs
                               })
    ir.setDataFlow(dataFlow)

    ir.updateVertices()

    device = tu.create_test_device()
    session = popart.InferenceSession.fromIr(ir=ir, deviceInfo=device)

    session.prepareDevice()

    # Create buffers for anchors
    anchors = session.initAnchorArrays()

    # Run the model
    stepio = popart.PyStepIO(inputs={}, outputs=anchors)
    session.weightsFromHost()
    session.run(stepio)
    y_zero_false_np = anchors['y_zero_false_stream']
    y_zero_true_np = anchors['y_zero_true_stream']

    device.detach()

    ## Test
    assert (y_zero_true_np == 0).all()
