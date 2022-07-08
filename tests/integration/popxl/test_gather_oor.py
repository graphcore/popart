# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
import popxl
import popxl.ops as ops
import numpy as np


@pytest.mark.parametrize("tiedgather", [False, True])
@pytest.mark.parametrize("uint", [False, True])
def test_gather_out_of_range(tiedgather, uint):
    """Tests whether the output of gather/tiedgather produces zeros for out of range (OOR) indices."""
    n_samples = 10
    sample_max = (2 ** 32) // 2 - 1
    weight_size = 20
    np.random.seed(1984)

    ir = popxl.Ir()
    main = ir.main_graph
    with main, popxl.in_sequence():
        if uint:
            input_data = np.random.randint(
                weight_size + 1, sample_max, (n_samples,)
            ).astype(np.uint32)
        else:
            input_data = np.concatenate(
                [
                    np.random.randint(-sample_max, -1, (n_samples // 2,)),
                    np.random.randint(weight_size + 1, sample_max, (n_samples // 2,)),
                ]
            ).astype(np.int32)
        weight_data = np.random.rand(4, weight_size).astype(np.float32)

        input_ = popxl.variable(input_data)
        weight = popxl.variable(weight_data)

        op = ops.tied_gather if tiedgather else ops.gather
        y_zero_false = op(weight, input_, zero_OOR=False)
        y_zero_true = op(weight, input_, zero_OOR=True)

        d2hs = []
        for t, name in ((y_zero_false, "y_zero_false"), (y_zero_true, "y_zero_true")):
            y_d2h = popxl.d2h_stream(t.shape, t.dtype, name=f"{name}_stream")
            ops.host_store(y_d2h, t)
            d2hs += [y_d2h]

    session = popxl.Session(ir, "ipu_model")

    with session:
        outputs = session.run()
    _ = outputs[d2hs[0]]  # y_zero_false_np
    y_zero_true_np = outputs[d2hs[1]]

    ## Test
    assert (y_zero_true_np == 0).all()
