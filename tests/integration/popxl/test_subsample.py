# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import List, Tuple, Union
from contextlib import suppress
import pytest
import torch
import numpy as np

import popxl
import popxl.ops as ops


class TestSubsample:
    @pytest.mark.parametrize(
        "input_shape, slices, expect_fail",
        [
            ((50, 24), [2, 4], False),
            ((50, 24), [2], False),
            ((10,), 2, False),
            ((10,), -2, True),
        ],
    )
    def test_fn(
        self,
        input_shape: Tuple[int, ...],
        slices: Union[int, List[int]],
        expect_fail: bool,
    ):

        ctx = pytest.raises(Exception) if expect_fail else suppress()

        pt_input = torch.randint(0, 20, input_shape, dtype=torch.int32)

        with ctx:
            if isinstance(slices, int):
                pt_output = pt_input[::slices]
            elif len(slices) == 1:
                pt_output = pt_input[:: slices[0]]
            else:
                pt_output = pt_input[:: slices[0], :: slices[1]]

        ir = popxl.Ir(replication=1)
        with ctx:
            with ir.main_graph:
                dtype = popxl.dtype.as_dtype(pt_input.dtype)
                input_stream = popxl.h2d_stream(
                    pt_input.shape, dtype=dtype, name="input"
                )
                input_tensor = ops.host_load(input_stream)

                with popxl.in_sequence(True):
                    o = ops.subsample(input_tensor, slices)

                    out_stream = popxl.d2h_stream(o.shape, o.dtype, "outputs")
                    ops.host_store(out_stream, o)

            ir.num_host_transfers = 1

            with popxl.Session(ir, "ipu_model") as session:
                output = session.run({input_stream: pt_input})
                px_output = output[out_stream]

            assert np.allclose(pt_output.detach().numpy(), px_output)
