# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from typing import List, Tuple, Union
from contextlib import suppress
import pytest
import torch
import numpy as np

import popxl
import popxl.ops as ops


class TestTensor:
    @pytest.mark.parametrize(
        "input_shape, slices",
        [
            ((50, 24), [[2, 4, 1]]),
            ((50, 24), [[2, 4]]),
            ((50, 24), [[2, 6, 2]]),
            ((50, 24), [[2]]),
            ((50, 24), [[2, 4, 2], [0, 20, 4]]),
            ((50, 24), [[None, None, 2]]),
            ((50, 24), [[None, None, 2], [0, 20, 4]]),
        ],
    )
    def test_slicing(self, input_shape: Tuple[int, ...], slices: Union[int, List[int]]):

        slices = tuple(slice(*s) for s in slices)
        pt_input = torch.randint(0, 20, input_shape, dtype=torch.int32)
        pt_output = pt_input[slices]

        ir = popxl.Ir(replication=1)
        with ir.main_graph:
            dtype = popxl.dtype.as_dtype(pt_input.dtype)
            input_stream = popxl.h2d_stream(pt_input.shape, dtype=dtype, name="input")
            input_tensor = ops.host_load(input_stream)

            with popxl.in_sequence(True):
                o = input_tensor[slices]

                out_stream = popxl.d2h_stream(o.shape, o.dtype, "outputs")
                ops.host_store(out_stream, o)

        ir.num_host_transfers = 1

        with popxl.Session(ir, "ipu_model") as session:
            output = session.run({input_stream: pt_input})
            px_output = output[out_stream]

        assert np.allclose(pt_output.detach().numpy(), px_output)

    @pytest.mark.parametrize(
        "input_shape, expect_fail",
        [
            ((10, 10), False),
            ((10,), True),
            ((10, 10, 10), True),
        ],
    )
    def test_diag(self, input_shape, expect_fail):
        pt_input = torch.randint(0, 20, input_shape, dtype=torch.int32)

        ctx = pytest.raises(ValueError) if expect_fail else suppress()

        try:
            # This will fail for the 3d case, but that's ok.
            pt_output = pt_input.diag()
        except RuntimeError:
            pass

        ir = popxl.Ir(replication=1)
        with ctx, ir.main_graph:
            dtype = popxl.dtype.as_dtype(pt_input.dtype)
            input_stream = popxl.h2d_stream(pt_input.shape, dtype=dtype, name="input")
            input_tensor = ops.host_load(input_stream)

            with popxl.in_sequence(True):
                o = input_tensor.diag()

                out_stream = popxl.d2h_stream(o.shape, o.dtype, "outputs")
                ops.host_store(out_stream, o)

            ir.num_host_transfers = 1

            with popxl.Session(ir, "ipu_model") as session:
                output = session.run({input_stream: pt_input})
                px_output = output[out_stream]

            assert np.array_equal(pt_output.detach().numpy(), px_output)
