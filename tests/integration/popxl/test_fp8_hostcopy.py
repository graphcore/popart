# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest

import popxl
import popxl.ops as ops
from popxl.fp8_utils import (
    host_pow2scale_cast_from_fp8,
    host_pow2scale_cast_to_fp8,
)
from popxl.dtypes import np_dtype_float8_143, np_dtype_float8_152

# fmt: off
@pytest.mark.parametrize("shape,log2_scale,format_,np_fp8_dtype", [
    [(2, 2), -1, popxl.float8_143, np_dtype_float8_143],
    [(1, 5,), 0, popxl.float8_143, np_dtype_float8_143],
    [(2, 9, 2, 1,), 1, popxl.float8_143, np_dtype_float8_143],
    [(16,), -1, popxl.float8_152, np_dtype_float8_152],
    [(1, 5,), 0, popxl.float8_152, np_dtype_float8_152],
    [(2, 9, 2, 1,), 1, popxl.float8_152, np_dtype_float8_152],
])
# fmt: on
def test_fp8_host_load_and_store(shape, log2_scale, format_, np_fp8_dtype):
    """Test whether the host_load op accepts a np.uint8 intended
    to be FP8, along with a float8 dtype.

    Likewise, test whether host_store and session.run supports
    returning np.uint8 that represents Float8 on the device.
    """
    ir = popxl.Ir()
    main_graph = ir.main_graph
    x32 = np.random.rand(*shape)

    x8_host = host_pow2scale_cast_to_fp8(x32, format_, log2_scale, False)
    with main_graph:
        instream0 = popxl.h2d_stream(x8_host.shape, format_, "instream0")

        x8 = ops.host_load(instream0, "x8")

        assert x8.dtype == format_

        outstream = popxl.d2h_stream(x8.shape, dtype=format_, name="out_stream")
        ops.host_store(outstream, x8)

    with popxl.Session(ir, "ipu_model") as session:
        outputs = session.run({instream0: x8_host})

    x8_result = outputs[outstream]
    assert x8_result.dtype == np_fp8_dtype
    np.testing.assert_array_equal(x8_result, x8_host)
    x32_result = host_pow2scale_cast_from_fp8(x8_result, popxl.float32, -log2_scale)

    # There is loss of accuracy whenever converting from FP8 due to round off
    # error so we tolerate some difference.
    np.testing.assert_allclose(x32_result.reshape(shape), x32, rtol=0.1, atol=1)
