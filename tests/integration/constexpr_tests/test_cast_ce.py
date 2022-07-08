# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np

# importing test_session requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from test_session import PopartTestSession
import test_util as tu


def test_various_casts():
    np.random.seed(1)

    def run_test(input_data, const_data):
        cast_to = {
            np.dtype(np.int64): "INT64",
            np.dtype(np.int32): "INT32",
            np.dtype(np.float32): "FLOAT",
            np.dtype(np.float16): "FLOAT16",
        }[input_data.dtype]

        cast_id = ""
        out_id = ""

        def init_builder(builder):
            nonlocal out_id
            nonlocal cast_id
            d0 = builder.addInputTensor(input_data, "data0")

            t0 = builder.aiOnnx.constant(const_data)
            t1 = builder.aiOnnx.cast([t0], cast_to)
            t2 = builder.aiOnnx.identity([t1])
            t3 = builder.aiOnnx.mul([d0, t2])

            builder.addOutputTensor(t3)
            out_id = t3
            cast_id = t2

            _ = builder.setLoss(t3)
            return [t3, t2]

        session = PopartTestSession()

        # test a pipeline stage appearing on multiple virtual graphs
        with tu.create_test_device() as device:
            session.prepare(init_builder, device=device)
            anchors = session.run()
        # print(anchors)
        cast_output = anchors[cast_id]
        numpy_cast = const_data.astype(input_data.dtype)
        print(f"input data: {input_data}")
        print(f"const data: {const_data}")
        print(f"cast output: {cast_output}")
        print(f"numpy cast: {numpy_cast}")

        if const_data.dtype == np.float32 and input_data.dtype == np.float16:
            assert np.allclose(cast_output, numpy_cast, rtol=1e-03)
        else:
            assert np.allclose(cast_output, numpy_cast)

    def get_data(dtype):
        x = np.random.rand(2) * 100
        return x.astype(dtype)

    def test_cast(from_dtype, to_dtype):
        print(
            f"\n\nTesting cast from {np.dtype(from_dtype).name} to {np.dtype(to_dtype).name}"
        )
        input_data = get_data(to_dtype)
        const_data = get_data(from_dtype)
        run_test(input_data, const_data)

    for from_type in (np.int64, np.int32, np.float32, np.float16):
        for to_type in (np.int32, np.float32, np.float16):
            if from_type != to_type:
                test_cast(from_type, to_type)
