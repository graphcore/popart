# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Union
import numpy as np
import argparse

import popxl
import popxl.ops as ops

from leaky_relu_op_popxl import leaky_relu


def build_and_run_graph(input_data: Union[float, np.ndarray],
                        alpha: float) -> np.ndarray:
    """Build a PopXL graph with the leaky relu custom op in and run it.

    Args:
        input_data (Union[float, np.ndarray]): The input data to use,
            either a 1D float or a NumPy array of floats.
        alpha (float): The alpha vale to use in the leaky relu op.

    Returns:
        np.ndarray: The output data array to be used for checking.
    """
    # Creating a model with popxl
    ir = popxl.Ir()
    main = ir.main_graph
    input_array = np.array(input_data)
    with main:
        # host load
        input0 = popxl.h2d_stream(input_array.shape,
                                  popxl.float32,
                                  name="in_stream")
        a = ops.host_load(input0, "a")

        # custom leaky relu.
        o = leaky_relu(a, alpha=alpha)

        # host store
        o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
        ops.host_store(o_d2h, o)

    session = popxl.Session(ir, "ipu_model")
    outputs = session.run({input0: input_array})

    print("ALPHA param:", alpha)
    print("INPUT data:", input_data)
    print("OUTPUT result:", outputs[o_d2h])

    return outputs[o_d2h]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha",
                        help="sets the lrelu alpha attribute",
                        type=float,
                        default=0.02)
    parser.add_argument("--input_data",
                        metavar="X",
                        type=float,
                        nargs="+",
                        help="input tensor data",
                        default=0.01)

    args = parser.parse_args()

    result = build_and_run_graph(args.input_data, args.alpha)

    print("RESULT X:", result)
