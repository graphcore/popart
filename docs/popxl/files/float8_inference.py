# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""
The intention of this example is to show a simple example of float 8 inference using PopXL.
"""

from typing import Tuple
import numpy as np
import popxl
import popxl.ops as ops
from popxl.utils import host_pow2scale_then_cast
import argparse

# ConvFloat8 begin
class ConvFloat8(popxl.Module):
    """
    Define a float 8 convolution layer in PopXL.
    """

    def __init__(self, opts_: argparse.Namespace) -> None:

        self.in_channel = opts_.in_channel
        self.out_channel = opts_.out_channel
        self.h_kernel = opts_.h_kernel
        self.w_kernel = opts_.w_kernel
        self.strides = opts_.strides
        self.group = opts_.group

        self.W: popxl.Tensor = None

    def build(self, x: popxl.Tensor, log2_scale: popxl.Tensor) -> popxl.Tensor:
        """
        Override the `build` method to build a graph.
        Note:
            x is a popxl.float_143 tensor, and log2_scale is an popxl.int32 tensor,
            in the range [-32,32)
        """
        self.W = popxl.graph_input(
            (
                self.out_channel,
                self.in_channel // self.group,
                self.h_kernel,
                self.w_kernel,
            ),
            popxl.float8_143,
            "W",
        )

        # Note this is a pow2scaled convolution that needs a log2_scale tensor.
        y = ops.conv_pow2scaled(x, self.W, log2_scale, stride=self.strides)

        y = ops.gelu(y)
        return y


# ConvFloat8 end


def get_input_data(opts_: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    """Generate some input data and initial weight data.

    Alternatively, this could be input data and pre-trained weights loaded from file.

    Args:
        opts_ (argparse.Namespace): Convolution options to use, see below.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The tuple of numpy data arrays.
    """
    t = np.random.random(
        (opts_.batch_size, opts_.in_channel, opts_.height, opts_.width)
    ).astype(np.float16)
    weight = np.random.random(
        (
            opts_.out_channel,
            opts_.in_channel // opts_.group,
            opts_.h_kernel,
            opts_.w_kernel,
        )
    ).astype(np.float16)
    return t, weight


def run_main(opts_: argparse.Namespace) -> None:
    """Run the example float 8 convolution graph."""

    t, weight = get_input_data(opts_)

    # Creating a model with popxl
    ir = popxl.Ir()
    main = ir.main_graph
    with main:

        # This is out log scale tensor, it can be a variable, graph input, or constant.
        log2_scale_t = popxl.variable(opts_.log2_scale, dtype=popxl.int32)
        # host load
        input0 = popxl.h2d_stream(
            [opts_.batch_size, opts_.in_channel, opts_.height, opts_.width],
            popxl.float16,
            name="in_stream_0",
        )
        a = ops.host_load(input0, "a")

        # Cast begin
        # Cast to fp8 on device before conv layer
        # Note we not not scale here, as scaling is done within the conv op.
        a_fp8 = ops.pow2scale_then_cast(
            a, data_type=popxl.float8_143, log2_scale=popxl.constant(0)
        )

        conv_ = ConvFloat8(opts_)
        # Convert the weight data on the host.
        # Note we not not scale here, as scaling is done within the conv op.
        weight_fp8 = host_pow2scale_then_cast(weight, popxl.float8_143, 0, False)

        W_t = popxl.variable(weight_fp8, popxl.float8_143)
        conv_graph_0 = ir.create_graph(conv_, a_fp8, log2_scale_t)

        # Cast end

        fwd_call_info_0 = ops.call_with_info(
            conv_graph_0, a_fp8, log2_scale_t, inputs_dict={conv_.W: W_t}
        )
        # This output will be of type popxl.float16
        o = fwd_call_info_0.outputs[0]

        o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
        ops.host_store(o_d2h, o)

    with popxl.Session(ir, opts_.device) as session:

        # SessionRun begin
        # run the model
        with session:
            outputs = session.run({input0: t})

        print(f"Input a is {t}")
        print(f"Initial weight is {weight}")
        print(f"Result is {outputs[o_d2h]}")
        # SessionRun end


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Float 8 inference in ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Set the Batch size.",
    )
    parser.add_argument(
        "--in_channel", type=int, default=10, help="Number of input channels"
    )
    parser.add_argument(
        "--out_channel", type=int, default=20, help="Number of output channels."
    )
    parser.add_argument("--height", type=int, default=5, help="Image height.")

    parser.add_argument("--width", type=int, default=5, help="Image width.")

    parser.add_argument("--h_kernel", type=int, default=3, help="Height of kernel.")

    parser.add_argument("--w_kernel", type=int, default=3, help="Width of kernel.")
    parser.add_argument(
        "--strides",
        nargs="+",
        type=int,
        default=[1, 1],
        help="Stride along each spatial axis.",
    )
    parser.add_argument(
        "--group",
        type=int,
        default=1,
        help="Number of groups.",
    )
    parser.add_argument(
        "--log2_scale",
        type=int,
        default=1,
        help="Log 2 scale to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="ipu_model",
        choices=["ipu_hw", "ipu_model"],
        help='Device to use. "ipu_hw" or "ipu_model"',
    )
    opts = parser.parse_args()

    run_main(opts)
