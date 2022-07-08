# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import argparse
import sys
from pathlib import Path
from typing import Union

import cppimport.import_hook  # pylint: disable=unused-import
import numpy as np
import popart

# In the source dir the leaky_relu_impl op lives in docs/shared/files
# We add this to the system path so this file can be ran from the source dir.
dir_shared = Path(__file__).parents[2].resolve()
dir_shared_tests = dir_shared.joinpath("shared", "files")
sys.path.append(str(dir_shared_tests))

import leaky_relu_op_impl


# Define a function to build and run the leaky relu graph with
# specified input tensor data and alpha value
def build_and_run_graph(
    input_data: Union[float, np.ndarray], alpha: float
) -> np.ndarray:
    """Build a PopART graph with the leaky relu custom op in and run it.

    Args:
        input_data (Union[float, np.ndarray]): The input data to use,
            either a 1D float or a NumPy array of floats.
        alpha (float): The alpha vale to use in the leaky relu op.

    Returns:
        np.ndarray: The output data array to be used for checking.
    """
    builder = popart.Builder()
    input_len = len(input_data)

    input_tensor = builder.addInputTensor(popart.TensorInfo("FLOAT", [input_len]))

    opid = leaky_relu_op_impl.LeakyRelu.default_opid()
    output_tensor = builder.customOp(
        opName=opid.type,
        opVersion=opid.version,
        domain=opid.domain,
        inputs=[input_tensor],
        attributes={"alpha": alpha},
    )[0]

    builder.addOutputTensor(output_tensor)

    proto = builder.getModelProto()

    anchors = {output_tensor: popart.AnchorReturnType("FINAL")}
    dataFlow = popart.DataFlow(1, anchors)

    device = popart.DeviceManager().createIpuModelDevice({})

    print(f"alpha={alpha}")

    session = popart.InferenceSession(proto, dataFlow, device)

    session.prepareDevice()
    result = session.initAnchorArrays()

    X = (np.array(input_data)).astype(np.float32)
    print(f"X={X}")

    stepio = popart.PyStepIO({input_tensor: X}, result)
    session.run(stepio, "LeakyReLU")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alpha", help="sets the lrelu alpha attribute", type=float, default=0.02
    )
    parser.add_argument(
        "--input_data", metavar="X", type=float, nargs="+", help="input tensor data"
    )

    args = parser.parse_args()

    result = build_and_run_graph(args.input_data, args.alpha)

    print("RESULT X")
    print(result)
