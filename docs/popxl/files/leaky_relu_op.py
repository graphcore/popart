# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import sys
from pathlib import Path

# In the source dir the leaky_relu_impl op lives in docs/shared/files
# We add this to the system path so this file can be ran from the source dir.
dir_shared = Path(__file__).parents[2].resolve()
dir_shared_tests = dir_shared.joinpath("shared", "files")
sys.path.append(str(dir_shared_tests))

# leaky_relu begin

from typing import Optional

import cppimport.import_hook  # pylint: disable=unused-import
from popxl.context import get_current_context, op_debug_context
from popxl.ops.utils import check_in_graph
from popxl.tensor import Tensor

# The custom op and its pybinding will be automatically compiled by cppimport
# into a module of this name.
import leaky_relu_op_impl


@op_debug_context
def leaky_relu(x: Tensor, alpha: Optional[float] = 0.01, **kwargs) -> Tensor:  # pylint: disable=unused-argument
    """Compute the leaky relu operator element-wise on the input tensor.

    See https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html

    Args:
        x (Tensor): The tensor to input.
        alpha (float, optional): The value to use to determine
            the slope of the negative axis. Defaults to 0.01.
        **kwargs: Any additional arguments to passed. Left here as an example for
            other operators.

    Returns:
        Tensor: The output tensor.
    """
    ctx = get_current_context()
    graph = ctx.graph
    pb_graph = graph._pb_graph

    settings = ctx._get_op_settings("leaky_relu")
    params = leaky_relu_op_impl.LeakyReluParams(alpha=alpha)
    # Inputs & outputs mapping (checking inputs are in graph!).
    inputs = {0: x.id}
    outputs = {0: graph._create_tensor_id("leaky_relu_out")}
    check_in_graph(graph, **{x.id: x})

    # Building the op using default operator id
    op = leaky_relu_op_impl.LeakyRelu.create_op_in_graph(
        graph=pb_graph,
        inputs=inputs,
        outputs=outputs,
        params=params,
        settings=settings,
    )
    # Applying context all registered hooks to the new op.
    # NOTE: crucial to support PopXL graph transforms.
    ctx._op_created(op)
    return Tensor._from_pb_tensor(op.outTensor(0))
