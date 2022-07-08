# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Dict

import popart._internal.ir as _ir
from popxl import Graph
from popxl.context import get_current_context, op_debug_context
from typing_extensions import Literal

CODE_LOCATION = Literal["executable", "compute_buffer", "io_buffer"]
TO_IR_ENUM_MAP: Dict[CODE_LOCATION, _ir.CodeLocation] = {
    "io_buffer": _ir.CodeLocation.Buffer,
    "compute_buffer": _ir.CodeLocation.Buffer,
    "executable": _ir.CodeLocation.ExecutableMemory,
}


def _to_code_location_enum(type_: CODE_LOCATION) -> _ir.CodeLocation:
    """Convert a python string to a pybind _ir.CodeLocation.

    Args:
        type_ (CODE_LOCATION): The destination string.

    Raises:
        ValueError: If an invalid string is provided.

    Returns:
        _ir.CodeLocation: The _ir.CodeLocation to pass down to C++.
    """
    try:
        return TO_IR_ENUM_MAP[type_]
    except KeyError as key_error:
        raise ValueError(
            f"Not a valid location: {type_}. "
            f"Must choose from: {', '.join(TO_IR_ENUM_MAP.keys())}"
        ) from key_error


@op_debug_context
def remote_code_load(graph: Graph, destination: CODE_LOCATION) -> None:
    """Copy the provided graph's code to the destination location from remote memory.

    A graph's code refers to the poplar::Function that is a handle that represents both some code
    generated based on the poplar::program::Program that defines it and a Call()able object that
    represents some executable storage on the device.

    .. warning:: It is the user's responsibility to ensure that their subsequent calls
        to remote_code_load make sense. In particular, if the user never copies a graph to
        the 'executable' location and calls it, it will be assumed to have always been in
        executable memory, and hence always live.

    Args:
        graph (Graph): The Graph for which code will be copied.
        destination (CODE_LOCATION):
            The destination the specified graph's code will be copied to. One of:
            - "executable": Load code to executable tile memory on the chip.
            - "compute_buffer": Load code to compute tile buffer memory on the chip.
            - "io_buffer": Load code to reserved IO tile memory on the chip.
            Currently only "executable" is supported.

    Raises:
        ValueError: If the current context's graph and the provided graph are the same.
        NotImplementedError: If the destination is not yet implemented.
    """
    ctx = get_current_context()
    g = ctx.graph  # parent graph
    pb_g = g._pb_graph

    if g == graph:
        raise ValueError(
            f"The remote_code_load op cannot load the code for the graph it resides in ({g.id})."
        )
    settings = ctx._get_op_settings("remote_code_load")
    opid = _ir.OperatorIdentifier(
        "ai.graphcore", "RemoteCodeLoad", 1, _ir.NumInputs(0, 0), 0
    )
    if destination == "executable":
        _ = pb_g.createConnectedOp_RemoteCodeLoadOp(
            {},
            {},
            opid=opid,
            graphid=graph._pb_graph.id,  # Note: this is not the parent graph.
            destinationType=_to_code_location_enum(destination),
            settings=settings,
        )
    else:
        raise NotImplementedError(
            f"destination == '{destination}' for remote_code_load op, graph {graph.id} not supported."
        )


@op_debug_context
def code_copy(graph: Graph, source: CODE_LOCATION, destination: CODE_LOCATION) -> None:
    """Copy the provided graph's code internally, from `destination` to `source` on the chip.

    Args:
        graph (Graph): The Graph for which code will be copied.
        source (CODE_LOCATION):
            The source the specified graph's code will be copied from. One of:
            - "executable": Load code from executable tile memory on the chip.
            - "compute_buffer": Load code from compute tile buffer memory on the chip.
            - "io_buffer": Load code from reserved IO tile memory on the chip.
        destination (CODE_LOCATION):
            The destination the specified graph's code will be copied to. One of:
            - "executable": Load code to executable tile memory on the chip.
            - "compute_buffer": Load code to compute tile buffer memory on the chip.
            - "io_buffer": Load code to reserved IO tile memory on the chip.
            Currently only "executable" is supported.

    Raises:
        NotImplementedError: In any case. This op is not yet implemented.
    """
    raise NotImplementedError(f"{__name__} op is not yet implemented.")
