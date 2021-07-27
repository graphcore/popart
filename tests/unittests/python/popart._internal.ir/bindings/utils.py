# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import List
import popart._internal.ir as _ir
import numpy as np


def create_ir(graph_ids: List[str]):
    """Small helper function to create an Ir with some graphs

    Args:
        graph_ids (List[str]): List of graph Ids to create

    Returns:
        Ir: An Ir with the required graphs.
    """
    ir = _ir.Ir()
    graphs = []
    for name in graph_ids:
        id = _ir.GraphId(name)
        g = _ir.Graph(ir, id)
        graphs.append(g)
    return ir, graphs


def create_op(op_domain: str, op_type: str, op_version: int, num_inputs: int,
              num_outputs: int) -> _ir.Op:
    """Create an op with the provided properties.

    Args:
        op_domain (str): Op domain
        op_type (str): Op type name
        op_version (int): Op version
        num_inputs (int): Max = min number of outputs
        num_outputs (int): Number of outputs

    Returns:
        _ir.Op: The op in question.
    """
    ir, graphs = create_ir(["graph_123"])
    graph = graphs[0]
    settings = _ir.Settings(graph, "new_settings")
    num_inputs_obj = _ir.NumInputs(num_inputs, num_inputs)
    opid = _ir.OperatorIdentifier(op_domain, op_type, op_version,
                                  num_inputs_obj, num_outputs)
    return _ir.Op(opid, settings), ir, graph


def add_tensor(graph: _ir.Graph,
               tid: str = "t",
               shape: List[int] = [1],
               t_type: _ir.TensorType = _ir.TensorType.ActGrad,
               d_type: _ir.DataType = _ir.DataType.FLOAT) -> _ir.Tensor:
    """Add a tensor to the given graph, with given properties.

    Args:
        graph (_ir.Graph): Graph to use
        tid (str, optional): Id for the tensor. Defaults to "t".
        shape (List[int], optional): Shape for the tensor. Defaults to [1].
        t_type (_ir.TensorType, optional): Tensor type. Defaults to _ir.TensorType.ActGrad.
        d_type (_ir.DataType, optional): DataType for the tensor. Defaults to _ir.DataType.FLOAT.

    Returns:
        _ir.Tensor: the tensor that was just added.
    """
    ts = _ir.Tensors(graph)
    tinfo = _ir.TensorInfo(d_type, shape)
    if t_type == _ir.TensorType.ActGrad:
        ts.addActGrad(tid)
    elif t_type == _ir.TensorType.Variable:
        data = np.random.rand(*shape).astype(np.float32)
        ts.addVarInit(tid, tinfo, data)
    elif t_type == _ir.TensorType.Const:
        data = np.random.rand(*shape).astype(np.float32)
        ts.addConstInit(tid, tinfo, data)
    return ts.get(tid)
