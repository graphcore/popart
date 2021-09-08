# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Dict, List
import popart._internal.ir as _ir
import numpy as np


def create_ir(graph_ids: List[str] = []):
    """Small helper function to create an Ir with some graphs

    Args:
        graph_ids (List[str]): List of graph Ids to create

    Returns:
        Ir: An Ir with the required graphs.
    """
    ir = _ir.Ir()
    graphs = [ir.getMainGraph()]
    for name in graph_ids:
        id = _ir.GraphId(name)
        g = _ir.Graph(ir, id)
        graphs.append(g)

    return ir, graphs


def create_dummy_op(op_domain: str, op_type: str, op_version: int,
                    num_inputs: int, num_outputs: int) -> _ir.Op:
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


def create_new_op(inputs: Dict[int, "_ir.Tensor"],
                  outputs: Dict[int, "_ir.Tensor"],
                  op_name: str,
                  g: "_ir.Graph",
                  inplace: bool = False,
                  connected: bool = False,
                  **kwargs):
    num_inputs = _ir.NumInputs(len(inputs), len(inputs))
    opid = _ir.OperatorIdentifier("ai.onnx", op_name, 1, num_inputs,
                                  len(outputs))
    settings = _ir.Settings(g, "new_settings")
    if connected:
        create_new_op_fn = getattr(g, f"createConnectedOp_{op_name}")
        inp: Dict[int, str] = {}
        outp: Dict[int, str] = {}
        for i, t in inputs.items():
            inp[i] = t.id
        for i, t in outputs.items():
            outp[i] = t.id
        # For some reason inplace ops don't need opid
        if inplace:
            op = create_new_op_fn(inp, outp, settings=settings, **kwargs)
        else:
            op = create_new_op_fn(inp,
                                  outp,
                                  opid=opid,
                                  settings=settings,
                                  **kwargs)
    else:
        create_new_op_fn = getattr(g, f"createOp_{op_name}")
        if inplace:
            op = create_new_op_fn(settings=settings, **kwargs)
        else:
            op = create_new_op_fn(opid=opid, settings=settings, **kwargs)
        for i, t in inputs.items():
            op.connectInTensor(i, t.id)
        for i, t in outputs.items():
            op.connectOutTensor(i, t.id)
    op.setup()
    return op


def add_actgrad_tensor(id_: str, shape: List[int], g: "_ir.Graph"):
    t_info = _ir.TensorInfo(_ir.DataType.FLOAT, shape)
    g.addActGrad(id_)
    t = g.getTensor(id_)
    t.info = t_info
    return t


def add_zeros_tensor(id_: str, t_type: "_ir.TensorType", shape: List[int],
                     g: "_ir.Graph") -> "_ir.Tensor":
    data = np.zeros(shape, dtype=np.float32)
    t_info = _ir.TensorInfo(_ir.DataType.FLOAT, shape)
    if t_type == _ir.TensorType.ActGrad:
        raise TypeError("Use add_actgrad_tensor for adding actgrad tensors. "
                        "Actgrad tensors have no initial value.")
    elif t_type == _ir.TensorType.Variable:
        g.addVarInit(id_, t_info, data, "")
        return g.getTensor(id_)
    elif t_type == _ir.TensorType.Const:
        g.addConstInit(id_, t_info, data, "")
        return g.getTensor(id_)

    else:
        raise TypeError("Incorrect tensor type")


def add_random_tensor(id_: str, t_type: "_ir.TensorType", shape: List[int],
                      g: "_ir.Graph") -> "_ir.Tensor":
    data = np.random.random(shape).astype(np.float32)
    t_info = _ir.TensorInfo(_ir.DataType.FLOAT, shape)
    if t_type == _ir.TensorType.ActGrad:
        raise TypeError("Use add_actgrad_tensor for adding actgrad tensors. "
                        "Actgrad tensors have no initial value.")
    elif t_type == _ir.TensorType.Variable:
        g.addVarInit(id_, t_info, data, "")
        return g.getTensor(id_)
    elif t_type == _ir.TensorType.Const:
        g.addConstInit(id_, t_info, data, "")
        return g.getTensor(id_)

    else:
        raise TypeError("Incorrect tensor type")
