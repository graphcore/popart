# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Dict, List, Tuple
import popart._internal.ir as _ir
import popart
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


def add_actgrad_tensor(id_: str,
                       shape: List[int],
                       g: "_ir.Graph",
                       dtype: "_ir.DataType" = _ir.DataType.FLOAT):
    t_info = _ir.TensorInfo(dtype, shape)
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


def make_main_graph(num_inputs: int = 2
                    ) -> Tuple[_ir.Ir, List[_ir.Tensor], List[_ir.Tensor]]:
    """
    Creates the following graph, with num_inputs inputs,
    alternating data inputs and variable inputs.

    Init (act) Init (var)  Init (act) Init (var)
    |          |           |          |
    V          V           V          V
    Hostload   Hostload    Hostload   Hostload    etc..
    |          |           |          |
    |          |           |          |
    V          V           V          V
    .------------------------------------.
    |                Call                |
    |                                    |
    '------------------+-----------------'
                       |
                       |
                       V
                    HostStore
    Args:
        num_inputs (int, optional): Number of main graph inputs. Defaults to 2.

    Returns:
        Tuple[_ir.Ir, List[_ir.Tensor], List[_ir.Tensor]]:
            The ir, The subgraph output tensors, and the variable tensors
    """
    ir, _ = create_ir()

    main = ir.getMainGraph()

    t_info = _ir.TensorInfo(_ir.DataType.FLOAT, [1, 2, 3])

    inits: Dict[int, _ir.Op] = dict()
    hostloads: Dict[int, _ir.Op] = dict()
    inputs = {i: t_info for i in range(num_inputs)}

    for i in range(len(inputs)):
        # init i
        opid = _ir.OperatorIdentifier("ai.onnx", f"Init{i}", 1,
                                      _ir.NumInputs(0, 0), 1)
        actgrads = []
        vars_ = []
        settings = _ir.Settings(main, f"input{i}")
        if i % 2 == 0:
            ttype = _ir.TensorType.ActGrad
            inits[i] = main.createConnectedOp_InitOp(
                {}, {0: f"init{i}"}, opid, t_info, ttype, _ir.InitType.Zero,
                settings, 0)
            actgrads.append(inits[i])
        else:
            ttype = _ir.TensorType.Variable
            inits[i] = main.createConnectedOp_InitOp(
                {}, {0: f"init{i}"}, opid, t_info, ttype, _ir.InitType.Zero,
                settings, 0)
            vars_.append(inits[i].outTensor(0))

        # hostload i
        opid = _ir.OperatorIdentifier("ai.onnx", f"HostLoad{i}", 1,
                                      _ir.NumInputs(1, 1), 1)
        hostloads[i] = main.createConnectedOp_HostLoadOp(
            {0: inits[i].outTensor(0).id}, {0: f"hostload{i}"}, opid, settings,
            f"hl{i}")

    settings = _ir.Settings(main, "call")
    fwd = make_sub_graph(ir, inputs)

    fwd_outs = [fwd.getTensor(tid) for tid in fwd.getOutputIds()]

    opid = _ir.OperatorIdentifier("ai.graphcore", "Call", 1,
                                  _ir.NumInputs(num_inputs, num_inputs), 1)
    call = main.createConnectedOp_CallOp(
        {i: hostloads[i].outTensor(0).id
         for i in range(len(hostloads))}, {0: "call0"}, opid, fwd, settings)

    # host store
    opid = _ir.OperatorIdentifier("ai.onnx", "HostStore", 1, _ir.NumInputs(
        1, 1), 1)
    settings = _ir.Settings(main, "host_store")

    _ = main.createConnectedOp_HostStoreOp({0: call.outTensor(0).id}, {}, opid,
                                           settings, "hs1")

    deviceInfo = popart.DeviceManager().createIpuModelDevice({})
    ir.setDeviceInfo(deviceInfo)

    ir.setIsPrepared()
    ir.logIr()

    print(fwd_outs, vars_)

    return ir, fwd_outs, vars_


def make_sub_graph(ir: _ir.Ir, ins: Dict[int, _ir.TensorInfo]) -> _ir.Graph:
    """
    Makes the following subgraph, with len(ins) inputs.

    input0  input1  input2  ...  input n
    |       |       |            |
    |       |       |            |
    |       |       |            |
    '->add <'       |            |
        |           |            |
        '------>add<'            |
                |                |
                |                |
                |                |
                '---->add ...    V


                               add
                                |
                                V
                             softmax
                                |
                                V
                               out

    Args:
        ir (_ir.Ir): The ir to add the subgraph to
        ins (Dict[int, _ir.TensorInfo]): The map of in indices to tensorinfos.

    Returns:
        _ir.Graph: The subgraph in question.
    """
    g = ir.createGraph(_ir.GraphId("fwd"))

    for i, tinfo in ins.items():
        g.addInput(_ir.addScope(g, f"in{i}"), tinfo)

    inputs = g.getInputIds()

    t = g.getTensor(inputs[0])
    for i in range(1, len(ins)):
        settings = _ir.Settings(g, f"add{i}")
        opid = _ir.OperatorIdentifier("ai.onnx", f"Add{i}", 1,
                                      _ir.NumInputs(2, 2), 1)
        add = g.createConnectedOp_AddOp({
            0: t.id,
            1: inputs[i]
        }, {0: _ir.addScope(g, f"add{i}")}, opid, settings)
        t = add.outTensor(0)

    settings = _ir.Settings(g, "softmax0")
    opid = _ir.OperatorIdentifier("ai.onnx", "SoftMax", 1, _ir.NumInputs(1, 1),
                                  1)
    sm = g.createConnectedOp_SoftmaxOp({0: t.id}, {0: _ir.addScope(g, "sm0")},
                                       opid=opid,
                                       axis_=0,
                                       settings=settings)

    g.markAsOutput(sm.outTensor(0).id)

    return g
