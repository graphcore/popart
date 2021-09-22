# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Any, Dict, List, Tuple
import pytest
import popart._internal.ir as _ir
import numpy as np
import popart
from utils import *


def unary_op_tester(op_name: str,
                    g: _ir.Graph,
                    inplace: bool = False,
                    connected: bool = False,
                    **kwargs):
    """Helper to test unary ops

    Args:
        op_name (str): Name of op to create. Must match the create<op_name> function.
        g (_ir.Graph): The graph to add the op to.
        inplace (bool, optional): Whether to use the inplace variant. Defaults to False.
        connected (bool, optional): Whether to use the createConnected<opname> function or 
            just create<opname>. Defaults to False.
    """
    in0 = add_actgrad_tensor("in0", [1, 2, 3], g)
    out0 = add_actgrad_tensor("out0", [1, 2, 3], g)
    ins = {0: in0}
    outs = {0: out0}
    op = create_new_op(ins, outs, op_name, g, inplace, connected, **kwargs)
    for i, t in ins.items():
        assert op.inTensor(i) == t
        assert op.hasInput(i)
        assert op.inId(i) == t.id
    for i, t in outs.items():
        assert op.outTensor(i) == t
        assert op.hasOutput(i)
        assert op.outId(i) == t.id
    assert op.getCalledGraphs() == []
    assert op.getCalledGraphIds() == []


def binary_op_tester(op_name: str,
                     g: _ir.Graph,
                     inplace: bool = False,
                     connected: bool = False,
                     *args):
    """Helper to test binary ops

    Args:
        op_name (str): Name of op to create. Must match the create<op_name> function.
        g (_ir.Graph): The graph to add the op to.
        inplace (bool, optional): Whether to use the inplace variant. Defaults to False.
        connected (bool, optional): [Whether to use the createConnected<opname> function or 
            just create<opname>. Defaults to False.
    """
    in0 = add_actgrad_tensor("in0", [1, 2, 3], g)
    in1 = add_random_tensor("in1", _ir.TensorType.Variable, [1, 2, 3], g)
    out0 = add_actgrad_tensor("out0", [1, 2, 3], g)
    ins = {0: in0, 1: in1}
    outs = {0: out0}
    op = create_new_op(ins, outs, op_name, g, inplace, connected, *args)
    for i, t in ins.items():
        assert op.inTensor(i) == t
        assert op.hasInput(i)
        assert op.inId(i) == t.id
    for i, t in outs.items():
        assert op.outTensor(i) == t
        assert op.hasOutput(i)
        assert op.outId(i) == t.id
    assert op.getCalledGraphs() == []
    assert op.getCalledGraphIds() == []


# yapf mangles these param lists
# yapf: disable, pylint: disable-all
@pytest.mark.parametrize("op_name,inplace",
[("AddOp", False),
("MulOp", False),
("DivOp", False),
("EqualOp", False),
("AndOp", False),
("NotOp", False),
("PowOp", False),
("OrOp", False),
("SumOp", False),
("AddLhsInplaceOp", True),
("AddRhsInplaceOp", True),
("MulLhsInplaceOp", True),
("MulRhsInplaceOp", True),
("PowLhsInplaceOp", True),
])
# yapf: enable, pylint: enable-all
@pytest.mark.parametrize("connected", [True, False])
def test_binary_ops(op_name: str, inplace: bool, connected: bool) -> None:
    """Test binary ops

    Args:
        op_name (str): The op name e.g. AddOp
        inplace (bool): Whether this op is inplace
        connected (bool): Whether to use the createConnected<opname> function or 
            just create<opname>
    """
    _, graphs = create_ir()
    g = graphs[0]
    binary_op_tester(op_name, g, inplace, connected)


@pytest.mark.parametrize("connected", [True, False])
# yapf: disable, pylint: disable-all
@pytest.mark.parametrize("op_name,kwargs",
[
("HostLoadOp", {"sid_": "streamTensor"}),
("ReluOp", {}),
("GeluOp", {}),
("TransposeOp", {"perm_": [0, 2, 1]}),
("SliceOp", {"starts_":[1], "ends_":[3], "steps_":[1], "axes_":[0]}),
("ReshapeOp", {"s": [3, 1, 2], "handleZero": False}),
("NegateOp", {}),
("TanhOp", {}),
("NotOp", {}),
("SoftmaxOp", {"axis_": 0}),
("SplitOp", {"axis_": 0,"split_": [1]}),
("CastOp", {"_to": _ir.DataType.FLOAT})
])
# yapf: enable, pylint: enable-all
def test_unary_ops(connected: bool, op_name: str,
                   kwargs: Dict[str, Any]) -> None:
    """Test unary (1 in, 1 out) ops

    Args:
        connected (bool): Whether to use the createConnected<opname> function or 
            just create<opname>
        op_name (str): Name of the op e.g. AddOp
        kwargs (Dict[str, Any]): Additional kwargs to pass to the ops
    """
    _, graphs = create_ir()
    g = graphs[0]
    unary_op_tester(op_name, g, False, connected, **kwargs)


@pytest.mark.parametrize("connected", [True, False])
def test_host_store_op(connected: bool) -> None:
    """Test the host store op

    Args:
        connected (bool): Whether to use the createConnected<opname> function or 
            just create<opname>
    """
    _, graphs = create_ir()
    g = graphs[0]
    in0 = add_actgrad_tensor("out0", [1, 2, 3], g)
    opid = _ir.OperatorIdentifier("ai.onnx", "Init", 1, _ir.NumInputs(0, 0), 1)
    settings = _ir.Settings(g, "new_settings")
    if connected:
        op = g.createConnectedOp_HostStoreOp({0: in0.id}, {}, opid, settings,
                                             "streamTensor")
    else:
        op = g.createOp_HostStoreOp(opid, settings, "streamTensor")
        op.connectInTensor(0, in0.id)
    op.setup()

    assert op.inTensor(0) == in0
    assert op.hasInput(0)
    assert not op.hasOutput(0)
    assert op.inId(0) == in0.id


@pytest.mark.parametrize("init_type", [_ir.InitType.Zero, _ir.InitType.NoInit])
@pytest.mark.parametrize("connected", [True, False])
def test_init_op(init_type: "_ir.InitType", connected: bool):
    """Test the special case of the init op

    Args:
        init_type (_ir.InitType): The initialisation type to use (zero/no init)
        connected (bool): Whether to use the createConnected<opname> function or 
            just create<opname>
    """
    _, graphs = create_ir()
    g = graphs[0]
    out0 = add_actgrad_tensor("out0", [1, 2, 3], g)
    opid = _ir.OperatorIdentifier("ai.onnx", "Init", 1, _ir.NumInputs(0, 0), 1)
    settings = _ir.Settings(g, "new_settings")
    if connected:
        op = g.createConnectedOp_InitOp({}, {0: out0.id}, opid, out0.info,
                                        out0.tensorType(), init_type, settings,
                                        0)
    else:
        op = g.createOp_InitOp(opid, out0.info, out0.tensorType(), init_type,
                               settings, 0)
        op.connectOutTensor(0, out0.id)
        op.setup()
    assert not op.hasInput(0)
    assert op.outTensor(0) == out0
    assert op.hasOutput(0)
    assert op.outId(0) == out0.id


@pytest.mark.parametrize("connected", [True, False])
def test_call_op(connected: bool):
    """Test the special case of the call op

    Args:
        connected (bool): Whether to use the createConnected<opname> function or 
            just create<opname>
    """
    _, graphs = create_ir(["sub_graph"])  # main graph and 'sub_graph'
    main = graphs[0]
    sub_graph = graphs[1]
    num_inputs = _ir.NumInputs(1, 1)
    in0 = add_actgrad_tensor("in0", [1, 2, 3], main)
    out0 = add_actgrad_tensor("out0", [1, 2, 3], main)

    sub_graph.addInput("inputA", in0.info)

    opid = _ir.OperatorIdentifier("ai.graphcore", "Call", 1, num_inputs, 1)

    settings = _ir.Settings(main, "new_settings")

    if connected:
        ins: Dict[int, str] = {0: in0.id}
        outs: Dict[int, str] = {0: out0.id}
        op = main.createConnectedOp_CallOp(ins, outs, opid, sub_graph,
                                           settings)
    else:
        op = main.createOp_CallOp(opid, sub_graph, settings)
        op.connectInTensor(0, in0.id)
        op.connectOutTensor(0, out0.id)
        op.setup()

    assert op.getCalledGraphs()[0] == sub_graph
    assert op.getCalledGraphIds()[0] == "sub_graph"


@pytest.mark.parametrize("reduction", [
    _ir.ReductionType.Mean, _ir.ReductionType.Sum,
    _ir.ReductionType.NoReduction
])
@pytest.mark.parametrize("ignoreIndex", [0, 1, 2])
@pytest.mark.parametrize("connected", [True, False])
def test_nll_op(reduction: _ir.ReductionType, ignoreIndex: int,
                connected: bool) -> None:
    """Test the Nll Op, special case with unusual arguments.

    Args:
        reduction (_ir.ReductionType): The reduction type to use
        ignoreIndex (int): The index to ignore. Note this has to be converted to
            an _ir.OptionalInt
        connected (bool): Whether to use the createConnected<opname> function or
            just create<opname>
    """
    _, graphs = create_ir()
    main = graphs[0]
    num_inputs = _ir.NumInputs(2, 2)
    in0 = add_actgrad_tensor("in0", [8, 2, 3], main)

    t_info = _ir.TensorInfo(_ir.DataType.INT32, [8, 2])
    main.addActGrad("label")
    l = main.getTensor("label")
    l.info = t_info

    out0 = add_actgrad_tensor("out0", [1, 2, 3], main)

    opid = _ir.OperatorIdentifier("ai.graphcore", "nll", 1, num_inputs, 1)
    settings = _ir.Settings(main, "nll")

    if connected:
        ins: Dict[int, str] = {0: in0.id, 1: l.id}
        outs: Dict[int, str] = {0: out0.id}
        op = main.createConnectedOp_NllOp(
            ins,
            outs,
            ignoreIndex=_ir.OptionalInt(ignoreIndex),  # <- Note this
            reduction=_ir.ReductionType.Mean,
            inputIsLogProbability=True,
            opid=opid,
            settings=settings)
        return
    op = main.createOp_NllOp(
        opid=opid,
        ignoreIndex=_ir.OptionalInt(ignoreIndex),  # <- Note this
        reduction=_ir.ReductionType.Mean,
        inputIsLogProbability=True,
        settings=settings)
    op.connectInTensor(0, in0.id)
    op.connectInTensor(1, l.id)
    op.connectOutTensor(0, out0.id)
    op.setup()


@pytest.mark.parametrize("connected", [True, False])
def test_gather_op(connected: bool) -> None:
    """Test the Gather Op.

    Args:
        connected (bool): Whether to use the createConnected<opname> function or
            just create<opname>
    """
    _, graphs = create_ir()
    main = graphs[0]
    num_inputs = _ir.NumInputs(2, 2)
    in0 = add_actgrad_tensor("in0", [8], main, _ir.DataType.INT32)
    indices = add_random_tensor("indices", _ir.TensorType.Variable, [16, 4],
                                main)
    out0 = add_actgrad_tensor("out0", [8, 4], main)

    opid = _ir.OperatorIdentifier("ai.graphcore", "Gather", 11, num_inputs, 1)
    settings = _ir.Settings(main, "gather")
    if connected:
        ins: Dict[int, str] = {0: in0.id, 1: indices.id}
        outs: Dict[int, str] = {0: out0.id}
        op = main.createConnectedOp_GatherOp(
            ins,
            outs,
            opid=opid,
            axis_=0,
            available_memory_proportion_=_ir.OptionalFloat(0.4),
            settings=settings)

        op.setup()
        return
    op = main.createOp_GatherOp(
        opid=opid,
        axis_=0,
        available_memory_proportion_=_ir.OptionalFloat(0.4),
        settings=settings)
    op.connectInTensor(0, in0.id)
    op.connectInTensor(1, indices.id)
    op.connectOutTensor(0, out0.id)
    op.setup()


@pytest.mark.parametrize("connected", [True, False])
def test_tiedgather_op(connected: bool) -> None:
    """Test the Tied Gather Op.

    Args:
        connected (bool): Whether to use the createConnected<opname> function or
            just create<opname>
    """
    _, graphs = create_ir()
    main = graphs[0]
    num_inputs = _ir.NumInputs(2, 2)
    in0 = add_actgrad_tensor("in0", [8], main, _ir.DataType.INT32)
    indices = add_random_tensor("indices", _ir.TensorType.Variable, [16, 4],
                                main)
    out0 = add_actgrad_tensor("out0", [8, 4], main)

    settings = _ir.Settings(main, "tiedgather")
    if connected:
        ins: Dict[int, str] = {0: in0.id, 1: indices.id}
        outs: Dict[int, str] = {0: out0.id}
        op = main.createConnectedOp_TiedGatherOp(
            ins,
            outs,
            axis_=0,
            available_memory_proportion_=_ir.OptionalFloat(0.4),
            settings=settings)
        op.setup()
        return
    op = main.createOp_TiedGatherOp(
        axis_=0,
        available_memory_proportion_=_ir.OptionalFloat(0.4),
        settings=settings)
    op.connectInTensor(0, in0.id)
    op.connectInTensor(1, indices.id)
    op.connectOutTensor(0, out0.id)
    op.setup()


@pytest.mark.parametrize("num_groups", [1, 2])
@pytest.mark.parametrize("connected", [True, False])
def test_group_norm_op(connected: bool, num_groups: int) -> None:
    """Test the Group Norm Op.

    Args:
        num_groups (int): The number of groups used by the Op
        connected (bool): Whether to use the createConnected<opname> function or
            just create<opname>
    """
    _, graphs = create_ir()
    main = graphs[0]
    num_inputs = _ir.NumInputs(3, 3)
    in0 = add_actgrad_tensor("in0", [8, 4], main)
    weight = add_random_tensor("weight", _ir.TensorType.Variable, [4], main)
    bias = add_random_tensor("bias", _ir.TensorType.Variable, [4], main)
    out = add_actgrad_tensor("out", [8, 4], main)
    mean = add_actgrad_tensor("mean", [8 * num_groups], main)
    invstddev = add_actgrad_tensor("invstddev", [8 * num_groups], main)

    opid = _ir.OperatorIdentifier("ai.graphcore", "GroupNormalization", 1,
                                  num_inputs, 3)
    settings = _ir.Settings(main, "nll")

    if connected:
        ins: Dict[int, str] = {0: in0.id, 1: weight.id, 2: bias.id}
        outs: Dict[int, str] = {0: out.id, 1: mean.id, 2: invstddev.id}
        op = main.createConnectedOp_GroupNormOp(ins,
                                                outs,
                                                opid=opid,
                                                num_groups_=num_groups,
                                                epsilon_=1e-5,
                                                settings=settings)
        return
    op = main.createOp_GroupNormOp(opid=opid,
                                   num_groups_=num_groups,
                                   epsilon_=1e-5,
                                   settings=settings)
    op.connectInTensor(0, in0.id)
    op.connectInTensor(1, weight.id)
    op.connectInTensor(2, bias.id)
    op.connectOutTensor(0, out.id)
    op.connectOutTensor(1, mean.id)
    op.connectOutTensor(2, invstddev.id)
    op.setup()
