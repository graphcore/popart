# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Dict, List, Tuple
import pytest
import popart._internal.ir as _ir
import numpy as np
import popart
from utils import *
"""
Currently bound:
['AddArg0GradOp', 'AddArg1GradOp', 'AddLhsInplaceOp', 'AddOp', 'AddRhsInplaceOp',
 'CallGradOp', 'CallOp', 'InitOp', 'MulArg0GradOp', 'MulArg1GradOp', 'MulLhsInplaceOp',
  'MulOp', 'MulRhsInplaceOp', '__doc__', '__loader__', '__name__', '__package__', '__spec__', 'exchange']
"""


def unary_op_tester(op_name: str,
                    g: _ir.Graph,
                    inplace: bool = False,
                    connected: bool = False,
                    *args):
    in0 = add_actgrad_tensor("in0", [1, 2, 3], g)
    out0 = add_actgrad_tensor("out0", [1, 2, 3], g)
    ins = {0: in0}
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


def binary_op_tester(op_name: str,
                     g: _ir.Graph,
                     inplace: bool = False,
                     connected: bool = False,
                     *args):
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


@pytest.mark.parametrize("op_name,inplace", [("AddOp", False),
                                             ("MulOp", False),
                                             ("AddLhsInplaceOp", True),
                                             ("AddRhsInplaceOp", True),
                                             ("MulLhsInplaceOp", True),
                                             ("MulRhsInplaceOp", True)])
@pytest.mark.parametrize("connected", [True, False])
def test_binary_ops(op_name, inplace, connected):
    _, graphs = create_ir()
    g = graphs[0]
    binary_op_tester(op_name, g, inplace, connected)


@pytest.mark.parametrize("op_name,args", [("HostLoadOp", ("streamTensor"))])
@pytest.mark.parametrize("connected", [True, False])
def test_unary_ops(op_name: str, args: Tuple[str, Tuple], connected: bool):
    _, graphs = create_ir()
    g = graphs[0]
    unary_op_tester(op_name, g, False, connected, args)


@pytest.mark.parametrize("connected", [True, False])
def test_host_store_op(connected: bool):
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
    _, graphs = create_ir(["sub_graph"])  # main graph and 'sub_graph'
    main = graphs[0]
    sub_graph = graphs[1]
    num_inputs = _ir.NumInputs(1, 1)
    in0 = add_actgrad_tensor("in0", [1, 2, 3], main)
    out0 = add_actgrad_tensor("out0", [1, 2, 3], main)

    sub_graph.addInput("inputA", in0.info)

    opid = _ir.OperatorIdentifier("ai.onnx", "Call", 1, num_inputs, 1)

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
