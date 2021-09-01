# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import List

import numpy as np
import popart
import popart._internal.ir as _ir
import pytest
from attr import attr

from utils import *


def test_op_creation():
    """Test simple op creation.
    """
    ir, graphs = create_ir(["A"])
    g = graphs[0]
    settings = _ir.Settings(g, "new_settings")
    num_inputs = _ir.NumInputs(1, 1)

    opid = _ir.OperatorIdentifier("ai.onnx", "Identity", 1, num_inputs, 1)
    op = _ir.Op(opid, settings)
    assert op.id == 100  # default Id
    assert op.opid == opid
    assert op.opid.domain == "ai.onnx"
    assert op.opid.type == "Identity"
    assert op.opid.version == 1
    assert op.opid.numOutputs == 1


@pytest.mark.parametrize("attribute,shorthand",
                         [("ExecutionPhase", "ExecutionPhase"),
                          ("BatchSerializedPhase", "BatchSerializedPhase"),
                          ("PipelineStage", "PipelineStage"),
                          ("VirtualGraphId", "VGraphId")])
@pytest.mark.parametrize("input_id", [0, 123, 537896234])
def test_op_attributes(attribute: str, shorthand: str, input_id: int):
    """Test the various attributes that can be applied to ops.

    Args:
        attribute (str): Name of the attribute 
        shorthand (str): Shorthand of the attribute e.g. VirtualGraphId -> VGraphId
        input_id (int): Long int for the id to use for the attribute. 
    """
    _, graphs = create_ir(["A"])
    g = graphs[0]
    settings = _ir.Settings(g, "new_settings")
    num_inputs = _ir.NumInputs(1, 1)

    opid = _ir.OperatorIdentifier("ai.onnx", "Identity", 1, num_inputs, 1)
    op = _ir.Op(opid, settings)
    getter = getattr(op, "get" + attribute)
    setter = getattr(op, "set" + attribute)
    hasser = getattr(op, "has" + attribute)
    get_optional = getattr(op, "getOptional" + shorthand)
    assert not hasser()
    id_ = getattr(_ir, "Optional" + shorthand)
    # Unset optional
    setter(id_())
    assert get_optional() == id_()
    with pytest.raises(popart.popart_exception) as e_info:
        getter()
        assert (e_info.value.args[0] == f"Cannot return {attribute} for Op")

    assert not hasser()
    # Set optional == 0
    setter(id_(input_id))
    assert getter() == input_id
    assert hasser()
    assert get_optional() == id_(input_id)


def test_multi_graph():
    """Test adding ops to multiple graphs.
    """
    ir, graphs = create_ir(["A", "B"])
    g = graphs[0]
    h = graphs[1]
    settings_g = _ir.Settings(g, "settings_g")
    settings_h = _ir.Settings(h, "settings_h")
    num_inputs = _ir.NumInputs(1, 1)

    opid = _ir.OperatorIdentifier("ai.onnx", "Identity", 1, num_inputs, 1)
    op1 = _ir.Op(opid, settings_g)
    op2 = _ir.Op(opid, settings_h)
    assert op1.id == 100  # default Id
    assert op2.id == 101  # default Id + 1
    assert op1.opid == opid == op2.opid
    assert op1.getGraph() == g
    assert op2.getGraph() == h


def test_settings():
    """Test creating settings
    """
    ir, graphs = create_ir(["A"])
    g = graphs[0]
    settings = _ir.Settings(g, "new_settings")

    assert settings.name == "new_settings"
    assert settings.getIr() == ir


def test_op_clone():
    """Op::Clone is pure virtual, this should throw an error. Derived classes should
    be able to call without issue.
    """
    ir, graphs = create_ir(["A"])
    g = graphs[0]
    settings = _ir.Settings(g, "new_settings")
    num_inputs = _ir.NumInputs(1, 1)

    opid = _ir.OperatorIdentifier("ai.onnx", "Identity", 1, num_inputs, 1)
    op = _ir.Op(opid, settings)
    with pytest.raises(RuntimeError) as e_info:
        op2 = op.clone()
        assert (
            e_info.value.args[0] ==
            "RuntimeError: Tried to call pure virtual function \"Op::clone\"")


def test_bools():
    """Test default behaviour of bool returns.
    """
    ir, graphs = create_ir(["A"])
    g = graphs[0]
    settings = _ir.Settings(g, "new_settings")
    num_inputs = _ir.NumInputs(1, 1)

    opid = _ir.OperatorIdentifier("ai.onnx", "Identity", 1, num_inputs, 1)
    op = _ir.Op(opid, settings)
    assert op.isInplaceViewChange() == False
    assert op.isOutplaceViewChange() == False
    assert op.isLossOp() == False
    assert op.isIpuCopyOp() == False
    assert op.isOptimizerOp() == False
    assert op.requiresRandomSeed() == False
    assert op.isOutlineable()
    assert op.hasSideEffect() == False
    assert op.isNorm() == False
    assert op.canBeReplacedByIdentity() == False
    assert op.copiesOptimizerTensors() == False
    assert op.inputsUnmodifiable() == False
    assert op.isElementWiseUnary() == False


def test_graph_in_outs():
    """Test default behaviour for no inputs or outputs.
    """
    ir, graphs = create_ir(["A"])
    g = graphs[0]
    settings = _ir.Settings(g, "new_settings")
    num_inputs = _ir.NumInputs(1, 1)

    opid = _ir.OperatorIdentifier("ai.onnx", "Identity", 1, num_inputs, 1)
    op = _ir.Op(opid, settings)
    assert op.hasInput(0) == False
    assert op.hasOutput(0) == False
    assert op.optionalInputs() == set()
    assert op.getInBatchAxis(0) == 0
    assert op.getOutBatchAxis(0) == 0


@pytest.mark.parametrize("shape1,shape2,expected", [
    [[256, 256, 3], [3], [256, 256, 3]],
    [[8, 1, 6, 1], [7, 1, 5], [8, 7, 6, 5]],
    [[5, 4], [1], [5, 4]],
    [[5, 4], [4], [5, 4]],
    [[15, 3, 5], [15, 1, 5], [15, 3, 5]],
    [[15, 3, 5], [3, 5], [15, 3, 5]],
    [[15, 3, 5], [3, 1], [15, 3, 5]],
])
@pytest.mark.parametrize("dtype", ["FLOAT", "FLOAT16", "BOOL"])
def test_shapes(shape1: List[int], shape2: List[int], expected: List[int],
                dtype: str):
    """Test the shapes and np broadcasting. Don't really need to test the 
    broadcasting as that is tested at C++ level. But try a few cases to be sure
    binding works correctly.

    Args:
        shape1 (List[int]): First tensor shape
        shape2 (List[int]): Second Tensor Shape
        expected (List[int]): Expected shape
        dtype (Str): Popart data type to use
    """
    ir, graphs = create_ir(["A"])
    g = graphs[0]
    settings = _ir.Settings(g, "new_settings")
    num_inputs = _ir.NumInputs(1, 1)

    opid = _ir.OperatorIdentifier("ai.onnx", "Identity", 1, num_inputs, 1)
    op = _ir.Op(opid, settings)
    shape = op.prettyNpOut(shape1, shape2)
    assert shape == list(expected)
    t1 = _ir.TensorInfo(dtype, shape1)
    t2 = _ir.TensorInfo(dtype, shape2)
    shape = op.prettyNpOut(t1, t2)
    assert shape == _ir.TensorInfo(dtype, expected)


@pytest.mark.parametrize(
    "op_name,domain,op_type,op_num,op_version",
    [("OpName123", "ai.onnx", "Identity", 100, 1),
     ("OpName456", "ai.graphcore", "DynamicSlice", 100, 1)])
def test_string_methods(op_name: str, domain: str, op_type: str, op_num: int,
                        op_version: int):
    """Test various string methods (name, id etc)

    Args:
        op_name (str): Name for the op
        domain (str): Domain e.g. ai.onnx
        op_type (str): Op type 
        op_num (int): Op number to test against (default 100)
        op_version (int): Op version
    """
    ir, graphs = create_ir(["A"])
    g = graphs[0]
    settings = _ir.Settings(g, "new_settings")
    num_inputs = _ir.NumInputs(1, 1)

    opid = _ir.OperatorIdentifier(domain, op_type, op_version, num_inputs, 1)
    op = _ir.Op(opid, settings)
    op.setName(op_name)
    assert op.getName() == op_name
    assert op.name() == op_name
    assert op.str() == f"{op_num} ({domain}.{op_type}:{op_version})"
    assert op.debugName(
    ) == f"Op({op_name} ({domain}.{op_type}:{op_version}), inputs=[], outputs=[])"


# TODO: T45197 investigate this
@pytest.mark.skip("Currently failing on CentOS.")
def test_debug_methods():
    """Test the debug info methods work.
    """
    op, _, _ = create_op("ai.onnx", "dummy", 1, 1, 1)

    op.finalizeDebugInfo()


@pytest.mark.parametrize("index", [0, 2, 123])
@pytest.mark.parametrize("id", [i for i in "abcd"])
def test_default_outputs(index: int, id: str):
    """Test default behaviour of createAndConnectOutTensor

    Args:
        index (int): Output index
        id (str): Tensor id
    """
    op, ir, g = create_op("ai.onnx", "dummy", 1, 1, 1)

    op.createAndConnectOutTensor(index, id)
    assert op.hasOutput(index)
    assert op.outTensor(index)
    assert op.outId(index) == id
    assert op.outInfo(index).shape() == []
    assert op.outShape(index) == []
    # TODO: T42845 investigate this.
    # assert op.outInfo(0).dataType() == _ir.DataType.FLOAT # Segfault?
    assert op.outRank(index) == 0
    assert op.outTensorCount() == 1


def test_default_properties():
    """Test default returns for various methods
    """
    op, ir, g = create_op("ai.onnx", "dummy", 1, 1, 1)
    assert op.getCalledGraphs() == []
    assert op.getCalledGraphIds() == []
    for m in [
            "opInToSubgraphInIndex", "subgraphInToOpInIndex",
            "opOutToSubgraphOutIndex", "subgraphOutToOpOutIndex"
    ]:
        with pytest.raises(popart.popart_exception) as e_info:
            method = getattr(op, m)
            method(0, 0)
            assert e_info.value.args[0].startswith(
                "Op Op(new_settings (ai.onnx.dummy:1), inputs=[], outputs=[])")
    assert op.getSubgraphEquivId() == "ai.onnx::dummy::1___NO_none_0_"
    assert op.getSubgraphInputs() == {}
    assert op.getSubgraphOutputs() == {}
    assert op.calcAutoVirtualGraphCost(set()) == 0.0
    assert op.getHighSubgraphValue() == 1000.0
    # floating point error vs python:
    assert round(op.getLowSubgraphValue(), 5) == 0.1
    assert op.inplacePriorityDefault() == []


def test_grad_methods():
    """Test errors for gradient methods (no gradient op will have been generated.)
    """
    op, ir, g = create_op("ai.onnx", "dummy", 1, 1, 1)
    with pytest.raises(popart.popart_exception) as e_info:
        op.gradInputInfo()
        assert e_info.value.args[0].startswith(
            "Op Op(new_settings (ai.onnx.dummy:1), inputs=[], outputs=[])")
    with pytest.raises(popart.popart_exception) as e_info:
        op.gradOutToNonGradIn()
        assert e_info.value.args[0].startswith(
            "Op Op(new_settings (ai.onnx.dummy:1), inputs=[], outputs=[])")


def test_scope():
    """Test setting and getting scope on ops
    """
    op, ir, g = create_op("ai.onnx", "dummy", 1, 1, 1)
    op.setScope(g.getScope())
    assert op.getScope() == g.getScope()


# TODO: T41718 many methods will not work for the generic `Op` class.
# In particular we cannot move a 'no-name' base op type into a graph
# and so cannot test tensor inputs/outputs. Once derived Op classes
# are created add tests here or in test_ops.py
