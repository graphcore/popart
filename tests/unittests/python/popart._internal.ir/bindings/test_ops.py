# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Any, Dict

import pytest
from utils import *

import popart
import popart._internal.ir as _ir


def unary_op_tester(op_name: str,
                    g: _ir.Graph,
                    inplace: bool = False,
                    connected: bool = False,
                    *args,
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
    op = create_new_op(ins, outs, op_name, g, inplace, connected, *args,
                       **kwargs)
    op_assertions(op, ins, outs)


def binary_op_tester(op_name: str,
                     g: _ir.Graph,
                     inplace: bool = False,
                     connected: bool = False,
                     *args,
                     **kwargs):
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
    op = create_new_op(ins, outs, op_name, g, inplace, connected, *args,
                       **kwargs)
    op_assertions(op, ins, outs)


def ternary_op_tester(op_name: str,
                      g: _ir.Graph,
                      inplace: bool = False,
                      connected: bool = False,
                      *args,
                      **kwargs):
    """Helper to test ternary ops

    Args:
        op_name (str): Name of op to create. Must match the create<op_name> function.
        g (_ir.Graph): The graph to add the op to.
        inplace (bool, optional): Whether to use the inplace variant. Defaults to False.
        connected (bool, optional): [Whether to use the createConnected<opname> function or 
            just create<opname>. Defaults to False.
    """
    in0 = add_actgrad_tensor("in0", [1, 2, 3], g)
    in1 = add_random_tensor("in1", _ir.TensorType.Variable, [1, 2, 3], g)
    in2 = add_random_tensor("in2", _ir.TensorType.Variable, [1, 2, 3], g)
    out0 = add_actgrad_tensor("out0", [1, 2, 3], g)
    ins = {0: in0, 1: in1, 2: in2}
    outs = {0: out0}
    op = create_new_op(ins, outs, op_name, g, inplace, connected, *args,
                       **kwargs)
    op_assertions(op, ins, outs)


def op_assertions(op: Any, ins: Dict[int, "_ir.Tensor"],
                  outs: Dict[int, "_ir.Tensor"]):
    """
    Assert that the operators has proper input, output and no call graphs.

    Args:
        op (Any): The operator to check
        ins (dict): The inputs to the operator
        outs (dict): The outputs to the operator
    """
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


@pytest.mark.parametrize("connected", [True, False])
# yapf: disable, pylint: disable-all
@pytest.mark.parametrize("op_name,inplace,kwargs",
[
("DynamicUpdateOp", False, {"axes_":[0], "sizes_":[1], "noOverlap_":False, "updateInInfo_": _ir.TensorInfo()}),
("DynamicUpdateInplaceOp", False, {"axes_":[0], "sizes_":[1], "noOverlap_":False, "updateInInfo_": _ir.TensorInfo()}),
])
# yapf: enable, pylint: enable-all
def test_ternary_ops(connected: bool, inplace: bool, op_name: str,
                     kwargs: Dict[str, Any]) -> None:
    """Test unary (3 in, 1 out) ops

    Args:
        connected (bool): Whether to use the createConnected<opname> function or 
            just create<opname>
        inplace (bool): Whether this op is inplace
        op_name (str): Name of the op e.g. AddOp
        kwargs (Dict[str, Any]): Additional kwargs to pass to the ops
    """
    _, graphs = create_ir()
    g = graphs[0]
    ternary_op_tester(op_name, g, inplace, connected, **kwargs)


# yapf mangles these param lists
# yapf: disable, pylint: disable-all
@pytest.mark.parametrize("op_name,inplace,kwargs",
[("AddOp", False, {}),
("MulOp", False, {}),
("DivOp", False, {}),
("EqualOp", False, {}),
("AndOp", False, {}),
("NotOp", False, {}),
("PowOp", False, {}),
("OrOp", False, {}),
("SumOp", False, {}),
("DynamicSliceOp", False, {"axes_":[0], "sizes_":[1], "noOverlap_":False}),
("AddLhsInplaceOp", True, {}),
("AddRhsInplaceOp", True, {}),
("MulLhsInplaceOp", True, {}),
("MulRhsInplaceOp", True, {}),
("PowLhsInplaceOp", True, {}),
("CopyVarUpdateOp", True, {}),
])
# yapf: enable, pylint: enable-all
@pytest.mark.parametrize("connected", [True, False])
def test_binary_ops(op_name: str, inplace: bool, connected: bool,
                    kwargs: Dict[str, Any]) -> None:
    """Test binary (2 in, 1 out) ops

    Args:
        op_name (str): The op name e.g. AddOp
        inplace (bool): Whether this op is inplace
        connected (bool): Whether to use the createConnected<opname> function or 
            just create<opname>
        kwargs (Dict[str, Any]): Additional kwargs to pass to the ops
    """
    _, graphs = create_ir()
    g = graphs[0]
    binary_op_tester(op_name, g, inplace, connected, **kwargs)


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
("CastOp", {"_to": _ir.DataType.FLOAT}),
("DetachOp", {}),
("DropoutOp", {"ratio_": 0.7}),
("RandomUniformOp", {"shape_": (2,3), "dataType_": _ir.OptionalDataType(_ir.DataType.FLOAT),  "low_": 0.0, "high_": 1.0}),
("RandomNormalOp", {"shape_": (2,3), "dataType_": _ir.OptionalDataType(_ir.DataType.FLOAT),  "mean_": 0.0, "scale_": 1.0}),
("VarUpdateOp", {}),
("VarUpdateWithUpdaterOp", {}),
("IncrementModOp", {"increment_": 1, "modulus_": 3}),
("IncrementModInplaceOp", {"increment_": 1, "modulus_": 3}),
("ReplicatedAllReduceOp", {"op_": _ir.CollectiveOperator.Add, "group": _ir.CommGroup()}),
("ReplicatedReduceScatterOp", {"op_": _ir.CollectiveOperator.Add, "group": _ir.CommGroup()}),
("ReplicatedAllGatherOp", {"group": _ir.CommGroup()}),
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
    unary_op_tester(op_name, g, "Inplace" in op_name, connected, **kwargs)


@pytest.mark.parametrize("connected", [True, False])
@pytest.mark.parametrize("use_offset", [True, False])
def test_remote_store_op(connected: bool, use_offset: bool) -> None:
    """Test that the input and output tensors of remote store op are correct

    Args:
        connected (bool): Whether to use the createConnected<opname> function or 
            just create<opname>
        use_offset (bool): Whether or not to specify the optional offset Tensor
    """
    _, graphs = create_ir()
    g = graphs[0]
    t = add_actgrad_tensor("t", [1, 2, 3], g)
    opid = _ir.OperatorIdentifier("ai.onnx", "Init", 1, _ir.NumInputs(0, 0), 1)
    settings = _ir.Settings(g, "new_settings")

    offset = add_actgrad_tensor("offset", [1], g)
    if use_offset:
        if connected:
            op = g.createConnectedOp_RemoteStoreOp({
                0: t.id,
                1: offset.id
            }, {}, opid, settings, 1)
        else:
            op = g.createOp_RemoteStoreOp(opid, settings, 1)
            op.connectInTensor(0, t.id)
            op.connectInTensor(1, offset.id)
    else:
        if connected:
            op = g.createConnectedOp_RemoteStoreOp({
                0: t.id,
            }, {}, opid, settings, 1)
        else:
            op = g.createOp_RemoteStoreOp(opid, settings, 1)
            op.connectInTensor(0, t.id)

    op.setup()

    assert op.hasInput(0)
    assert op.inTensor(0) == t
    assert op.inId(0) == t.id
    if use_offset:
        assert op.hasInput(1)
        assert op.inTensor(1) == offset
        assert op.inId(1) == offset.id
    else:
        assert not op.hasInput(1)
    assert not op.hasOutput(0)


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


@pytest.mark.parametrize("source,destination", ([0, 1], [2, 4], [3, 1]))
@pytest.mark.parametrize("connected", [True, False])
def test_ipu_copy_op(source: int, destination: int, connected: bool) -> None:
    """Test the ipu copy op

    Args:
        source (int): Source IPU
        destination (int): Destination IPU
        connected (bool): Whether to use the createConnected<opname> function or 
            just create<opname>
    """
    _, graphs = create_ir()
    g = graphs[0]
    in0 = add_actgrad_tensor("in0", [1, 2, 3], g)
    opid = _ir.OperatorIdentifier("ai.graphcore", "IpuCopy", 1,
                                  _ir.NumInputs(0, 0), 1)
    settings = _ir.Settings(g, "new_settings")
    if connected:
        op = g.createConnectedOp_IpuCopyOp({0: in0.id}, {0: "outId"}, opid,
                                           source, destination, settings)
        op.setup()
    else:
        op = g.createOp_IpuCopyOp(opid, destination, settings)
        op.connectInTensor(0, in0.id, source)
        op.createAndConnectOutTensor(0, "outId")
        op.setup()

    assert op.inTensor(0) == in0
    assert op.hasInput(0)
    assert op.hasOutput(0)
    assert op.outId(0) == "outId"
    assert op.getDestIpu() == destination
    assert op.getSourceIpu() == source
    assert op.getSourceIpu("in0") == source
    assert op.getMinSourceIpu() == source
    assert op.getMaxSourceIpu() == source


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


@pytest.mark.parametrize("serialise_mode,serialise_factor",
                         [(_ir.op.SerialiseSettingsMode.NoSerialisation, 0),
                          (_ir.op.SerialiseSettingsMode.InputChannels, 2),
                          (_ir.op.SerialiseSettingsMode.ReducingDim, 2),
                          (_ir.op.SerialiseSettingsMode.OutputChannels, 2)])
@pytest.mark.parametrize(
    "partials_type",
    [_ir.op.MatMulPartialsType.FLOAT, _ir.op.MatMulPartialsType.HALF])
@pytest.mark.parametrize("connected", [True, False])
def test_matmul_op(serialise_mode: _ir.op.SerialiseSettingsMode,
                   serialise_factor: int,
                   partials_type: _ir.op.MatMulPartialsType, connected: bool):
    """Test the bindings for matmul op, a special case op with extra settings.

    Args:
        serialise_mode (_ir.op.SerialiseSettingsMode): Serialisation mode (see matmul.hpp)
        serialise_factor (int): Factor to serialise by.
        partials_type (_ir.op.MatMulPartialsType): Partials calculation type (FLOAT, HALF)
        connected (bool): Whether to use the createConnected<opname> function or 
            just create<opname>
    """
    _, graphs = create_ir()
    g = graphs[0]
    in0 = add_actgrad_tensor("in0", [4, 6], g)
    in1 = add_actgrad_tensor("in1", [6, 12], g)
    out0 = add_actgrad_tensor("out0", [4, 12], g)
    opid = _ir.OperatorIdentifier("ai.onnx", "MatMul", 1, _ir.NumInputs(2, 2),
                                  1)
    serialise_settings = _ir.op.SerialiseSettings()
    serialise_settings.mode = serialise_mode
    serialise_settings.factor = serialise_factor
    optfloat = _ir.OptionalFloat(1.0)
    settings = _ir.Settings(g, "new_settings")
    dtype = _ir.OptionalDataType(_ir.DataType.FLOAT)

    if connected:
        op = g.createConnectedOp_MatMulOp({
            0: in0.id,
            1: in1.id
        }, {0: out0.id}, opid, settings, optfloat, serialise_settings, dtype,
                                          partials_type)
        return

    op = g.createOp_MatMulOp(opid, settings, optfloat, serialise_settings,
                             dtype, partials_type)
    op.connectInTensor(0, in0.id)
    op.connectInTensor(1, in1.id)
    op.connectOutTensor(0, out0.id)
    op.setup()
    assert isinstance(op, _ir.op.MatMulOp)
    assert op.inTensor(0) == in0
    assert op.inTensor(1) == in1
    assert op.outTensor(0) == out0


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


def test_loop_op():
    """Test setting up the loop op.
    """
    _, graphs = create_ir(["sub_graph"])  # main graph and 'sub_graph'
    main = graphs[0]
    sub_graph = graphs[1]
    num_inputs = _ir.NumInputs(2, 2)
    main.addConstInit("loopcount", _ir.TensorInfo(_ir.DataType.INT64, []),
                      np.array(10))
    main.addConstInit("keepgoing", _ir.TensorInfo(_ir.DataType.BOOL, []),
                      np.array(True))
    a = add_actgrad_tensor("a", [1, 2, 3], main, _ir.DataType.FLOAT)
    b = add_actgrad_tensor("b", [1, 2, 3], main, _ir.DataType.FLOAT)

    sub_graph.addInput("loopcount", _ir.TensorInfo(_ir.DataType.INT64, []))
    sub_graph.addInput("keepgoing", _ir.TensorInfo(_ir.DataType.BOOL, []))
    sub_graph.addInput("a", a.info)
    sub_graph.addInput("b", b.info)

    opid = _ir.OperatorIdentifier("ai.graphcore", "Loop", 1, num_inputs, 1)

    settings = _ir.Settings(main, "new_settings")

    add = sub_graph.createConnectedOp_AddOp(
        {
            0: a.id,
            1: b.id
        },
        {0: "out"},
        _ir.OperatorIdentifier("ai.onnx", "Add", 1, num_inputs, 1),
        settings,
    )
    sub_graph.markAsOutput(add.outTensor(0).id)
    sub_graph.markAsOutput("keepgoing")

    ins: Dict[int, str] = {0: "loopcount", 1: "keepgoing", 2: a.id}
    outs: Dict[int, str] = {0: add.outTensor(0).id}
    op = main.createConnectedOp_LoopOp(ins, outs, opid, settings, sub_graph)

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


@pytest.mark.parametrize("connected", [True, False])
def test_scatter_op(connected: bool) -> None:
    """Test the scatter Op.

    Args:
        connected (bool): Whether to use the createConnected<opname> function or
            just create<opname>
    """
    _, graphs = create_ir()
    main = graphs[0]
    num_inputs = _ir.NumInputs(3, 3)
    in0 = add_actgrad_tensor("in0", [3, 3], main, _ir.DataType.INT32)
    indices = add_random_tensor("indices", _ir.TensorType.Variable, [2, 3],
                                main)
    updates = add_actgrad_tensor("updates", [2, 3], main, _ir.DataType.INT32)
    out0 = add_actgrad_tensor("out0", [3, 3], main)

    opid = _ir.OperatorIdentifier("ai.onnx", "Scatter", 11, num_inputs, 1)
    settings = _ir.Settings(main, "scatter")
    if connected:
        ins: Dict[int, str] = {0: in0.id, 1: indices.id, 2: updates.id}
        outs: Dict[int, str] = {0: out0.id}
        op = main.createConnectedOp_ScatterOp(
            ins,
            outs,
            axis_=0,
            opid=opid,
            available_memory_proportion_=_ir.OptionalFloat(0.4),
            settings=settings)
        op.setup()
        return
    op = main.createOp_ScatterOp(
        axis_=0,
        opid=opid,
        available_memory_proportion_=_ir.OptionalFloat(0.4),
        settings=settings)
    op.connectInTensor(0, in0.id)
    op.connectInTensor(1, indices.id)
    op.connectInTensor(2, updates.id)
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


@pytest.mark.parametrize("connected", [True, False])
def test_accumulate_base_op(connected: bool) -> None:
    """Test the AccumulateBaseOp.

    Args:
        connected (bool): Whether to use the createConnected<opname> function or
            just create<opname>
    """
    _, graphs = create_ir()
    main = graphs[0]
    num_inputs = _ir.NumInputs(2, 2)
    input_ = add_actgrad_tensor("input", [4], main)
    updater = add_actgrad_tensor("updater", [4], main)
    factor = add_actgrad_tensor("factor", [4], main)
    out = add_actgrad_tensor("updated_weight", [4], main)

    opid = _ir.OperatorIdentifier("ai.graphcore", "AccumulateBaseOp", 1,
                                  num_inputs, 1)
    settings = _ir.Settings(main, "AccumulateBaseOp")

    if connected:
        ins: Dict[int, str] = {0: input_.id, 1: updater.id, 2: factor.id}
        outs: Dict[int, str] = {0: out.id}
        op = main.createConnectedOp_AccumulateBaseOp(
            ins,
            outs,
            _ir.OperatorIdentifier("ai.graphcore", "AccumulateBaseOp", 1,
                                   _ir.NumInputs(1, 1), 1),
            _ir.AccumulationType.Add,
            _ir.OptimizerValue(0.5),
            settings=settings)
        return
    op = main.createOp_AccumulateBaseOp(_ir.OperatorIdentifier(
        "ai.graphcore", "AccumulateBaseOp", 1, _ir.NumInputs(1, 1), 1),
                                        _ir.AccumulationType.Add,
                                        _ir.OptimizerValue(0.5),
                                        settings=settings)
    op.connectInTensor(0, input_.id)
    op.connectInTensor(1, updater.id)
    op.connectInTensor(2, factor.id)
    op.connectOutTensor(0, out.id)
    op.setup()


@pytest.mark.parametrize("connected", [True, False])
def test_accumulate_op(connected: bool) -> None:
    """Test the Accumulate Op.

    Args:
        connected (bool): Whether to use the createConnected<opname> function or
            just create<opname>
    """
    _, graphs = create_ir()
    main = graphs[0]
    num_inputs = _ir.NumInputs(2, 2)
    weight = add_random_tensor("weight", _ir.TensorType.Variable, [4], main)
    grad = add_actgrad_tensor("grad", [4], main)
    out = add_actgrad_tensor("updated_weight", [4], main)

    opid = _ir.OperatorIdentifier("ai.graphcore", "Accumulate", 1, num_inputs,
                                  1)
    settings = _ir.Settings(main, "accumulate")

    if connected:
        ins: Dict[int, str] = {0: weight.id, 1: grad.id}
        outs: Dict[int, str] = {0: out.id}
        op = main.createConnectedOp_AccumulateOp(ins,
                                                 outs,
                                                 _ir.AccumulationType.Add,
                                                 _ir.OptimizerValue(0.5),
                                                 settings=settings)
        return
    op = main.createOp_AccumulateOp(_ir.AccumulationType.Add,
                                    _ir.OptimizerValue(0.5),
                                    settings=settings)
    op.connectInTensor(0, weight.id)
    op.connectInTensor(1, grad.id)
    op.connectOutTensor(0, out.id)
    op.setup()


@pytest.mark.parametrize("connected", [True, False])
def test_sparse_accumulate_op(connected: bool) -> None:
    """Test the SparseAccumulateOp.

    Args:
        connected (bool): Whether to use the createConnected<opname> function or
            just create<opname>
    """
    _, graphs = create_ir()
    main = graphs[0]
    num_inputs = _ir.NumInputs(2, 2)
    input_ = add_actgrad_tensor("input", [4], main)
    updater = add_actgrad_tensor("updater", [4], main)
    factor = add_actgrad_tensor("factor", [4], main)
    indices = add_actgrad_tensor("indices", [4], main)
    original_var = add_random_tensor("original_var", _ir.TensorType.Variable,
                                     [4], main)
    out = add_actgrad_tensor("updated_weight", [4], main)

    opid = _ir.OperatorIdentifier("ai.graphcore", "SparseAccumulate", 1,
                                  num_inputs, 1)
    settings = _ir.Settings(main, "SparseAccumulate")

    if connected:
        ins: Dict[int, str] = {
            0: input_.id,
            1: updater.id,
            2: factor.id,
            3: indices.id,
            4: original_var.id
        }
        outs: Dict[int, str] = {0: out.id}
        op = main.createConnectedOp_SparseAccumulateOp(
            ins,
            outs,
            _ir.AccumulationType.Add,
            _ir.OptimizerValue(0.5),
            0,
            settings=settings)
        return
    op = main.createOp_SparseAccumulateOp(_ir.AccumulationType.Add,
                                          _ir.OptimizerValue(0.5),
                                          0,
                                          settings=settings)
    op.connectInTensor(0, input_.id)
    op.connectInTensor(1, updater.id)
    op.connectInTensor(2, factor.id)
    op.connectInTensor(3, indices.id)
    op.connectInTensor(4, original_var.id)
    op.connectOutTensor(0, out.id)
    op.setup()


@pytest.mark.parametrize("connected", [True, False])
def test_accumulate_scale_op(connected: bool) -> None:
    """Test the AccumulatorScaleOp.

    Args:
        connected (bool): Whether to use the createConnected<opname> function or
            just create<opname>
    """
    _, graphs = create_ir()
    main = graphs[0]
    num_inputs = _ir.NumInputs(2, 2)
    input_ = add_actgrad_tensor("input", [4], main)
    updater = add_actgrad_tensor("updater", [4], main)
    factor = add_actgrad_tensor("factor", [4], main)
    out = add_actgrad_tensor("updated_weight", [4], main)

    settings = _ir.Settings(main, "AccumulatorScaleOp")

    if connected:
        ins: Dict[int, str] = {
            0: input_.id,
            1: updater.id,
            2: factor.id,
        }
        outs: Dict[int, str] = {0: out.id}
        op = main.createConnectedOp_AccumulatorScaleOp(ins,
                                                       outs,
                                                       _ir.OptimizerValue(0.5),
                                                       settings=settings)
        return
    op = main.createOp_AccumulatorScaleOp(_ir.OptimizerValue(0.5),
                                          settings=settings)
    op.connectInTensor(0, input_.id)
    op.connectInTensor(1, updater.id)
    op.connectInTensor(2, factor.id)
    op.connectOutTensor(0, out.id)
    op.setup()


@pytest.mark.parametrize("connected", [True, False])
def test_accumulate_zero_op(connected: bool) -> None:
    """Test the AccumulatorZeroOp.

    Args:
        connected (bool): Whether to use the createConnected<opname> function or
            just create<opname>
    """
    _, graphs = create_ir()
    main = graphs[0]
    num_inputs = _ir.NumInputs(2, 2)
    input_ = add_actgrad_tensor("input", [4], main)
    updater = add_actgrad_tensor("updater", [4], main)
    factor = add_actgrad_tensor("factor", [4], main)
    out = add_actgrad_tensor("updated_weight", [4], main)

    opid = _ir.OperatorIdentifier("ai.graphcore", "AccumulatorZeroOp", 1,
                                  num_inputs, 1)
    settings = _ir.Settings(main, "AccumulatorZeroOp")

    if connected:
        ins: Dict[int, str] = {0: input_.id, 1: updater.id, 2: factor.id}
        outs: Dict[int, str] = {0: out.id}
        op = main.createConnectedOp_AccumulatorZeroOp(ins,
                                                      outs,
                                                      settings=settings)
        return
    op = main.createOp_AccumulatorZeroOp(settings=settings)
    op.connectInTensor(0, input_.id)
    op.connectInTensor(1, updater.id)
    op.connectInTensor(2, factor.id)
    op.connectOutTensor(0, out.id)
    op.setup()


@pytest.mark.parametrize("connected", [True, False])
def test_adamupdater_op(connected: bool) -> None:
    """Test the AdamUpdater Op.

    Args:
        connected (bool): Whether to use the createConnected<opname> function or
            just create<opname>
    """
    _, graphs = create_ir()
    g = graphs[0]
    w = add_random_tensor("w", _ir.TensorType.Variable, [0, 1], g)
    mode = _ir.AdamMode.Adam
    wd = _ir.OptimizerValue(0.02)
    b1 = _ir.OptimizerValue(0.9)
    b2 = _ir.OptimizerValue(0.99)
    eps = _ir.OptimizerValue(0.4)
    m = add_random_tensor("m", _ir.TensorType.Variable, [4, 1], g)
    v = add_random_tensor("v", _ir.TensorType.Variable, [2, 1], g)
    t = add_random_tensor("t", _ir.TensorType.Variable, [1], g)
    out = add_actgrad_tensor("out", [1, 4], g)

    ins = {0: w.id, 1: m.id, 2: v.id, 3: t.id}
    outs: Dict[int, str] = {0: out.id}
    settings = _ir.Settings(g, "new_settings")

    if connected:
        op = g.createConnectedOp_AdamUpdaterOp(ins, outs, mode, wd, b1, b2,
                                               eps, settings)
        op.setup()
        return
    op = g.createOp_AdamUpdaterOp(mode, wd, b1, b2, eps, settings)
    op.connectInTensor(0, w.id)
    op.connectInTensor(1, m.id)
    op.connectInTensor(2, v.id)
    op.connectInTensor(3, t.id)

    op.connectOutTensor(0, out.id)
    op.setup()
