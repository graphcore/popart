# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart
import popart._internal.ir as _ir
import pytest


def test_tensor_type_creation():
    """Test that we can create a popart._internal.ir.TensorType enum."""
    _ir.TensorType.ActGrad
    _ir.TensorType.Const
    _ir.TensorType.Stream
    _ir.TensorType.Unknown
    _ir.TensorType.Variable
    _ir.TensorType.N


def test_variable_update_type_creation():
    """Test that we can create a popart._internal.ir.VariableUpdateType enum."""
    _ir.VariableUpdateType.None_
    _ir.VariableUpdateType.Gradient
    _ir.VariableUpdateType.Copy


def test_tensor_construction():
    """Test that we can construct a popart._internal.ir.Tensor object."""
    ir = _ir.Ir()
    g = ir.createGraph("g")
    tId = "t"
    tType = _ir.TensorType.ActGrad
    dc = _ir.DebugContext()
    _ = _ir.Tensor(tId, tType, g)
    _ = _ir.Tensor(tId, tType, g, dc)


def test_tensor_str():
    """Test the str() method of a popart._internal.ir.Tensor object."""
    ir = _ir.Ir()
    g = ir.createGraph("g")
    Tensor = lambda id: _ir.Tensor(id, _ir.TensorType.ActGrad, g)
    assert Tensor("t0").str() == "t0"
    assert Tensor("t1").str() == "t1"


def test_tensor_clone():
    """Test the clone() method of a popart._internal.ir.Tensor object."""
    ir = _ir.Ir()
    g = ir.createGraph("g")
    t0 = _ir.Tensor("t0", _ir.TensorType.ActGrad, g)
    t1 = t0.clone(g)
    assert f"clone_{t0.str()}" == t1.str()
    assert t0.info == t1.info


def test_tensor_tensor_type0():
    """Test the tensorType() method of a popart._internal.ir.Tensor object."""
    ir = _ir.Ir()
    g = ir.createGraph("g")
    Tensor = lambda id, type: _ir.Tensor(id, type, g)
    tTypes = [_ir.TensorType.ActGrad, _ir.TensorType.Const]
    for i, tType in enumerate(tTypes):
        assert Tensor(f"t{i}", tType).tensorType() == tType


def test_tensor_tensor_type1():
    """Test the tensor_type() method of a popart._internal.ir.Tensor object."""
    ir = _ir.Ir()
    g = ir.createGraph("g")
    Tensor = lambda id, type: _ir.Tensor(id, type, g)
    tTypes = {_ir.TensorType.ActGrad: "ActGrad", _ir.TensorType.Const: "Const"}
    for i, (tType, tTypeStr) in enumerate(tTypes.items()):
        assert Tensor(f"t{i}", tType).tensor_type() == tTypeStr


def test_tensor_set_tensor_type():
    """Test the setTensorType() method of a popart._internal.ir.Tensor object."""
    ir = _ir.Ir()
    g = ir.createGraph("g")
    tTypeOld = _ir.TensorType.ActGrad
    tTypeNew = _ir.TensorType.Const
    t = _ir.Tensor("t", tTypeOld, g)
    assert t.tensorType() == tTypeOld
    t.setTensorType(tTypeNew)
    assert t.tensorType() == tTypeNew


def test_tensor_get_set_replicated_streaming_mode():
    """Test the getReplicatedStreamMode() and setReplicatedStreamMode() methods
    of a popart._internal.ir.Tensor object.
    """
    ir = _ir.Ir()
    g = ir.createGraph("g")
    t = _ir.Tensor("t", _ir.TensorType.ActGrad, g)
    assert t.getReplicatedStreamMode() == popart.ReplicatedStreamMode.Replicate
    t.setReplicatedStreamMode(popart.ReplicatedStreamMode.Broadcast)
    assert t.getReplicatedStreamMode() == popart.ReplicatedStreamMode.Broadcast


def test_tensor_has_tensor_data():
    """Test the hasTensorData() method of a popart._internal.ir.Tensor object."""
    ir = _ir.Ir()
    g = ir.createGraph("g")
    t = _ir.Tensor("t", _ir.TensorType.ActGrad, g)
    assert not t.hasTensorData()


def test_tensor_get_graph():
    """Test the getGraph() method of a popart._internal.ir.Tensor object."""
    ir = _ir.Ir()
    g = ir.createGraph("g")
    t = _ir.Tensor("t", _ir.TensorType.ActGrad, g)
    gFromTensor = t.getGraph()
    assert g.id == gFromTensor.id
    gFromTensor = t.getGraph_const()
    assert g.id == gFromTensor.id


def test_tensor_get_ir():
    """Test the getIr() method of a popart._internal.ir.Tensor object."""
    ir = _ir.Ir()
    g = ir.createGraph("g")
    t = _ir.Tensor("t", _ir.TensorType.ActGrad, g)
    irFromTensor = t.getIr()
    assert g.id == irFromTensor.getAllGraphs()[1].id
    irFromTensor = t.getIr_const()
    assert g.id == irFromTensor.getAllGraphs()[1].id


def test_tensor_has_virtual_graph_id():
    """Test the hasVirtualGraphId() method of a popart._internal.ir.Tensor
    object.
    """
    ir = _ir.Ir()
    g = ir.createGraph("g")
    t = _ir.Tensor("t", _ir.TensorType.ActGrad, g)
    t.hasVirtualGraphId()


def test_tensor_get_virtual_graph_id():
    """Test the getVirtualGraphId() method of a popart._internal.ir.Tensor
    object.
    """
    ir = _ir.Ir()
    g = ir.createGraph("g")
    t = _ir.Tensor("t", _ir.TensorType.ActGrad, g)
    with pytest.raises(popart.popart_exception) as e_info:
        t.getVirtualGraphId()
        assert e_info.value.args[0] == (
            "Invalid call to getVirtualGraphId, Tensor does not have one"
        )


def test_tensor_get_virtual_graph_id_unsafe():
    """Test the getVirtualGraphIdUnsafe() method of a
    popart._internal.ir.Tensor object.
    """
    ir = _ir.Ir()
    g = ir.createGraph("g")
    t = _ir.Tensor("t", _ir.TensorType.ActGrad, g)
    t.getVirtualGraphIdUnsafe()


def test_tensor_get_batch_axis():
    """Test the getBatchAxis() method of a popart._internal.ir.Tensor object."""
    ir = _ir.Ir()
    g = ir.createGraph("g")
    t = _ir.Tensor("t", _ir.TensorType.ActGrad, g)
    assert t.getBatchAxis() == -1


def test_tensor_get_debug_info():
    """Test the getDebugInfo() method of a popart._internal.ir.Tensor object."""
    ir = _ir.Ir()
    g = ir.createGraph("g")
    t = _ir.Tensor("t", _ir.TensorType.ActGrad, g)
    _ = t.getDebugInfo()


def test_tensor_id():
    """Test the id attribute of a popart._internal.ir.Tensor object."""
    ir = _ir.Ir()
    g = ir.createGraph("g")
    t = _ir.Tensor("t", _ir.TensorType.ActGrad, g)
    assert t.id == "t"


def test_replicated_stream_mode_creation():
    """Test that we can create a
    popart._internal.ir.Tensor.ReplicatedStreamMode enum.
    """
    popart.ReplicatedStreamMode.Replicate
    popart.ReplicatedStreamMode.Broadcast
