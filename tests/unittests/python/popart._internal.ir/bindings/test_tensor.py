# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import popart._internal.ir as _ir
import pytest


def test_tensor_type_creation():
    """ Test that we can create a popart._internal.ir.TensorType enum. """
    _ir.TensorType.ActGrad
    _ir.TensorType.Const
    _ir.TensorType.Stream
    _ir.TensorType.Unknown
    _ir.TensorType.Variable
    _ir.TensorType.N


def test_variable_update_type_creation():
    """ Test that we can create a popart._internal.ir.VariableUpdateType enum.
    """
    _ir.VariableUpdateType.None_
    _ir.VariableUpdateType.Gradient
    _ir.VariableUpdateType.Copy


def test_tensor_type_info_construction():
    """ Test that we can construct a popart._internal.ir.TensorTypeInfo object.
    """
    _ = _ir.TensorTypeInfo(_ir.TensorType.ActGrad, "ActGrad")


def test_tensor_type_info_type_type_s():
    """ Test the type() and type_s methods of a
    popart._internal.ir.TensorTypeInfo object.
    """
    type = _ir.TensorType.Const
    typeS = "Const"
    tTypeInfo = _ir.TensorTypeInfo(type, typeS)
    assert tTypeInfo.type() == type
    assert tTypeInfo.type_s() == typeS


def test_tensor_construction():
    """ Test that we can construct a popart._internal.ir.Tensor object. """
    ir = _ir.Ir()
    g = ir.createGraph("g")
    tId = popart.TensorId("t")
    tType = _ir.TensorType.ActGrad
    dc = _ir.DebugContext()
    _ = _ir.Tensor(tId, tType, g)
    _ = _ir.Tensor(tId, tType, g, dc)


def test_tensor_str():
    """ Test the str() method of a popart._internal.ir.Tensor object. """
    ir = _ir.Ir()
    g = ir.createGraph("g")
    Tensor = lambda id: _ir.Tensor(id, _ir.TensorType.ActGrad, g)
    assert Tensor(popart.TensorId("t0")).str() == "t0"
    assert Tensor(popart.TensorId("t1")).str() == "t1"


def test_tensor_clone():
    """ Test the clone() method of a popart._internal.ir.Tensor object. """
    ir = _ir.Ir()
    g = ir.createGraph("g")
    t0 = _ir.Tensor(popart.TensorId("t0"), _ir.TensorType.ActGrad, g)
    t1 = t0.clone(g)
    assert f"clone_{t0.str()}" == t1.str()
    assert t0.info == t1.info


def test_tensor_tensor_type0():
    """ Test the tensorType() method of a popart._internal.ir.Tensor object. """
    ir = _ir.Ir()
    g = ir.createGraph("g")
    Tensor = lambda id, type: _ir.Tensor(id, type, g)
    tTypes = [_ir.TensorType.ActGrad, _ir.TensorType.Const]
    for i, tType in enumerate(tTypes):
        assert Tensor(popart.TensorId(f"t{i}"), tType).tensorType() == tType


def test_tensor_tensor_type1():
    """ Test the tensor_type() method of a popart._internal.ir.Tensor object.
    """
    ir = _ir.Ir()
    g = ir.createGraph("g")
    Tensor = lambda id, type: _ir.Tensor(id, type, g)
    tTypes = {_ir.TensorType.ActGrad: "ActGrad", _ir.TensorType.Const: "Const"}
    for i, (tType, tTypeStr) in enumerate(tTypes.items()):
        assert Tensor(popart.TensorId(f"t{i}"),
                      tType).tensor_type() == tTypeStr


def test_tensor_get_tensor_type_info():
    """ Test the getTensorTypeInfo() method of a popart._internal.ir.Tensor
    object.
    """
    ir = _ir.Ir()
    g = ir.createGraph("g")
    Tensor = lambda id, type: _ir.Tensor(popart.TensorId(id), type, g)
    tTypes = {_ir.TensorType.ActGrad: "ActGrad", _ir.TensorType.Const: "Const"}
    for i, (tType, tTypeStr) in enumerate(tTypes.items()):
        tTypeInfo = Tensor(f"t{i}", tType).getTensorTypeInfo()
        assert tTypeInfo.type() == tType
        assert tTypeInfo.type_s() == tTypeStr


def test_tensor_set_tensor_type():
    """ Test the setTensorType() method of a popart._internal.ir.Tensor object.
    """
    ir = _ir.Ir()
    g = ir.createGraph("g")
    tTypeOld = _ir.TensorType.ActGrad
    tTypeNew = _ir.TensorType.Const
    t = _ir.Tensor(popart.TensorId("t"), tTypeOld, g)
    assert t.tensorType() == tTypeOld
    t.setTensorType(tTypeNew)
    assert t.tensorType() == tTypeNew


def test_tensor_get_set_replicated_streaming_mode():
    """ Test the getReplicatedStreamMode() and setReplicatedStreamMode() methods
    of a popart._internal.ir.Tensor object.
    """
    ir = _ir.Ir()
    g = ir.createGraph("g")
    t = _ir.Tensor(popart.TensorId("t"), _ir.TensorType.ActGrad, g)
    assert t.getReplicatedStreamMode(
    ) == _ir.Tensor.ReplicatedStreamMode.Replicate
    t.setReplicatedStreamMode(_ir.Tensor.ReplicatedStreamMode.Broadcast)
    assert t.getReplicatedStreamMode(
    ) == _ir.Tensor.ReplicatedStreamMode.Broadcast


def test_tensor_has_tensor_data():
    """ Test the hasTensorData() method of a popart._internal.ir.Tensor object.
    """
    ir = _ir.Ir()
    g = ir.createGraph("g")
    t = _ir.Tensor(popart.TensorId("t"), _ir.TensorType.ActGrad, g)
    assert t.hasTensorData() == False
    buffer = np.random.rand(2, 3, 4)
    tInfo = _ir.TensorInfo(_ir.DataType.FLOAT, buffer.shape)
    t.setTensorData(tInfo, buffer)
    assert t.hasTensorData() == True


def test_tensor_tensor_data():
    """ Test the tensorData() and setTensorData() methods of a
    popart._internal.ir.Tensor object.
    """
    ir = _ir.Ir()
    g = ir.createGraph("g")
    t = _ir.Tensor(popart.TensorId("t"), _ir.TensorType.ActGrad, g)

    with pytest.raises(popart.popart_exception) as e_info:
        t.tensorData()
        assert e_info.value.args[0] == "Data not set for t"
    with pytest.raises(popart.popart_exception) as e_info:
        t.tensorData_const()
        assert e_info.value.args[0] == "Data not set for t"

    buffer = np.random.rand(2, 3, 4)
    tInfo = _ir.TensorInfo(_ir.DataType.FLOAT, buffer.shape)
    t.setTensorData(tInfo, buffer)

    # TODO(T42205): Test that the returned tensor data matches the one that was
    # set.
    t.tensorData()
    t.tensorData_const()


def test_tensor_get_graph():
    """ Test the getGraph() method of a popart._internal.ir.Tensor object. """
    ir = _ir.Ir()
    g = ir.createGraph("g")
    t = _ir.Tensor(popart.TensorId("t"), _ir.TensorType.ActGrad, g)
    gFromTensor = t.getGraph()
    assert g.id == gFromTensor.id
    gFromTensor = t.getGraph_const()
    assert g.id == gFromTensor.id


def test_tensor_get_ir():
    """ Test the getIr() method of a popart._internal.ir.Tensor object. """
    ir = _ir.Ir()
    g = ir.createGraph("g")
    t = _ir.Tensor(popart.TensorId("t"), _ir.TensorType.ActGrad, g)
    irFromTensor = t.getIr()
    assert g.id == irFromTensor.getAllGraphs()[1].id
    irFromTensor = t.getIr_const()
    assert g.id == irFromTensor.getAllGraphs()[1].id


def test_tensor_has_virtual_graph_id():
    """ Test the hasVirtualGraphId() method of a popart._internal.ir.Tensor
    object.
    """
    ir = _ir.Ir()
    g = ir.createGraph("g")
    t = _ir.Tensor(popart.TensorId("t"), _ir.TensorType.ActGrad, g)
    # TODO(T42205): Test that hasVirtualGraphId() returns the expected values.
    t.hasVirtualGraphId()


def test_tensor_get_virtual_graph_id():
    """ Test the getVirtualGraphId() method of a popart._internal.ir.Tensor
    object.
    """
    ir = _ir.Ir()
    g = ir.createGraph("g")
    t = _ir.Tensor(popart.TensorId("t"), _ir.TensorType.ActGrad, g)
    with pytest.raises(popart.popart_exception) as e_info:
        t.getVirtualGraphId()
        assert e_info.value.args[0] == (
            "Invalid call to getVirtualGraphId, Tensor does not have one")
    # TODO(T42205): Test that getVirtualGraphId() returns the expected values.


def test_tensor_get_virtual_graph_id_unsafe():
    """ Test the getVirtualGraphIdUnsafe() method of a
    popart._internal.ir.Tensor object.
    """
    ir = _ir.Ir()
    g = ir.createGraph("g")
    t = _ir.Tensor(popart.TensorId("t"), _ir.TensorType.ActGrad, g)
    # TODO(T42205): Test that getVirtualGraphIdUnsafe() returns the expected
    # values.
    t.getVirtualGraphIdUnsafe()


def test_tensor_get_batch_axis():
    """ Test the getBatchAxis() method of a popart._internal.ir.Tensor object.
    """
    ir = _ir.Ir()
    g = ir.createGraph("g")
    t = _ir.Tensor(popart.TensorId("t"), _ir.TensorType.ActGrad, g)
    assert t.getBatchAxis() == -1
    # TODO(T42205): Test that getBatchAxis() returns the expected values when
    # the tensor has producers/consumers.


def test_tensor_get_debug_info():
    """ Test the getDebugInfo() method of a popart._internal.ir.Tensor object.
    """
    ir = _ir.Ir()
    g = ir.createGraph("g")
    t = _ir.Tensor(popart.TensorId("t"), _ir.TensorType.ActGrad, g)
    _ = t.getDebugInfo()


def test_tensor_id():
    """ Test the id attribute of a popart._internal.ir.Tensor object. """
    ir = _ir.Ir()
    g = ir.createGraph("g")
    t = _ir.Tensor(popart.TensorId("t"), _ir.TensorType.ActGrad, g)
    assert t.id == "t"


def test_replicated_stream_mode_creation():
    """ Test that we can create a
    popart._internal.ir.Tensor.ReplicatedStreamMode enum.
    """
    _ir.Tensor.ReplicatedStreamMode.Replicate
    _ir.Tensor.ReplicatedStreamMode.Broadcast


# TODO(T42205): Write unit test for the following methods and attributes of the
# Tensor class:
#   - Tensor.isUnmodifiable()
#   - Tensor.isCheckpointTensor()
#   - Tensor.isImplicitRecomputeTensor()
#   - Tensor.isRestoreInplaceTensor()
#   - Tensor.idIncludesPrefix()
#   - Tensor.isOptimizerTensor()
#   - Tensor.isRemoteArgTensor()
#   - Tensor.isRandomSeedTensor()
#   - Tensor.isOptimizerStateTensor()
#   - Tensor.isAccumulatorTensor()
#   - Tensor.isHostLoadTensor()
#   - Tensor.isWeightTensor()
#   - Tensor.isAnchored()
#   - Tensor.isRootAnchor()
#   - Tensor.anyAlias()
#   - Tensor.associatedOps()
#   - Tensor.getVirtualGraphIdAndTileSet()
#   - Tensor.getVirtualGraphIdAndTileSetUnsafe()
#   - Tensor.consumersAllPreLoss()
#   - Tensor.isModified()
#   - Tensor.isAliased()
#   - Tensor.getDataViaGraphTraversal()
#   - Tensor.consumers
#   - Tensor.info
#   - Tensor.tensorLocationInfo
#   - Tensor.inputSettings
