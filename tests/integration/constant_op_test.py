# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart
import onnx
import test_util as tu
import pytest


# The builder does not currently support the `sparse_value`
# attribute of the `Constant` op.
def _create_model_with_sparse_constant():
    data_tensor = onnx.helper.make_tensor(
        'ConstData', onnx.helper.TensorProto.FLOAT, [4], [0.0, 0.1, 0.2, 0.3])

    indices_tensor = onnx.helper.make_tensor(
        'Indices', onnx.helper.TensorProto.INT64, [4], [0, 2, 4, 6])

    sparse_tensor = onnx.helper.make_sparse_tensor(data_tensor, indices_tensor,
                                                   [8])

    const_node = onnx.helper.make_node('Constant', [], ['ConstOut'],
                                       sparse_value=sparse_tensor)
    identity_node = onnx.helper.make_node('Identity', ['ConstOut'],
                                          ['IdentityOut'])

    out_info = onnx.helper.make_tensor_value_info(
        'IdentityOut', onnx.helper.TensorProto.FLOAT, [4])
    graph_def = onnx.helper.make_graph([const_node, identity_node],
                                       'test-model', [], [out_info])

    model_def = onnx.helper.make_model(graph_def, producer_name='onnx-example')
    model_def.opset_import[0].version = 11

    onnx.checker.check_model(model_def)
    return model_def


def test_sparse_data():
    model = _create_model_with_sparse_constant()
    proto = model.SerializeToString()

    anchors = {
        output.name: popart.AnchorReturnType("All")
        for output in model.graph.output
    }

    dataFlow = popart.DataFlow(1, anchors)
    with tu.create_test_device() as device:
        with pytest.raises(popart.popart_exception) as e_info:
            sess = popart.InferenceSession(proto, dataFlow, device)
    assert (e_info.value.args[0] ==
            "The Constant op attribute 'sparse_value' is not supported.")
