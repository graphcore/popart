# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest
import test_util as tu
import torch
import onnx
from onnx import helper, numpy_helper, version_converter
from onnx import AttributeProto, TensorProto, GraphProto
from onnx.backend.base import Backend
from onnx import numpy_helper


def test_type_cast_UINT8ToINT32():

    # Build an onnx proto with a single constant node
    # Create output tensors of the type to cast
    X = helper.make_tensor_value_info('X', TensorProto.UINT8, [3, 2])
    Y = helper.make_tensor_value_info('Y', TensorProto.UINT8, [3, 2])

    values = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.uint8)
    # We use ConstantOfShape even if it is not followint the onnx
    # spec because we need an op that can store several actual values
    # in the graph to make sure the type conversion is succesful. Constant
    # would not work here becauseit has only been available since opset 12.
    # Constant-9 is not compatible with the default opset-version 11
    node_def = onnx.helper.make_node('ConstantOfShape',
                                     inputs=['X'],
                                     outputs=['Y'],
                                     value=onnx.helper.make_tensor(
                                         name='const_tensor',
                                         data_type=onnx.TensorProto.UINT8,
                                         dims=values.shape,
                                         vals=values.flatten().astype(
                                             np.uint8).tobytes(),
                                         raw=True))

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [X],
        [Y],
    )

    # Create the model (ModelProto)
    onnx_model = helper.make_model(graph_def)

    # Make sure the opset version is version 9 (by default it would be 11
    # which would crash subsequent function calls)
    onnx_model = version_converter.convert_version(onnx_model, 9)

    # Compile the model to an onnx graph
    onnx.save_model(onnx_model, "type_test.onnx")

    # Load proto into a graph transfomer and apply cast
    graph_transformer = popart.GraphTransformer("type_test.onnx")
    graph_transformer.convertUINT8ToINT32()

    # Retrieve modeified graph proto
    proto = graph_transformer.getModelProto()
    popart.Builder(proto).saveModelProto("type_test_modified.onnx")

    # Load the model as an onnx model again
    # modified_onnx_model = onnx.load(proto)
    modified_onnx_model = onnx.load("type_test_modified.onnx")

    # Make sure the graph is still good
    onnx.checker.check_model(modified_onnx_model)

    # Get only the first input of the input array (there should only be one)
    i = modified_onnx_model.graph.input[0]
    o = modified_onnx_model.graph.output[0]

    input_type = i.type.tensor_type
    output_type = o.type.tensor_type

    # Make sure shapes remain untouched
    assert (input_type.HasField("shape")
            ), "Modified graph output has no shape attribute"
    assert (output_type.HasField("shape")
            ), "Modified graph output has no shape attribute"
    assert (input_type.shape.dim[0].dim_value == 3
            ), "Dimensions were not conserved by cast"
    assert (input_type.shape.dim[1].dim_value == 2
            ), "Dimensions were not conserved by cast"
    assert (output_type.shape.dim[0].dim_value == 3
            ), "Dimensions were not conserved by cast"
    assert (output_type.shape.dim[1].dim_value == 2
            ), "Dimensions were not conserved by cast"

    # Test whether the new tensor has the right size
    assert (len(
        modified_onnx_model.graph.node[0].attribute[0].t.int32_data) == len(
            onnx_model.graph.node[0].attribute[0].t.raw_data)
            ), "Wrong number of Bytes in casted version."

    # Retrieve the two constant tensors and compare the values
    assert np.allclose(
        modified_onnx_model.graph.node[0].attribute[0].t.int32_data,
        values.flatten()), "Data was not conserved by cast"


def test_type_cast_UINT16ToINT32():

    # Build an onnx proto with a single constant node
    # Create output tensors of the type to cast
    X = helper.make_tensor_value_info('X', TensorProto.UINT16, [3, 2])
    Y = helper.make_tensor_value_info('Y', TensorProto.UINT16, [3, 2])

    values = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.uint16)
    node_def = onnx.helper.make_node('ConstantOfShape',
                                     inputs=['X'],
                                     outputs=['Y'],
                                     value=onnx.helper.make_tensor(
                                         name='const_tensor',
                                         data_type=onnx.TensorProto.UINT16,
                                         dims=values.shape,
                                         vals=values.flatten().astype(
                                             np.uint16).tobytes(),
                                         raw=True))

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [X],
        [Y],
    )

    # Create the model (ModelProto)
    onnx_model = helper.make_model(graph_def)

    # Make sure the opset version is version 9 (by default it would be 11
    # which would crash subsequent function calls)
    onnx_model = version_converter.convert_version(onnx_model, 9)

    # Compile the model to an onnx graph
    onnx.save_model(onnx_model, "type_test.onnx")

    # Load proto into a graph transfomer and apply cast
    graph_transformer = popart.GraphTransformer("type_test.onnx")
    graph_transformer.convertUINT16ToINT32()

    # Retrieve modeified graph proto
    proto = graph_transformer.getModelProto()
    popart.Builder(proto).saveModelProto("type_test_modified.onnx")

    # Load the model as an onnx model again
    # modified_onnx_model = onnx.load(proto)
    modified_onnx_model = onnx.load("type_test_modified.onnx")

    # Make sure the graph is still good
    onnx.checker.check_model(modified_onnx_model)

    # Get only the first input of the input array (there should only be one)
    i = modified_onnx_model.graph.input[0]
    o = modified_onnx_model.graph.output[0]

    input_type = i.type.tensor_type
    output_type = o.type.tensor_type

    # Make sure shapes remain untouched
    assert (input_type.HasField("shape")
            ), "Modified graph output has no shape attribute"
    assert (output_type.HasField("shape")
            ), "Modified graph output has no shape attribute"
    assert (input_type.shape.dim[0].dim_value == 3
            ), "Dimensions were not conserved by cast"
    assert (input_type.shape.dim[1].dim_value == 2
            ), "Dimensions were not conserved by cast"
    assert (output_type.shape.dim[0].dim_value == 3
            ), "Dimensions were not conserved by cast"
    assert (output_type.shape.dim[1].dim_value == 2
            ), "Dimensions were not conserved by cast"

    # Test whether the new tensor has the right size
    assert (2 * len(
        modified_onnx_model.graph.node[0].attribute[0].t.int32_data) == len(
            onnx_model.graph.node[0].attribute[0].t.raw_data)
            ), "Wrong number of Bytes in casted version."

    # Retrieve the two constant tensors and compare the values
    assert np.allclose(
        modified_onnx_model.graph.node[0].attribute[0].t.int32_data,
        values.flatten()), "Data was not conserved by cast"


def test_type_cast_INT8ToINT32():

    # Build an onnx proto with a single constant node
    # Create output tensors of the type to cast
    X = helper.make_tensor_value_info('X', TensorProto.INT8, [3, 2])
    Y = helper.make_tensor_value_info('Y', TensorProto.INT8, [3, 2])

    values = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.uint8)
    node_def = onnx.helper.make_node('ConstantOfShape',
                                     inputs=['X'],
                                     outputs=['Y'],
                                     value=onnx.helper.make_tensor(
                                         name='const_tensor',
                                         data_type=onnx.TensorProto.INT8,
                                         dims=values.shape,
                                         vals=values.flatten().astype(
                                             np.int8).tobytes(),
                                         raw=True))

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [X],
        [Y],
    )

    # Create the model (ModelProto)
    onnx_model = helper.make_model(graph_def)

    # Make sure the opset version is version 9 (by default it would be 11
    # which would crash subsequent function calls)
    onnx_model = version_converter.convert_version(onnx_model, 9)

    # Compile the model to an onnx graph
    onnx.save_model(onnx_model, "type_test.onnx")

    # Load proto into a graph transfomer and apply cast
    graph_transformer = popart.GraphTransformer("type_test.onnx")
    graph_transformer.convertINT8ToINT32()

    # Retrieve modeified graph proto
    proto = graph_transformer.getModelProto()
    popart.Builder(proto).saveModelProto("type_test_modified.onnx")

    # Load the model as an onnx model again
    # modified_onnx_model = onnx.load(proto)
    modified_onnx_model = onnx.load("type_test_modified.onnx")

    # Make sure the graph is still good
    onnx.checker.check_model(modified_onnx_model)

    # Get only the first input of the input array (there should only be one)
    i = modified_onnx_model.graph.input[0]
    o = modified_onnx_model.graph.output[0]

    input_type = i.type.tensor_type
    output_type = o.type.tensor_type

    # Make sure shapes remain untouched
    assert (input_type.HasField("shape")
            ), "Modified graph output has no shape attribute"
    assert (output_type.HasField("shape")
            ), "Modified graph output has no shape attribute"
    assert (input_type.shape.dim[0].dim_value == 3
            ), "Dimensions were not conserved by cast"
    assert (input_type.shape.dim[1].dim_value == 2
            ), "Dimensions were not conserved by cast"
    assert (output_type.shape.dim[0].dim_value == 3
            ), "Dimensions were not conserved by cast"
    assert (output_type.shape.dim[1].dim_value == 2
            ), "Dimensions were not conserved by cast"

    # Test whether the new tensor has the right size
    assert (len(
        modified_onnx_model.graph.node[0].attribute[0].t.int32_data) == len(
            onnx_model.graph.node[0].attribute[0].t.raw_data)
            ), "Wrong number of Bytes in casted version."

    # Retrieve the two constant tensors and compare the values
    assert np.allclose(
        modified_onnx_model.graph.node[0].attribute[0].t.int32_data,
        values.flatten()), "Data was not conserved by cast"


def test_type_cast_INT16ToINT32():

    # Build an onnx proto with a single constant node
    # Create output tensors of the type to cast
    X = helper.make_tensor_value_info('X', TensorProto.INT16, [3, 2])
    Y = helper.make_tensor_value_info('Y', TensorProto.INT16, [3, 2])

    values = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.int16)
    node_def = onnx.helper.make_node('ConstantOfShape',
                                     inputs=['X'],
                                     outputs=['Y'],
                                     value=onnx.helper.make_tensor(
                                         name='const_tensor',
                                         data_type=onnx.TensorProto.INT16,
                                         dims=values.shape,
                                         vals=values.flatten().astype(
                                             np.int16).tobytes(),
                                         raw=True))

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [X],
        [Y],
    )

    # Create the model (ModelProto)
    onnx_model = helper.make_model(graph_def)

    # Make sure the opset version is version 9 (by default it would be 11
    # which would crash subsequent function calls)
    onnx_model = version_converter.convert_version(onnx_model, 9)

    # Compile the model to an onnx graph
    onnx.save_model(onnx_model, "type_test.onnx")

    # Load proto into a graph transfomer and apply cast
    graph_transformer = popart.GraphTransformer("type_test.onnx")
    graph_transformer.convertINT16ToINT32()

    # Retrieve modeified graph proto
    proto = graph_transformer.getModelProto()
    popart.Builder(proto).saveModelProto("type_test_modified.onnx")

    # Load the model as an onnx model again
    # modified_onnx_model = onnx.load(proto)
    modified_onnx_model = onnx.load("type_test_modified.onnx")

    # Make sure the graph is still good
    onnx.checker.check_model(modified_onnx_model)

    # Get only the first input of the input array (there should only be one)
    i = modified_onnx_model.graph.input[0]
    o = modified_onnx_model.graph.output[0]

    input_type = i.type.tensor_type
    output_type = o.type.tensor_type

    # Make sure shapes remain untouched
    assert (input_type.HasField("shape")
            ), "Modified graph output has no shape attribute"
    assert (output_type.HasField("shape")
            ), "Modified graph output has no shape attribute"
    assert (input_type.shape.dim[0].dim_value == 3
            ), "Dimensions were not conserved by cast"
    assert (input_type.shape.dim[1].dim_value == 2
            ), "Dimensions were not conserved by cast"
    assert (output_type.shape.dim[0].dim_value == 3
            ), "Dimensions were not conserved by cast"
    assert (output_type.shape.dim[1].dim_value == 2
            ), "Dimensions were not conserved by cast"

    # Test whether the new tensor has the right size
    assert (2 * len(
        modified_onnx_model.graph.node[0].attribute[0].t.int32_data) == len(
            onnx_model.graph.node[0].attribute[0].t.raw_data)
            ), "Wrong number of Bytes in casted version."

    # Retrieve the two constant tensors and compare the values
    assert np.allclose(
        modified_onnx_model.graph.node[0].attribute[0].t.int32_data,
        values.flatten()), "Data was not conserved by cast"


def test_type_cast_INT64ToINT32():

    # Build an onnx proto with a single constant node
    # Create output tensors of the type to cast
    X = helper.make_tensor_value_info('X', TensorProto.INT64, [3, 2])
    Y = helper.make_tensor_value_info('Y', TensorProto.INT64, [3, 2])

    values = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.int64)
    node_def = onnx.helper.make_node(
        'ConstantOfShape',
        inputs=['X'],
        outputs=['Y'],
        value=onnx.helper.make_tensor(name='const_tensor',
                                      data_type=onnx.TensorProto.INT64,
                                      dims=values.shape,
                                      vals=values.flatten().astype(np.int64),
                                      raw=False))

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [X],
        [Y],
    )

    # Create the model (ModelProto)
    onnx_model = helper.make_model(graph_def)

    # Make sure the opset version is version 9 (by default it would be 11
    # which would crash subsequent function calls)
    onnx_model = version_converter.convert_version(onnx_model, 9)

    # Compile the model to an onnx graph
    onnx.save_model(onnx_model, "type_test.onnx")

    # Load proto into a graph transfomer and apply cast
    graph_transformer = popart.GraphTransformer("type_test.onnx")
    graph_transformer.convertINT64ToINT32()

    # Retrieve modeified graph proto
    proto = graph_transformer.getModelProto()
    popart.Builder(proto).saveModelProto("type_test_modified.onnx")

    # Load the model as an onnx model again
    # modified_onnx_model = onnx.load(proto)
    modified_onnx_model = onnx.load("type_test_modified.onnx")

    # Make sure the graph is still good
    onnx.checker.check_model(modified_onnx_model)

    # Get only the first input of the input array (there should only be one)
    i = modified_onnx_model.graph.input[0]
    o = modified_onnx_model.graph.output[0]

    input_type = i.type.tensor_type
    output_type = o.type.tensor_type

    # Make sure shapes remain untouched
    assert (input_type.HasField("shape")
            ), "Modified graph output has no shape attribute"
    assert (output_type.HasField("shape")
            ), "Modified graph output has no shape attribute"
    assert (input_type.shape.dim[0].dim_value == 3
            ), "Dimensions were not conserved by cast"
    assert (input_type.shape.dim[1].dim_value == 2
            ), "Dimensions were not conserved by cast"
    assert (output_type.shape.dim[0].dim_value == 3
            ), "Dimensions were not conserved by cast"
    assert (output_type.shape.dim[1].dim_value == 2
            ), "Dimensions were not conserved by cast"

    # Test whether the new tensor has the right amount of data in it
    assert (len(
        modified_onnx_model.graph.node[0].attribute[0].t.int32_data) == len(
            onnx_model.graph.node[0].attribute[0].t.int64_data)
            ), "Wrong number of Bytes in casted version."

    # Retrieve the two constant tensors and compare the values
    assert np.allclose(
        modified_onnx_model.graph.node[0].attribute[0].t.int32_data,
        values.flatten()), "Data was not conserved by cast"


def test_type_cast_DoubleToHalf():

    # Build an onnx proto with a single constant node
    # Create output tensors of the type to cast
    X = helper.make_tensor_value_info('X', TensorProto.DOUBLE, [3, 2])
    Y = helper.make_tensor_value_info('Y', TensorProto.DOUBLE, [3, 2])

    values = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]).astype(np.double)
    node_def = onnx.helper.make_node(
        'ConstantOfShape',
        inputs=['X'],
        outputs=['Y'],
        value=onnx.helper.make_tensor(name='const_tensor',
                                      data_type=onnx.TensorProto.DOUBLE,
                                      dims=values.shape,
                                      vals=values.flatten().astype(np.double),
                                      raw=False))

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [X],
        [Y],
    )

    # Create the model (ModelProto)
    onnx_model = helper.make_model(graph_def)

    # Make sure the opset version is version 9 (by default it would be 11
    # which would crash subsequent function calls)
    onnx_model = version_converter.convert_version(onnx_model, 9)

    # Compile the model to an onnx graph
    onnx.save_model(onnx_model, "type_test.onnx")

    # Load proto into a graph transfomer and apply cast
    graph_transformer = popart.GraphTransformer("type_test.onnx")
    graph_transformer.convertDoublesToHalfs()

    # Retrieve modeified graph proto
    proto = graph_transformer.getModelProto()
    popart.Builder(proto).saveModelProto("type_test_modified.onnx")

    # Load the model as an onnx model again
    # modified_onnx_model = onnx.load(proto)
    modified_onnx_model = onnx.load("type_test_modified.onnx")

    # Make sure the graph is still good
    onnx.checker.check_model(modified_onnx_model)

    # Get only the first input of the input array (there should only be one)
    i = modified_onnx_model.graph.input[0]
    o = modified_onnx_model.graph.output[0]

    input_type = i.type.tensor_type
    output_type = o.type.tensor_type

    # Make sure shapes remain untouched
    assert (input_type.HasField("shape")
            ), "Modified graph output has no shape attribute"
    assert (output_type.HasField("shape")
            ), "Modified graph output has no shape attribute"
    assert (input_type.shape.dim[0].dim_value == 3
            ), "Dimensions were not conserved by cast"
    assert (input_type.shape.dim[1].dim_value == 2
            ), "Dimensions were not conserved by cast"
    assert (output_type.shape.dim[0].dim_value == 3
            ), "Dimensions were not conserved by cast"
    assert (output_type.shape.dim[1].dim_value == 2
            ), "Dimensions were not conserved by cast"

    # Test whether the new tensor has the right size
    assert (len(
        modified_onnx_model.graph.node[0].attribute[0].t.raw_data) == 2 *
            len(onnx_model.graph.node[0].attribute[0].t.double_data)
            ), "Wrong number of Bytes in casted version."

    # Retrieve the two constant tensors and compare the values (note that even if numpy does some magic there,
    # normal cast from double to half will not be able to preserve correctly some values such as 0.3, hence the high rtol)
    assert np.allclose(np.frombuffer(
        modified_onnx_model.graph.node[0].attribute[0].t.raw_data,
        dtype=np.half),
                       onnx_model.graph.node[0].attribute[0].t.double_data,
                       rtol=1e-3), "Data was not conserved by cast"


def test_type_cast_DoubleToFloat():

    # Build an onnx proto with a single constant node
    # Create output tensors of the type to cast
    X = helper.make_tensor_value_info('X', TensorProto.DOUBLE, [3, 2])
    Y = helper.make_tensor_value_info('Y', TensorProto.DOUBLE, [3, 2])

    values = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]).astype(np.double)
    node_def = onnx.helper.make_node(
        'ConstantOfShape',
        inputs=['X'],
        outputs=['Y'],
        value=onnx.helper.make_tensor(name='const_tensor',
                                      data_type=onnx.TensorProto.DOUBLE,
                                      dims=values.shape,
                                      vals=values.flatten().astype(np.double),
                                      raw=False))

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [X],
        [Y],
    )

    # Create the model (ModelProto)
    onnx_model = helper.make_model(graph_def)

    # Make sure the opset version is version 9 (by default it would be 11
    # which would crash subsequent function calls)
    onnx_model = version_converter.convert_version(onnx_model, 9)

    # Compile the model to an onnx graph
    onnx.save_model(onnx_model, "type_test.onnx")

    # Load proto into a graph transfomer and apply cast
    graph_transformer = popart.GraphTransformer("type_test.onnx")
    graph_transformer.convertDoublesToFloats()

    # Retrieve modeified graph proto
    proto = graph_transformer.getModelProto()
    popart.Builder(proto).saveModelProto("type_test_modified.onnx")

    # Load the model as an onnx model again
    # modified_onnx_model = onnx.load(proto)
    modified_onnx_model = onnx.load("type_test_modified.onnx")

    # Make sure the graph is still good
    onnx.checker.check_model(modified_onnx_model)

    # Get only the first input of the input array (there should only be one)
    i = modified_onnx_model.graph.input[0]
    o = modified_onnx_model.graph.output[0]

    input_type = i.type.tensor_type
    output_type = o.type.tensor_type

    # Make sure shapes remain untouched
    assert (input_type.HasField("shape")
            ), "Modified graph output has no shape attribute"
    assert (output_type.HasField("shape")
            ), "Modified graph output has no shape attribute"
    assert (input_type.shape.dim[0].dim_value == 3
            ), "Dimensions were not conserved by cast"
    assert (input_type.shape.dim[1].dim_value == 2
            ), "Dimensions were not conserved by cast"
    assert (output_type.shape.dim[0].dim_value == 3
            ), "Dimensions were not conserved by cast"
    assert (output_type.shape.dim[1].dim_value == 2
            ), "Dimensions were not conserved by cast"

    # Test whether the new tensor has the right size
    assert (len(
        modified_onnx_model.graph.node[0].attribute[0].t.float_data) == len(
            onnx_model.graph.node[0].attribute[0].t.double_data)
            ), "Wrong number of Bytes in casted version."

    # Retrieve the two constant tensors and compare the values
    assert np.allclose(
        modified_onnx_model.graph.node[0].attribute[0].t.float_data,
        values.flatten()), "Data was not conserved by cast"


def test_type_cast_BFloatToFloat():

    # Build an onnx proto with a single constant node
    # Create output tensors of the type to cast
    X = helper.make_tensor_value_info('X', TensorProto.BFLOAT16, [3, 2])
    Y = helper.make_tensor_value_info('Y', TensorProto.BFLOAT16, [3, 2])

    # Define the target values as float32 and cast to bytes
    float_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).astype(np.float32)
    float_bytes = float_values.tobytes()

    # Reinterpret byte string as int16 values. That way we have split the floats
    # in 2 sets of 16bits
    int16_values = np.frombuffer(float_bytes, dtype=np.uint16)

    # Keep only the second 2 bytes of each float (for some reason it seems
    # that np.array.tobytes() puts the fractional bytes first), ie every other int16
    # and convert back to bytes. We should now have some bfloat which values
    # are close enough to the original floats (precision loss of around 5e-3)
    bfloat_as_int16 = int16_values[1::2]
    bfloat = bfloat_as_int16.tobytes()

    # This data is generated to check against to make sure that we actually get
    # the same "truncated" data with our method
    bfloat_values = np.frombuffer(bfloat, dtype=np.uint16)
    int16_from_bfloat = bfloat_values
    for i in range(6):
        int16_from_bfloat = np.insert(int16_from_bfloat, 5 - i, 0)

    float_again_bytes = np.array(int16_from_bfloat).tobytes()
    float_again = np.frombuffer(float_again_bytes, dtype=np.float32)

    node_def = onnx.helper.make_node('ConstantOfShape',
                                     inputs=['X'],
                                     outputs=['Y'],
                                     value=onnx.helper.make_tensor(
                                         name='const_tensor',
                                         data_type=onnx.TensorProto.BFLOAT16,
                                         dims=[3, 2],
                                         vals=bfloat,
                                         raw=True))

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [X],
        [Y],
    )

    # Create the model (ModelProto)
    onnx_model = helper.make_model(graph_def)

    # Make sure the opset version is version 9 (by default it would be 11
    # which would crash subsequent function calls)
    onnx_model = version_converter.convert_version(onnx_model, 9)

    # Compile the model to an onnx graph
    onnx.save_model(onnx_model, "type_test.onnx")

    # Load proto into a graph transfomer and apply cast
    graph_transformer = popart.GraphTransformer("type_test.onnx")
    graph_transformer.convertBFloats16ToFloat32()

    # Retrieve modeified graph proto
    proto = graph_transformer.getModelProto()
    popart.Builder(proto).saveModelProto("type_test_modified.onnx")

    # Load the model as an onnx model again
    # modified_onnx_model = onnx.load(proto)
    modified_onnx_model = onnx.load("type_test_modified.onnx")

    # Make sure the graph is still good
    onnx.checker.check_model(modified_onnx_model)

    # Get only the first input of the input array (there should only be one)
    i = modified_onnx_model.graph.input[0]
    o = modified_onnx_model.graph.output[0]

    input_type = i.type.tensor_type
    output_type = o.type.tensor_type

    # Make sure shapes remain untouched
    assert (input_type.HasField("shape")
            ), "Modified graph output has no shape attribute"
    assert (output_type.HasField("shape")
            ), "Modified graph output has no shape attribute"
    assert (input_type.shape.dim[0].dim_value == 3
            ), "Dimensions were not conserved by cast"
    assert (input_type.shape.dim[1].dim_value == 2
            ), "Dimensions were not conserved by cast"
    assert (output_type.shape.dim[0].dim_value == 3
            ), "Dimensions were not conserved by cast"
    assert (output_type.shape.dim[1].dim_value == 2
            ), "Dimensions were not conserved by cast"

    # Test whether the new tensor has the right size
    assert (len(
        modified_onnx_model.graph.node[0].attribute[0].t.float_data) == 6
            ), "Wrong number of Bytes in casted version."

    # Retrieve the two constant tensors and compare the values
    assert np.allclose(
        modified_onnx_model.graph.node[0].attribute[0].t.float_data,
        float_values,
        rtol=1e-2), "Data was not conserved by cast"
    assert np.allclose(
        modified_onnx_model.graph.node[0].attribute[0].t.float_data,
        float_again), "Data was not conserved by cast"


def test_type_cast_INT64ToINT32_clip():
    """
    The model:

                     starts,ends,axes (int64)
                             |
    t0 -----------.------- Slice -- t1 ------.
                  |                          |
    indices ---- Gather ----------- t2 ----- Add - o
     (int64)

    - Gather takes two inputs:
      - Constant int64 'indices'
      - 't0'. A data input
    - It cannot be evaluated on host by the const expression util, as it takes
      a variable input
    - The IPU does not support int64. Therefore we must convert the int64
      tensors of the onnx model to int32
    - But conversion is only possible if all int64 tensor data is within the
      range of int32, unless we clip the tensor data to int32's numeric limits
    - In this case the 'starts' tensor of the slice is valid according to the
      onnxspec, but out of range of int32.
    - But we know in this case it is safe to clip it, as we will still get the
      same result
    """
    d1 = np.array([[-1, -2, -3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
    d2 = np.array([0, 1]).astype(np.int64)
    axis = 0

    axesV = np.array([0], dtype=np.int64)
    # Out of range value for int32!
    startsV = np.array([-9223372036854775807], dtype=np.int64)
    endsV = np.array([2], dtype=np.int64)

    builder = popart.Builder()

    i1 = builder.addInputTensor("FLOAT", d1.shape)
    i2 = builder.addInputTensor("INT64", d2.shape)
    g = builder.aiOnnx.gather([i1, i2], axis)

    axes = builder.aiOnnx.constant(axesV)
    starts = builder.aiOnnx.constant(startsV)
    ends = builder.aiOnnx.constant(endsV)
    s = builder.aiOnnx.slice([i1, starts, ends, axes])

    o = builder.aiOnnx.add([g, s])

    int64_proto = builder.getModelProto()
    graph_transformer = popart.GraphTransformer(int64_proto)
    graph_transformer.convertINT64ToINT32(clip=True)
    int32_proto = graph_transformer.getModelProto()

    session = popart.InferenceSession(fnModel=int32_proto,
                                      dataFlow=popart.DataFlow(1, [o]),
                                      deviceInfo=tu.create_test_device())
    session.prepareDevice()

    anchors = session.initAnchorArrays()
    stepio = popart.PyStepIO({i1: d1, i2: d2}, anchors)
    session.run(stepio)

    reference = 2 * np.take(d1, d2, axis=axis)
    assert np.allclose(anchors[o], reference)
