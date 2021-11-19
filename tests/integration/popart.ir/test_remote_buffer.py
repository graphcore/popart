# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Testing of pipelining with one forward pass for a simple model without pipeline stages."""

from popart.ir.tensor import Variable
from typing import Callable, Tuple, Dict, Optional
from popart.ir.streams import DeviceToHostStream
import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
from popart.ir.remote_buffer_handle import RemoteBufferHandle
from popart.ir.dtypes import dtype, float32
import popart._internal.ir as _ir
import popart

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def test_remote_buffer() -> None:
    """Test the interaction with the remote buffer through popart.ir.
    """
    # Prepare the input and output data
    shape_1 = (1, 3, 5)
    shape_2 = (7, 11)
    d_type_1 = np.dtype("float32")
    d_type_2 = np.dtype("float16")

    data = {}

    # Store and load data for the first tensor
    data["store_in_1"] = np.random.rand(*shape_1).astype(d_type_1)
    data["load_in_1"] = np.zeros(shape_1).astype(d_type_1)
    data["load_in_1_inplace"] = np.zeros(shape_1).astype(d_type_1)
    # Store and load data for the second tensor
    data["store_in_2"] = np.random.rand(*shape_2).astype(d_type_2)
    data["load_in_2"] = np.zeros(shape_2).astype(d_type_2)
    # Store and load data for the third tensor
    data["store_in_3"] = np.random.rand(*shape_2).astype(d_type_2)
    data["load_in_3_inplace"] = np.zeros(shape_2).astype(d_type_2)

    ir, d2h_streams = build_model(data)

    # Get the tensor_ids
    labels = (
        "load_in_1",
        "load_in_1_inplace",
        "load_out_1",
        "load_out_1_inplace",
        "load_in_2",
        "load_in_3_inplace",
        "load_out_2",
        "load_out_3_inplace",
    )
    t_ids = {label: d2h_streams[label].tensor_id() for label in labels}

    # Setup the anchors
    anchor_tensors = {
        t_ids[stream]: popart.AnchorReturnType("All")
        for stream in t_ids.keys()
    }

    # Set the data flow
    bps = 1
    data_flow = popart.DataFlow(bps, anchor_tensors)
    ir.setDataFlow(data_flow)

    # Set options
    opts = ir.getSessionOptions()
    opts.useHostCopyOps = True

    # Prepare graph
    ir.updateVertices()
    ir.setIsPrepared()

    # Create an IR inference session
    session = popart.InferenceSession.fromIr(
        ir=ir, deviceInfo=tu.create_test_device(numIpus=1))

    session.prepareDevice()

    # Create buffers for anchors
    anchors = session.initAnchorArrays()

    # Run the model
    step_io = popart.PyStepIO({}, anchors)
    session.weightsFromHost()
    session.run(step_io)

    # Assert that the tensors are correct
    remote_load_scenarios = (
        "1",
        "1_inplace",
        "2",
        "3_inplace",
    )
    for scenario in remote_load_scenarios:
        print(f"Now asserting remote load scenario {scenario}")
        # Get data to assert
        store_in_data = data[f"store_in_{scenario.replace('_inplace', '')}"]
        load_in_data_before_op_call = data[f"load_in_{scenario}"]
        load_in_data_after_op_call = anchors[t_ids[f"load_in_{scenario}"]]
        load_out_data = anchors[t_ids[f"load_out_{scenario}"]]
        shape = shape_1 if "1" in scenario else shape_2
        d_type = d_type_1 if "1" in scenario else d_type_2
        inplace = True if "inplace" in scenario else False
        # Assert shape and type
        assert load_in_data_after_op_call.shape == shape
        assert load_in_data_after_op_call.dtype == d_type
        assert load_out_data.shape == shape
        assert load_out_data.dtype == d_type

        # Assert that the data has been loaded
        assert np.allclose(store_in_data, load_out_data)
        if inplace:
            # Assert that the load in data has been overwritten
            assert np.allclose(load_in_data_after_op_call, store_in_data)
        else:
            # Assert that the load in data has not been overwritten
            assert np.allclose(load_in_data_after_op_call,
                               load_in_data_before_op_call)


def build_model(data: Dict[str, np.array]
                ) -> Tuple[_ir.Ir, Dict[str, DeviceToHostStream]]:
    """Build a model for storing and loading tensors from the remote buffer.

    Args:
        data(Dict[str, np.array]) : Dict of the data to be stored and loaded from the remote buffer

    Returns:
    (tuple): tuple containing:

        ir._pb_ir (_ir.Ir): The underlying IR
        d2h_streams (Dict[str, DeviceToHostStream]): The output streams
    """
    ir = pir.Ir()
    main = ir.main_graph()

    with main:
        # Placeholder for tensor ids
        tensors = {}
        # Create variable tensors from the data
        for name in data.keys():
            tensors[name] = pir.variable(data[name], name=name)

        # Placeholder for device to host streams
        d2h_streams = {}

        # Store and load the first tensor without specifying the remote buffer handle or offset
        ops.remote_store(t=tensors["store_in_1"])
        tensors["load_out_1"] = ops.remote_load(t=tensors["load_in_1"])
        tensors["load_out_1_inplace"] = ops.remote_load_(
            t=tensors["load_in_1_inplace"])
        # Anchor the input tensors to the load operator
        d2h_streams = make_anchor(d2h_streams, tensors, "load_in_1")
        d2h_streams = make_anchor(d2h_streams, tensors, "load_in_1_inplace")
        # Anchor the output tensors of the load operator
        d2h_streams = make_anchor(d2h_streams, tensors, "load_out_1")
        d2h_streams = make_anchor(d2h_streams, tensors, "load_out_1_inplace")

        # Store and load the second and third tensor using the same buffer id
        # Buffer 1 should already be assigned implicitly, so we chose a different id
        rbh = RemoteBufferHandle(
            remote_buffer_id=42,
            tensor_shape=tensors["store_in_2"]._pb_tensor.info.shape(),
            tensor_dtype=dtype.as_dtype(
                tensors["store_in_2"]._pb_tensor.info.data_type_lcase()),
            repeats=2)
        # Index starts at 0
        offset_tensor_2 = pir.constant(0, name="offset_2")
        offset_tensor_3 = pir.constant(1, name="offset_3")
        ops.remote_store(t=tensors["store_in_2"],
                         offset=offset_tensor_2,
                         remote_buffer_handle=rbh)
        ops.remote_store(t=tensors["store_in_3"],
                         offset=offset_tensor_3,
                         remote_buffer_handle=rbh)
        tensors["load_out_2"] = ops.remote_load(t=tensors["load_in_2"],
                                                offset=offset_tensor_2,
                                                remote_buffer_handle=rbh)
        tensors["load_out_3_inplace"] = ops.remote_load_(
            t=tensors["load_in_3_inplace"],
            offset=offset_tensor_3,
            remote_buffer_handle=rbh)

        # Anchor the input tensors to the load operator
        d2h_streams = make_anchor(d2h_streams, tensors, "load_in_2")
        d2h_streams = make_anchor(d2h_streams, tensors, "load_in_3_inplace")
        # Anchor the output tensors of the load operator
        d2h_streams = make_anchor(d2h_streams, tensors, "load_out_2")
        d2h_streams = make_anchor(d2h_streams, tensors, "load_out_3_inplace")

    return ir._pb_ir, d2h_streams


def make_anchor(d2h_streams: Dict[str, str], tensors: Dict[str, Variable],
                label: str) -> Dict[str, str]:
    """Insert device to host anchors.

    Args:
        d2h_streams (Dict[str, str]): Dict mapping DeviceToHostStream ids to label
        tensors (Dict[str, Variable]): Dict mapping label to a Variable
        label (str): The label in tensors to insert device to host anchor for

    Returns:
        Dict[str, str]: Updated dictionary mapping DeviceToHostStream ids to label
    """
    d2h_streams[label] = pir.d2h_stream(
        tensors[label]._pb_tensor.info.shape(),
        dtype.as_dtype(tensors[label]._pb_tensor.info.data_type_lcase()),
        name=f"{label}_stream")
    ops.host_store(d2h_streams[label], tensors[label])
    return d2h_streams
