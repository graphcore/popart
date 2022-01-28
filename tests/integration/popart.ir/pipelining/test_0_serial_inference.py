# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Testing of pipelining with two forward passes for a simple model without using pipelining."""

from typing import Tuple, Dict
import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
from popart.ir.streams import HostToDeviceStream, DeviceToHostStream
import popart

from .pipeline_test_runner import PipelineTestRunner


def serial_inference_model(ir: pir.Ir, stream_input_shape: Tuple[int],
                           weights_and_biases: Dict[str, np.array]
                           ) -> Tuple[HostToDeviceStream, DeviceToHostStream]:
    """Build the serial inference model using popart.ir API.

    The model consist of:
    - An input layer
    - A hidden layer
    - Gelu activation
    - A output layer
    - Softmax activation

    Args:
        ir (pir.Ir): The intermediate representation to create the model in.
        stream_input_shape (tuple): The shape of the streamed input tensor.
          NOTE: The batch size must be hardcoded in the shape.
          This is for the streamed input data to match the numbers with the number of host_loads
        weights_and_biases (Dict[str, np.array]): dict containing:

            hidden_weights (np.array): The (non-streamed) data for the weights of the hidden layer
            hidden_bias (np.array): The (non-streamed) data for the bias of the hidden layer
            output_weights (np.array): The (non-streamed) data for the weights of the output layer
            output_bias (np.array): The (non-streamed) data for the bias of the output layer

    Returns:
    (tuple): tuple containing:

        t_in_h2d (HostToDeviceStream): The input stream of t_in
        t_out_d2h (DeviceToHostStream): The output stream of t_out
    """
    main = ir.main_graph()
    with main:
        # These weights are streamed once to the device
        hidden_weights = pir.variable(weights_and_biases["hidden_weights"],
                                      name="hidden_weights")
        hidden_bias = pir.variable(weights_and_biases["hidden_bias"],
                                   name="hidden_bias")
        output_weights = pir.variable(weights_and_biases["output_weights"],
                                      name="output_weights")
        output_bias = pir.variable(weights_and_biases["output_bias"],
                                   name="output_bias")

        # Create the streams for loading and storing to the host
        t_in_h2d = pir.h2d_stream(stream_input_shape,
                                  pir.float32,
                                  name="t_in_stream")
        t_out_d2h = pir.d2h_stream(output_bias.shape,
                                   pir.float32,
                                   name="t_out_stream")

        # Operations on IPU 0 (the hidden layer)
        with pir.ipu(ipu=0):
            t_in = ops.host_load(t_in_h2d, "t_in")
            t_hidden_matmul = ops.matmul(t_in, hidden_weights)
            t_hidden_layer_result = ops.add(t_hidden_matmul, hidden_bias)
            # Copy to IPU 1
            t_hidden_layer_result_copy = ops.ipu_copy(t=t_hidden_layer_result,
                                                      destination=1)

        # Operations on IPU 1 (the hidden layer activation)
        with pir.ipu(ipu=1):
            t_hidden_layer_activation = ops.gelu(t_hidden_layer_result_copy)
            # Copy to IPU 2
            t_hidden_layer_activation_copy = ops.ipu_copy(
                t=t_hidden_layer_activation, destination=2)

        # Operations on IPU 2 (the output layer)
        with pir.ipu(ipu=2):
            t_output_matmul = ops.matmul(t_hidden_layer_activation_copy,
                                         output_weights)
            t_output_layer_result = ops.add(t_output_matmul, output_bias)
            # Copy to IPU 3
            t_output_layer_result_copy = ops.ipu_copy(t=t_output_layer_result,
                                                      destination=3)

        # Operations on IPU 3 (the output layer activation)
        with pir.ipu(ipu=3):
            # We've ordered the host_load data after
            # (compute_batches, n_input)
            # This corresponds to axis index 1
            t_output_activation = ops.softmax(t_output_layer_result_copy,
                                              axis=1)
            ops.host_store(t_out_d2h, t_output_activation)

    return t_in_h2d, t_out_d2h


def test_serial_inference():
    """
    Test the serial inference model (i.e. the model without pipelining).

    The model consist of:
    - An input layer
    - A hidden layer
    - Gelu activation
    - A output layer
    - Softmax activation

    The test will compare the result with the result from `run_inference_torch_reference_model`
    implemented in torch.

    This is one in a series of test which demonstrates the usage of pipelining.
    """
    # The model only has one host load, hence the micro batch will be 1
    pipeline_test_runner = PipelineTestRunner(model=serial_inference_model,
                                              n_step_io_calls=10,
                                              micro_batch_size=1)
    pipeline_test_runner.set_options()
    weights_and_biases = pipeline_test_runner.get_weights_and_biases()
    session, t_in_id, t_out_id = pipeline_test_runner.build_popart_ir_model(
        weights_and_biases)

    # Create buffers for anchors
    anchors = session.initAnchorArrays()
    # Initialize the result and full_dataset_data
    full_dataset_data = np.empty(
        (pipeline_test_runner.batches["n_step_io_calls"],
         pipeline_test_runner.batches["micro_batch_size"],
         *pipeline_test_runner.stream_input_shape)).astype(np.float32)
    result = np.empty(
        (pipeline_test_runner.batches["n_step_io_calls"],
         pipeline_test_runner.batches["micro_batch_size"],
         pipeline_test_runner.batches["compute_batches"],
         pipeline_test_runner.nn_dims["n_outputs"])).astype(np.float32)

    # Copy the weights and biases from the host
    session.weightsFromHost()

    # Run the model
    for run_nr in range(pipeline_test_runner.batches["n_step_io_calls"]):
        # Create the dataset data
        dataset_data = pipeline_test_runner.create_data(
            (pipeline_test_runner.batches["micro_batch_size"],
             *pipeline_test_runner.stream_input_shape))
        full_dataset_data[run_nr, ...] = dataset_data

        stepio = popart.PyStepIO({t_in_id: dataset_data}, anchors)
        session.run(stepio)

        # Append the result
        result[run_nr, ...] = anchors[t_out_id]

    # Compare outcome from popart.ir with outcome from pytorch
    pipeline_test_runner.compare_with_reference(result, full_dataset_data,
                                                weights_and_biases)
