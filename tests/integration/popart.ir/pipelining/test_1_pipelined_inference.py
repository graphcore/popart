# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""Testing of pipelining with two forward pass for a simple model using pipelining."""

from typing import Tuple, Dict
import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
from popart.ir.streams import HostToDeviceStream, DeviceToHostStream
from popart.ir.tensor import Tensor, Variable

from .pipeline_test_runner import PipelineTestRunner


def pipelined_inference_model(
        ir: pir.Ir, stream_input_shape: Tuple[int],
        weights_and_biases: Dict[str, np.array],
        repeat_count: int) -> Tuple[HostToDeviceStream, DeviceToHostStream]:
    """Build the (grouped) pipelined inference model using popart.ir API.

    The model consist of:
    - An input layer
    - A hidden layer
    - Gelu activation
    - A output layer
    - Softmax activation

    Pipelined the model will look like this

    Legend:
    MMA - Matmul and add (one for the hidden layer and one for the output layer)
    G   - Gelu
    S   - Softmax
    xx5 - Operator xx operating on micro batch number 5
    |   - Separation between two pipeline cycles (where copying happens)

    Pipeline Cycle -->
         <--- fill phase ---> <--- main --> <-- flush phase --->
    IPU0: MMA0 | MMA1 | MMA2 | MMA3 | MMA4 |      |       |    |
    IPU1:      | G0   | G1   | G2   | G3   | G4   |       |    |
    IPU2:      |      | MMA0 | MMA1 | MMA2 | MMA3 | MMA4  |    |
    IPU3:      |      |      | S0   | S1   | S2   | S3    | S4 |

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

        repeat_count (int): Number of times to repeat the main phase of the pipeline

    Returns:
    (tuple): tuple containing:

        t_in_h2d (HostToDeviceStream): The input stream of t_in
        t_out_d2h (DeviceToHostStream): The output stream of t_out
    """
    main = ir.main_graph()
    with main:
        # These weights are streamed once to the device
        tensor_variable = {}
        tensor_variable["hidden_weights"] = pir.variable(
            weights_and_biases["hidden_weights"], name="hidden_weights")
        tensor_variable["hidden_bias"] = pir.variable(
            weights_and_biases["hidden_bias"], name="hidden_bias")
        tensor_variable["output_weights"] = pir.variable(
            weights_and_biases["output_weights"], name="output_weights")
        tensor_variable["output_bias"] = pir.variable(
            weights_and_biases["output_bias"], name="output_bias")

        # Create the streams for loading and storing to the host
        t_in_h2d = pir.h2d_stream(stream_input_shape,
                                  pir.float32,
                                  name="t_in_stream")
        t_out_d2h = pir.d2h_stream(tensor_variable["output_bias"].shape,
                                   pir.float32,
                                   name="t_out_stream")

        # We use pir.in_sequence in order to prevent the scheduling to optimise for minimal
        # tensor liveness (and thereby change the serialisation of the graph)
        with pir.in_sequence(True):
            micro_batch_tensor = ramp_up(t_in_h2d, tensor_variable)
            micro_batch_tensor = main_phase(repeat_count, t_in_h2d, t_out_d2h,
                                            tensor_variable,
                                            micro_batch_tensor)

            # This is completing the last micro batch, so no need to return micro_batch_tensor
            ramp_down(t_out_d2h, tensor_variable, micro_batch_tensor)

    return t_in_h2d, t_out_d2h


def ramp_up(t_in_h2d: HostToDeviceStream, tensor_variable: Dict[str, Variable]
            ) -> Dict[int, Dict[str, Tensor]]:
    """Write the ramp up phase to the IR.

    The ramp up is also known as the "fill phase".
    In this example we have 4 pipeline stages (parts of the graph which can run in parallel).
    Hence we need 3 cycles for filling the pipeline
    (see ASCII illustration in pipelined_inference_model for details).

    Args:
        t_in_h2d (HostToDeviceStream): The host to device stream
        tensor_variable (Dict[str, Variable]): Dictionary containing the weights and biases

    Returns:
        Dict[int, Dict[str, Tensor]]: The map between the micro batch and the popart.ir tensors
    """
    # This dict will map the micro batch to the desired tensors
    micro_batch_tensor = {}

    micro_batch_tensor = pipeline_cycle_0(t_in_h2d, tensor_variable,
                                          micro_batch_tensor)
    micro_batch_tensor = pipeline_cycle_1(t_in_h2d, tensor_variable,
                                          micro_batch_tensor)
    micro_batch_tensor = pipeline_cycle_2(t_in_h2d, tensor_variable,
                                          micro_batch_tensor)

    return micro_batch_tensor


def main_phase(repeat_count: int, t_in_h2d: HostToDeviceStream,
               t_out_d2h: DeviceToHostStream,
               tensor_variable: Dict[str, Variable],
               micro_batch_tensor: Dict[int, Dict[str, Tensor]]
               ) -> Dict[int, Dict[str, Tensor]]:
    """Write the main phase to the IR.

    Note: We are here unrolling the loop (meaning that all the iterations are written explicitly to
    the IR). Whereas this sometimes can make the code more readable it can also increase the
    memory requirements significantly.
    An alternative to this approach is to use an operator which works as a loop, for example the
    repeat op

    Args:
        repeat_count (int): Number of times the main phase should be repeated
        t_in_h2d (HostToDeviceStream): The host to device stream
        t_out_d2h (DeviceToHostStream): The device to host stream
        tensor_variable (Dict[str, Variable]): Dictionary containing the weights and biases
        micro_batch_tensor (Dict[int, Dict[str, Tensor]]): The map between the micro batch and the
          popart.ir tensors

    Returns:
        Dict[int, Dict[str, Tensor]]: The map between the micro batch and the popart.ir tensors
    """

    for iteration_number in range(repeat_count):
        micro_batch_tensor = pipeline_main_cycle(iteration_number, t_in_h2d,
                                                 t_out_d2h, tensor_variable,
                                                 micro_batch_tensor)

    return micro_batch_tensor


def ramp_down(t_out_d2h: DeviceToHostStream,
              tensor_variable: Dict[str, Variable],
              micro_batch_tensor: Dict[int, Dict[str, Tensor]]):
    """Write the ramp down phase to the IR.

    The ramp down is also known as the "flush phase".
    In this example we have 4 pipeline stages (parts of the graph which can run in parallel).
    Hence we need 3 cycles for flushing the pipeline
    (see ASCII illustration in pipelined_inference_model for details).

    Args:
        t_in_h2d (HostToDeviceStream): The host to device stream
        tensor_variable (Dict[str, Variable]): Dictionary containing the weights and biases
        micro_batch_tensor (Dict[int, Dict[str, Tensor]]): The map between the micro batch and the
          popart.ir tensors
    """
    micro_batch_tensor = pipeline_cycle_n_minus_2(t_out_d2h, tensor_variable,
                                                  micro_batch_tensor)
    micro_batch_tensor = pipeline_cycle_n_minus_1(t_out_d2h, tensor_variable,
                                                  micro_batch_tensor)
    # This is completing the last micro batch, so no need to return micro_batch_tensor
    pipeline_cycle_n(t_out_d2h, micro_batch_tensor)


def pipeline_cycle_0(t_in_h2d: HostToDeviceStream,
                     tensor_variable: Dict[str, Variable],
                     micro_batch_tensor: Dict[int, Dict[str, Tensor]]
                     ) -> Dict[int, Dict[str, Tensor]]:
    """Write the 0th pipeline cycle to the IR.

    Args:
        t_in_h2d (HostToDeviceStream): The host to device stream
        tensor_variable (Dict[str, Variable]): Dictionary containing the weights and biases
        micro_batch_tensor (Dict[int, Dict[str, Tensor]]): The map between the micro batch and the
          popart.ir tensors

    Returns:
        Dict[int, Dict[str, Tensor]]: The map between the micro batch and the popart.ir tensors
    """
    micro_batch_tensor[0] = {}

    # Host load and compute part of the cycle
    with pir.ipu(ipu=0):
        # NOTE: host_load, host_store and ipu_copy act as communication barriers
        #       Thus we need to place everything we would like to happen in parallel between these
        #       barriers
        t_in = ops.host_load(t_in_h2d, "t_in")
        t_hidden_matmul = ops.matmul(t_in, tensor_variable["hidden_weights"])
        t_hidden_layer_result = ops.add(t_hidden_matmul,
                                        tensor_variable["hidden_bias"])

    # Copy part of the cycle
    with pir.ipu(ipu=0):
        # Copy to IPU 1
        micro_batch_tensor[0]["hidden_layer"] = ops.ipu_copy(
            t=t_hidden_layer_result, destination=1)

    return micro_batch_tensor


def pipeline_cycle_1(t_in_h2d: HostToDeviceStream,
                     tensor_variable: Dict[str, Variable],
                     micro_batch_tensor: Dict[int, Dict[str, Tensor]]
                     ) -> Dict[int, Dict[str, Tensor]]:
    """Write the 1st pipeline cycle to the IR.

    Args:
        t_in_h2d (HostToDeviceStream): The host to device stream
        tensor_variable (Dict[str, Variable]): Dictionary containing the weights and biases
        micro_batch_tensor (Dict[int, Dict[str, Tensor]]): The map between the micro batch and the
          popart.ir tensors

    Returns:
        Dict[int, Dict[str, Tensor]]: The map between the micro batch and the popart.ir tensors
    """
    micro_batch_tensor[1] = {}

    # Host load and compute part of the cycle
    with pir.ipu(ipu=0):
        # NOTE: This is the second host load in our graph
        #       When we feed data with PyStepIo we need to ensure that the first dimension of the
        #       data matches (the micro batches) matches the number of host_loads
        t_in = ops.host_load(t_in_h2d, "t_in")
        t_hidden_matmul = ops.matmul(t_in, tensor_variable["hidden_weights"])
        t_hidden_layer_result = ops.add(t_hidden_matmul,
                                        tensor_variable["hidden_bias"])

    with pir.ipu(ipu=1):
        # The activation uses the 0th micro batch passed from pipeline cycle 0
        t_hidden_layer_activation = ops.gelu(
            micro_batch_tensor[0]["hidden_layer"])

    # Copy part of the cycle
    # NOTE: We ensure that the copy operations are done last in order not to create communication
    #       barriers
    with pir.ipu(ipu=0):
        # Copy to IPU 1
        # This belong to micro batch 1 as it is the second time we host loaded data to the matmul
        # and add operations
        micro_batch_tensor[1]["hidden_layer"] = ops.ipu_copy(
            t=t_hidden_layer_result, destination=1)
    with pir.ipu(ipu=1):
        # Copy to IPU 2
        # The activation was working on micro batch 0, hence the result of the activation also
        # belongs to micro batch 0
        micro_batch_tensor[0]["hidden_layer_activation"] = ops.ipu_copy(
            t=t_hidden_layer_activation, destination=2)

    return micro_batch_tensor


def pipeline_cycle_2(t_in_h2d: HostToDeviceStream,
                     tensor_variable: Dict[str, Variable],
                     micro_batch_tensor: Dict[int, Dict[str, Tensor]]
                     ) -> Dict[int, Dict[str, Tensor]]:
    """Write the 2nd pipeline cycle to the IR.

    Args:
        t_in_h2d (HostToDeviceStream): The host to device stream
        tensor_variable (Dict[str, Variable]): Dictionary containing the weights and biases
        micro_batch_tensor (Dict[int, Dict[str, Tensor]]): The map between the micro batch and the
          popart.ir tensors

    Returns:
        Dict[int, Dict[str, Tensor]]: The map between the micro batch and the popart.ir tensors
    """
    micro_batch_tensor[2] = {}

    # Host load and compute part of the cycle
    with pir.ipu(ipu=0):
        # This is a new host load, meaning that IPU 0 will now work on micro batch 2
        t_in = ops.host_load(t_in_h2d, "t_in")
        t_hidden_matmul = ops.matmul(t_in, tensor_variable["hidden_weights"])
        t_hidden_layer_result = ops.add(t_hidden_matmul,
                                        tensor_variable["hidden_bias"])

    with pir.ipu(ipu=1):
        # IPU 1 get passed micro batch 1 from IPU 0
        t_hidden_layer_activation = ops.gelu(
            micro_batch_tensor[1]["hidden_layer"])

    with pir.ipu(ipu=2):
        # IPU 2 get passed micro batch 0 from IPU 1
        t_output_matmul = ops.matmul(
            micro_batch_tensor[0]["hidden_layer_activation"],
            tensor_variable["output_weights"])
        t_output_layer_result = ops.add(t_output_matmul,
                                        tensor_variable["output_bias"])

    # Copy part of the cycle
    with pir.ipu(ipu=0):
        # Copy to IPU 1
        micro_batch_tensor[2]["hidden_layer"] = ops.ipu_copy(
            t=t_hidden_layer_result, destination=1)
    with pir.ipu(ipu=1):
        # Copy to IPU 2
        micro_batch_tensor[1]["hidden_layer_activation"] = ops.ipu_copy(
            t=t_hidden_layer_activation, destination=2)
    with pir.ipu(ipu=2):
        # Copy to IPU 3
        micro_batch_tensor[0]["output_layer"] = ops.ipu_copy(
            t=t_output_layer_result, destination=3)

    return micro_batch_tensor


def pipeline_main_cycle(iteration_number: int, t_in_h2d: HostToDeviceStream,
                        t_out_d2h: DeviceToHostStream,
                        tensor_variable: Dict[str, Variable],
                        micro_batch_tensor: Dict[int, Dict[str, Tensor]]
                        ) -> Dict[int, Dict[str, Tensor]]:
    """Write a main pipeline cycle to the IR.

    Args:
        iteration_number (int): The number of the current iteration
        t_in_h2d (HostToDeviceStream): The host to device stream
        t_out_d2h (DeviceToHostStream): The device to host stream
        tensor_variable (Dict[str, Variable]): Dictionary containing the weights and biases
        micro_batch_tensor (Dict[int, Dict[str, Tensor]]): The map between the micro batch and the
          popart.ir tensors

    Returns:
        Dict[int, Dict[str, Tensor]]: The map between the micro batch and the popart.ir tensors
    """
    # The highest micro batch number is based on the iteration number
    # If we start from the end of our model:
    # - The activation of the output layer will work on the micro batch number which equals the
    #   iteration number
    # - The output layer will work on the micro batch which got introduced after the micro batch
    #   described above (+1)
    # - The activation of the hidden layer will work on the micro batch which got introduced
    #   after the micro batch described above (+1)
    # - The hidden layer will work on the micro batch which got introduced after the micro batch
    #   described above (+1)
    # Summing up we find that the highest iteration number worked on in this cycle is
    # iteration_number + 3
    highest_micro_batch = iteration_number + 3

    micro_batch_tensor[highest_micro_batch] = {}

    # Host load and compute part of the cycle
    with pir.ipu(ipu=0):
        # This will operate on the highest micro batch number
        t_in = ops.host_load(t_in_h2d, "t_in")
        t_hidden_matmul = ops.matmul(t_in, tensor_variable["hidden_weights"])
        t_hidden_layer_result = ops.add(t_hidden_matmul,
                                        tensor_variable["hidden_bias"])

    with pir.ipu(ipu=1):
        # This will work on the data from the highest micro batch number from the previous
        # iteration. That is the highest_micro_batch - 1
        t_hidden_layer_activation = ops.gelu(
            micro_batch_tensor[highest_micro_batch - 1]["hidden_layer"])

    with pir.ipu(ipu=2):
        # This will work on the data from the second highest micro batch number from the previous
        # iteration. That is the highest_micro_batch - 2
        t_output_matmul = ops.matmul(
            micro_batch_tensor[highest_micro_batch -
                               2]["hidden_layer_activation"],
            tensor_variable["output_weights"])
        t_output_layer_result = ops.add(t_output_matmul,
                                        tensor_variable["output_bias"])

    with pir.ipu(ipu=3):
        # This will work on the data from the third highest micro batch number from the previous
        # iteration. That is the highest_micro_batch - 3 = iteration_number
        t_output_activation = ops.softmax(
            micro_batch_tensor[iteration_number]["output_layer"], axis=1)

    # Copy part of the cycle
    with pir.ipu(ipu=0):
        # Copy to IPU 1
        micro_batch_tensor[highest_micro_batch]["hidden_layer"] = ops.ipu_copy(
            t=t_hidden_layer_result, destination=1)
    with pir.ipu(ipu=1):
        # Copy to IPU 2
        micro_batch_tensor[highest_micro_batch -
                           1]["hidden_layer_activation"] = ops.ipu_copy(
                               t=t_hidden_layer_activation, destination=2)
    with pir.ipu(ipu=2):
        # Copy to IPU 3
        micro_batch_tensor[highest_micro_batch -
                           2]["output_layer"] = ops.ipu_copy(
                               t=t_output_layer_result, destination=3)

    # Host store part of the cycle
    with pir.ipu(ipu=2):
        ops.host_store(t_out_d2h, t_output_activation)

    return micro_batch_tensor


def pipeline_cycle_n_minus_2(t_out_d2h: DeviceToHostStream,
                             tensor_variable: Dict[str, Variable],
                             micro_batch_tensor: Dict[int, Dict[str, Tensor]]
                             ) -> Dict[int, Dict[str, Tensor]]:
    """Write the third last pipeline cycle to the IR.

    Args:
        t_out_d2h (DeviceToHostStream): The device to host stream
        tensor_variable (Dict[str, Variable]): Dictionary containing the weights and biases
        micro_batch_tensor (Dict[int, Dict[str, Tensor]]): The map between the micro batch and the
          popart.ir tensors

    Returns:
        Dict[int, Dict[str, Tensor]]: The map between the micro batch and the popart.ir tensors
    """
    # NOTE: IPU 0 is done with all the work in this cycle

    highest_micro_batch = max(micro_batch_tensor.keys())

    # Compute part of the cycle
    with pir.ipu(ipu=1):
        # This will work on the data from the highest micro batch number from the previous
        # iteration.
        # As we are no longer doing any host load in this phase this equals to the
        # highest_micro_batch
        t_hidden_layer_activation = ops.gelu(
            micro_batch_tensor[highest_micro_batch]["hidden_layer"])

    with pir.ipu(ipu=2):
        # This will work on the data from the second highest micro batch number from the previous
        # iteration. That is the highest_micro_batch - 1
        t_output_matmul = ops.matmul(
            micro_batch_tensor[highest_micro_batch -
                               1]["hidden_layer_activation"],
            tensor_variable["output_weights"])
        t_output_layer_result = ops.add(t_output_matmul,
                                        tensor_variable["output_bias"])

    with pir.ipu(ipu=3):
        # This will work on the data from the third highest micro batch number from the previous
        # iteration. That is the highest_micro_batch - 2
        t_output_activation = ops.softmax(
            micro_batch_tensor[highest_micro_batch - 2]["output_layer"],
            axis=1)

    # Copy part of the cycle
    with pir.ipu(ipu=1):
        # Copy to IPU 2
        micro_batch_tensor[highest_micro_batch][
            "hidden_layer_activation"] = ops.ipu_copy(
                t=t_hidden_layer_activation, destination=2)
    with pir.ipu(ipu=2):
        # Copy to IPU 3
        micro_batch_tensor[highest_micro_batch -
                           1]["output_layer"] = ops.ipu_copy(
                               t=t_output_layer_result, destination=3)

    # Host store part of the cycle
    with pir.ipu(ipu=2):
        ops.host_store(t_out_d2h, t_output_activation)

    return micro_batch_tensor


def pipeline_cycle_n_minus_1(t_out_d2h: DeviceToHostStream,
                             tensor_variable: Dict[str, Variable],
                             micro_batch_tensor: Dict[int, Dict[str, Tensor]]
                             ) -> Dict[int, Dict[str, Tensor]]:
    """Write the second last pipeline cycle to the IR.

    Args:
        t_out_d2h (DeviceToHostStream): The device to host stream
        tensor_variable (Dict[str, Variable]): Dictionary containing the weights and biases
        micro_batch_tensor (Dict[int, Dict[str, Tensor]]): The map between the micro batch and the
          popart.ir tensors

    Returns:
        Dict[int, Dict[str, Tensor]]: The map between the micro batch and the popart.ir tensors
    """
    # NOTE: IPU 1 is done with all the work in this cycle

    highest_micro_batch = max(micro_batch_tensor.keys())

    # Compute part of the cycle
    with pir.ipu(ipu=2):
        t_output_matmul = ops.matmul(
            micro_batch_tensor[highest_micro_batch]["hidden_layer_activation"],
            tensor_variable["output_weights"])
        t_output_layer_result = ops.add(t_output_matmul,
                                        tensor_variable["output_bias"])

    with pir.ipu(ipu=3):
        t_output_activation = ops.softmax(
            micro_batch_tensor[highest_micro_batch - 1]["output_layer"],
            axis=1)

    # Copy part of the cycle
    with pir.ipu(ipu=2):
        # Copy to IPU 3
        micro_batch_tensor[highest_micro_batch]["output_layer"] = ops.ipu_copy(
            t=t_output_layer_result, destination=3)

    # Host store part of the cycle
    with pir.ipu(ipu=2):
        ops.host_store(t_out_d2h, t_output_activation)

    return micro_batch_tensor


def pipeline_cycle_n(t_out_d2h: DeviceToHostStream,
                     micro_batch_tensor: Dict[int, Dict[str, Tensor]]):
    """Write the last pipeline cycle to the IR.

    Args:
        t_out_d2h (DeviceToHostStream): The device to host stream
        micro_batch_tensor (Dict[int, Dict[str, Tensor]]): The map between the micro batch and the
          popart.ir tensors
    """
    # NOTE: IPU 2 is done with all the work in this cycle

    highest_micro_batch = max(micro_batch_tensor.keys())

    # Compute part of the cycle
    with pir.ipu(ipu=3):
        t_output_activation = ops.softmax(
            micro_batch_tensor[highest_micro_batch]["output_layer"], axis=1)
    # Host store part of the cycle
    with pir.ipu(ipu=2):
        ops.host_store(t_out_d2h, t_output_activation)


def test_pipelined_inference():
    """
    Test the pipelined inference model (using the grouped pipelining scheme).

    For pipelining nomenclature, please refer to
    https://docs.graphcore.ai/projects/popart-user-guide/en/latest/glossary.html

    For more about pipelining, please refer to docs/notes/transforms/pipelining.md

    The model consist of:
    - An input layer
    - A hidden layer
    - Gelu activation
    - A output layer
    - Softmax activation

    Pipelined the model will look like this

    Legend:
    MMA - Matmul and add (one for the hidden layer and one for the output layer)
    G   - Gelu
    S   - Softmax
    xx5 - Operator xx operating on micro batch number 5
    |   - Separation between two pipeline cycles (where copying happens)

    Pipeline Cycle -->
         <--- fill phase ---> <--- main --> <-- flush phase --->
    IPU0: MMA0 | MMA1 | MMA2 | MMA3 | MMA4 |      |       |    |
    IPU1:      | G0   | G1   | G2   | G3   | G4   |       |    |
    IPU2:      |      | MMA0 | MMA1 | MMA2 | MMA3 | MMA4  |    |
    IPU3:      |      |      | S0   | S1   | S2   | S3    | S4 |

    The test will compare the result with the result from `run_inference_torch_reference_model`
    implemented in torch.

    This is one in a series of test which demonstrates the usage of pipelining.
    """
    main_phase_repeats = 2  # How many times we will repeat the main phase of the pipelining
    # This model has 4 pipeline stages, which means that the fill phase will have
    # 4-1 = 3 pipeline cycles
    micro_batch_size = 3 + main_phase_repeats

    pipeline_test_runner = PipelineTestRunner(
        pipelined_inference_model, 2, micro_batch_size, main_phase_repeats)

    pipeline_test_runner.set_options()
    weights_and_biases = pipeline_test_runner.get_weights_and_biases()
    session, t_in_h2d, t_out_d2h = pipeline_test_runner.build_popart_ir_model(
        weights_and_biases)

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

    # Run the model
    for run_nr in range(pipeline_test_runner.batches["n_step_io_calls"]):
        # Create the dataset data
        dataset_data = pipeline_test_runner.create_data(
            (pipeline_test_runner.batches["micro_batch_size"],
             *pipeline_test_runner.stream_input_shape))
        full_dataset_data[run_nr, ...] = dataset_data

        data = {t_in_h2d: dataset_data}
        outputs = session.run(data)

        # Append the result
        result[run_nr, ...] = outputs[t_out_d2h]

    # Compare outcome from popart.ir with outcome from pytorch
    pipeline_test_runner.compare_with_reference(result, full_dataset_data,
                                                weights_and_biases)
