# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""Contains the PipelineTestRunner class."""

from typing import Tuple, Callable, Dict
import numpy as np
import popxl
import popart
from .reference_models import run_inference_torch_reference_model


class PipelineTestRunner:
    """Helper class to run the pipeline tests."""

    def __init__(
        self, model: Callable, n_step_io_calls: int, micro_batch_size: int, *args
    ) -> None:
        """Setup dimension, batches, the model and the IR.

        Args:
            model (Callable): The popxl model to run
            n_step_io_calls (int): Number of calls to PyStepIo
            micro_batch_size (int): Number of micro batches.
              This is the number of samples calculated in one full forward/backward pass of the algorithm
              and must equal the batches per step and number of ops.host_load in the popxl model
            *args: Additional arguments to the model
        """
        # Store the model
        self.model = model
        self.args = args

        # Setup dimensions of neural net
        self.nn_dims = {
            "n_inputs": 4,  # Number of parameters in the input
            "n_outputs": 2,  # Number of outputs from the NN
            "nodes_in_hidden_layer": 16,
        }

        # Setup batches
        self.batches = {
            # How many times we will call PyStepIO
            "n_step_io_calls": n_step_io_calls,
            # The number of samples for which activations/gradients are computed in parallel
            "compute_batches": 3,
            # The number of samples calculated in one full forward/backward pass of the algorithm
            # This must equal the batches per step and number of ops.host_load in the popxl model
            "micro_batch_size": micro_batch_size,
        }

        # Initialize the ir
        self.ir = popxl.Ir()

        # Specify the input shape to the input stream
        self.stream_input_shape = (
            self.batches["compute_batches"],
            self.nn_dims["n_inputs"],
        )

    @staticmethod
    def create_data(shape: Tuple[int, ...]) -> np.array:
        """Create random data.

        Args:
            shape (Tuple[int, ...]): The shape of the resulting data

        Returns:
            np.array: The resulting data.
        """
        return np.random.normal(0, 0.1, shape).astype(np.float32)

    def set_options(self) -> None:
        """Set session options needed for explicit pipelining."""
        opts = self.ir._pb_ir.getSessionOptions()
        opts.useHostCopyOps = (
            True  # Use IR graph operations for data and anchor streams
        )
        opts.constantWeights = (
            False  # Sets the weights in an inference session to constant
        )
        opts.enableExplicitMainLoops = True  # Disables implicit training loops
        opts.aliasZeroCopy = True  # Enable zero copy for subgraphs
        opts.explicitRecomputation = True  # Disable implicit recomputation
        opts.virtualGraphMode = (
            popart.VirtualGraphMode.Manual
        )  # Set the virt graph manually

    def get_weights_and_biases(self) -> Dict[str, np.array]:
        """Generate the weights and biases.

        Returns:
            (Dict[str, np.array]): dict containing:

                hidden_weights (np.array): Data for the weights of the hidden layer
                hidden_bias (np.array): Data for the bias of the hidden layer
                output_weights (np.array): Data for the weights of the output layer
                output_bias (np.array): Data for the bias of the output layer
        """
        # Define the needed tensor shapes
        hidden_weights_shape = (
            self.nn_dims["n_inputs"],
            self.nn_dims["nodes_in_hidden_layer"],
        )
        hidden_bias_shape = (
            self.batches["compute_batches"],
            self.nn_dims["nodes_in_hidden_layer"],
        )
        output_weights_shape = (
            self.nn_dims["nodes_in_hidden_layer"],
            self.nn_dims["n_outputs"],
        )
        output_bias_shape = (self.batches["compute_batches"], self.nn_dims["n_outputs"])

        # Initialize the weights and biases
        # These weights and biases will be made into popxl.variable
        # This means that their value will be streamed once to the IPU, and not every time
        # you call PyStepIo
        weights_and_biases = {
            "hidden_weights": self.create_data(hidden_weights_shape),
            "hidden_bias": self.create_data(hidden_bias_shape),
            "output_weights": self.create_data(output_weights_shape),
            "output_bias": self.create_data(output_bias_shape),
        }

        return weights_and_biases

    def build_popxl_model(
        self, weights_and_biases: Dict[str, np.array]
    ) -> Tuple[popxl.Session, popxl.HostToDeviceStream, popxl.DeviceToHostStream]:
        """Build the popxl model.

        Args:
            weights_and_biases (Dict[str, np.array]): dict containing:

                hidden_weights (np.array): Data for the weights of the hidden layer
                hidden_bias (np.array): Data for the bias of the hidden layer
                output_weights (np.array): Data for the weights of the output layer
                output_bias (np.array): Data for the bias of the output layer

        Returns:
            (tuple): tuple containing:

                session (popxl.Session): The popxl session
                t_in_h2d (HostToDeviceStream): The host to device stream for the input
                t_out_d2h (DeviceToHostStream): The device to host stream for the output.
        """
        # Build the model
        t_in_h2d, t_out_d2h = self.model(
            self.ir, self.stream_input_shape, weights_and_biases, *self.args
        )

        self.ir.num_host_transfers = self.batches["micro_batch_size"]

        # Create an IR inference session
        session = popxl.Session(self.ir, "ipu_model")

        return session, t_in_h2d, t_out_d2h

    def compare_with_reference(
        self,
        results: np.array,
        dataset_data: np.array,
        weights_and_biases: Dict[str, np.array],
    ) -> None:
        """Compare the popxl results with the results from the reference model.

        Args:
            results (np.array): The results of running the popxl model
            dataset_data (np.array): The dataset to run the model on
            weights_and_biases (Dict[str, np.array]): dict containing:

                hidden_weights (np.array): Data for the weights of the hidden layer
                hidden_bias (np.array): Data for the bias of the hidden layer
                output_weights (np.array): Data for the weights of the output layer
                output_bias (np.array): Data for the bias of the output layer
        """
        # NOTE: We compute the softmax along axis n_outputs.
        #       As we've ordered the input data in
        #       (n_step_io_calls, micro_batch_size, compute_batches, n_inputs)
        #       This corresponds to axis number 3
        expected_results = run_inference_torch_reference_model(
            dataset_data, weights_and_biases, dim=3
        )

        assert results.shape == expected_results.shape
        assert results.dtype == expected_results.dtype
        assert np.allclose(results, expected_results)
