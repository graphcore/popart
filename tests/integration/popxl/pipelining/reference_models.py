# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""Contains reference models written in torch."""

from typing import Dict
import torch
import numpy as np


def run_inference_torch_reference_model(
    dataset_data: np.array, weights_and_biases: Dict[str, np.array], dim: int
) -> np.array:
    """Run inference on the reference model implemented in torch.

    Args:
        dataset_data (np.array): The whole dataset to be processed
        weights_and_biases (Dict[str, np.array]): dict containing:

            hidden_weights (np.array): Data for the weights of the hidden layer
            hidden_bias (np.array): Data for the bias of the hidden layer
            output_weights (np.array): Data for the weights of the output layer
            output_bias (np.array): Data for the bias of the output layer
        dim (int): The dimension along which Softmax will be computed
          (so every slice along dim will sum to 1).

    Returns:
        np.array: The result of the pytorch operations
    """
    t_in = torch.from_numpy(dataset_data)
    hidden_weights = torch.from_numpy(weights_and_biases["hidden_weights"])
    hidden_bias = torch.from_numpy(weights_and_biases["hidden_bias"])
    output_weights = torch.from_numpy(weights_and_biases["output_weights"])
    output_bias = torch.from_numpy(weights_and_biases["output_bias"])

    # The hidden layer
    t_hidden_matmul = torch.matmul(t_in, hidden_weights)
    t_hidden_bias = torch.add(t_hidden_matmul, hidden_bias)
    t_hidden_layer_activation = torch.nn.functional.gelu(t_hidden_bias)

    # The output layer
    t_output_matmul = torch.matmul(t_hidden_layer_activation, output_weights)
    t_output_bias = torch.add(t_output_matmul, output_bias)
    t_output_layer_activation = torch.nn.functional.softmax(t_output_bias, dim=dim)

    return t_output_layer_activation.numpy()
