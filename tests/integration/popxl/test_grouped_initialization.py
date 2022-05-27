# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Dict, Iterable, List, Tuple
from typing_extensions import Literal

import torch
import numpy as np
import popart
import pytest

import popxl
import popxl.ops as ops
import popxl.dtypes as dtypes
import popxl.transforms as transforms

# Dimension sizes
# Our model multiplies AxB and BxC matrices
A = 3
B = 5
C = 4  # This needs to be divisible by group size when RTS is enabled

# Configurations used for the test
configs = [
    # ---------------------
    {
        "replicas": 2,
        "commType": popart.CommGroupType.All,
        # ->
        "inputs": {
            "stride": 1,
            "group_size": 2
        }
    },
    # ---------------------
    {
        "replicas": 4,
        "commType": popart.CommGroupType.Orthogonal,
        # ->
        "inputs": {
            "stride": 2,
            "group_size": 2
        }
    },
    # ----------------------
    {
        "replicas": 4,
        "commType": popart.CommGroupType.Consecutive,
        # ->
        "inputs": {
            "stride": 1,
            "group_size": 2
        }
    },
    # ----------------------
    {
        "replicas": 2,
        "commType": popart.CommGroupType.Ungrouped,
        # ->
        "inputs": {
            "stride": 1,
            "group_size": 1
        }
    },
    # ----------------------
    {
        "replicas": 2,
        "commType": popart.CommGroupType.All,
        # ->
        "inputs": None  # No replica_grouping to check the defaults
    },
]


def create_variable(x: np.ndarray, rts: bool, remote: bool,
                    rg: popxl.ReplicaGrouping, retrieval_mode: str,
                    replication_factor: int,
                    name: str) -> popxl.tensor.Variable:
    """
    Create a (possibly RTS / remote) variable given the replica grouping.

    Args:
        x: data to use for creating variable
        rts: whether the variable should use RTS
        remote: whether the variable should be remote
        rg: ReplicaGrouping to use for the variable
        retrieval_mode: Whether to return one replica per group,
            or all replicas.
        replication_factor: the replication factor
        name: name of the variable
    Returns:
        The following tuple:
        * v_loaded: the loaded variable to be used in the model.
        * v_remote: the remote variable for reading back the weights.
        * buffer: the buffer for reading back weights. Only relevant for remote variables.
    """
    buffer = None
    v_remote = None
    if rts and remote:
        shard_shape: Tuple[int, ...] = (x.size // replication_factor, )
        buffer = popxl.RemoteBuffer(shard_shape, dtypes.float32, 1)
        v_remote = popxl.remote_replica_sharded_variable(
            x,
            buffer,
            dtype=dtypes.float32,
            name=name,
            replica_grouping=rg,
            retrieval_mode=retrieval_mode)
        v_loaded = ops.remote_load(buffer, 0)
    elif rts and not remote:
        v_remote, v_loaded = popxl.replica_sharded_variable(
            x,
            dtypes.float32,
            name=name,
            replica_grouping=rg,
            retrieval_mode=retrieval_mode)
    elif not rts and remote:
        shape = x.shape
        if rg is not None and rg.num_groups > 1:
            # Strip first (num_groups) dimension
            shape = list(x.shape)[1:]
        buffer = popxl.RemoteBuffer(shape, dtypes.float32, 1)
        v_remote = popxl.remote_variable(x,
                                         buffer,
                                         dtype=dtypes.float32,
                                         name=name,
                                         replica_grouping=rg,
                                         retrieval_mode=retrieval_mode)
        v_loaded = ops.remote_load(buffer, 0)
    else:
        v_loaded = popxl.variable(x,
                                  dtypes.float32,
                                  name=name,
                                  replica_grouping=rg,
                                  retrieval_mode=retrieval_mode)
        v_remote = v_loaded
    return v_loaded, v_remote, buffer


class Linear(popxl.Module):
    """
    Simple model to return matrix multiplication of 2 weights.

    Calculates y = X @ W, with X being the input and W the variable
    """

    def __init__(self, weights_shape: Iterable[int]):
        self.W: popxl.Tensor = None
        self.weights_shape = weights_shape

    def build(self, X: popxl.Tensor) -> Tuple[popxl.Tensor, ...]:
        self.W = popxl.graph_input(self.weights_shape, popxl.float32,
                                   "W_subgraph")
        y = X @ self.W.transpose()
        return y


def get_weights_array(shape: List[int], num_groups: int = 1,
                      seed: int = 10111) -> np.ndarray:
    """
    Create num_groups random numpy arrays of size shape.

    If num_groups == 1, we keep the original shape, otherwise we
    introduce an additional num_groups dimension at the start of shape.

    Args:
        shape: the shape of the weight on a single replica
        num_groups: number of groups for this weight variable
        seed: numpy seed
    Returns:
        A numpy array of random data of the required shape
    """
    np.random.seed(seed)
    if num_groups > 1:
        # We need one weight per group
        shape = [num_groups] + shape

    array = np.random.random_sample(shape).astype(np.float32)
    return array


def create_simple_model(
        ir: popxl.Ir, session_type: Literal["inference", "training"],
        rg: popxl.ReplicaGrouping,
        retrieval_mode: Literal["one_per_group", "all_replicas"], rts: bool,
        remote: bool
) -> Tuple[np.ndarray, popxl.tensor.
           Variable, Dict[popxl.HostToDeviceStream, np.ndarray], popxl.
           DeviceToHostStream, np.ndarray, np.ndarray]:
    """
    Create a simple model to run for our test.

    * Initialise the data
    * Use the Linear class above for the forward pass
    * Calculate a weight gradient and subtract it from the initial weights.

    Args:
        ir: the Ir to be used for building the model
        session_type: whether to run a step of inference or training
        rg: replica grouping to be used for the weights variable
        retrieval_mode: Whether to return one replica per group,
            or all replicas.
        rts: whether we should use rts for weights variable
        remote: whether the weights variable should be remote
    Returns:
        A tuple of the following:
        * initial data passed to the weights variable
        * weights variable that can be used to retrieve the updated weights
        * inputs to be passed to Session
        * stream to retrieve output of forward pass
        * input (X) data
        * label data
    """

    # Set up shapes for input tensors
    data_shape = [A, B]
    output_shape = [A, C]
    full_weights_shape = [C, B]
    label_shape = [A]

    default_rg = rg if rg else ir.replica_grouping()
    num_groups = default_rg.num_groups

    weights_array = get_weights_array(full_weights_shape, num_groups)
    weights_array_init = get_weights_array(full_weights_shape, num_groups)

    # Sanity check
    assert np.allclose(weights_array, weights_array_init)

    # Attach replication factor to the shapes if necessary
    re_data_shape = data_shape
    re_label_shape = label_shape
    if ir.replication_factor > 1:
        re_data_shape = [ir.replication_factor] + data_shape
        re_label_shape = [ir.replication_factor] + label_shape

    # Set up inputs dictionary to be passed to Session
    inputs = {}
    # Set up model
    main = ir.main_graph
    with main, popxl.in_sequence():

        # Set up input stream for the input to be multiplied by the weight
        X_h2d = popxl.h2d_stream(data_shape, dtypes.float32, "X_stream")
        X = ops.host_load(X_h2d, "X")
        X_data = np.random.normal(0, 0.4,
                                  re_data_shape).astype(X.dtype.as_numpy())
        inputs[X_h2d] = X_data.copy()

        # Set up variable for which we use ReplicaGrouping
        W_loaded, W_remote, buffer1 = create_variable(weights_array,
                                                      rts,
                                                      remote,
                                                      rg,
                                                      retrieval_mode,
                                                      ir.replication_factor,
                                                      name="W")
        W_gathered = W_loaded
        if rts:
            W_gathered = ops.collectives.replicated_all_gather(
                W_loaded, default_rg)

        # Create linear model with required shapes and grouping
        linear = Linear(full_weights_shape)
        # Create and call subgraph for the linear model
        linear_graph = ir.create_graph(linear, X)
        fwd_call_info = ops.call_with_info(linear_graph,
                                           X,
                                           inputs_dict={linear.W: W_gathered})
        # Collect output
        y = fwd_call_info.outputs[0]
        y_d2h = popxl.d2h_stream(output_shape, popxl.float32, name="y_stream")
        ops.host_store(y_d2h, y)

        if session_type == "training":
            # Set up labels
            label_h2d = popxl.h2d_stream(label_shape, dtypes.int32,
                                         "label_stream")
            label = ops.host_load(label_h2d, "label")
            # Random one-hot vector
            label_data = np.random.randint(C, size=re_label_shape)
            inputs[label_h2d] = label_data.copy()

            # Pass inputs through softmax and nll optimizer to get gradients
            softmax = ops.softmax(y, 1)
            _, dy = ops.nll_loss_with_softmax_grad(softmax, label)
            bwd_graph_info = transforms.autodiff(linear_graph,
                                                 grads_required=[linear.W])
            activations = bwd_graph_info.inputs_dict(fwd_call_info)
            W_grad = ops.call(bwd_graph_info.graph,
                              dy,
                              inputs_dict=activations)[0]
            w_grad_d2h = popxl.d2h_stream(full_weights_shape,
                                          popxl.float32,
                                          name="w_grad_stream")
            ops.host_store(w_grad_d2h, W_grad)

            if rts:
                # Scatter W_grad
                W_grad = ops.collectives.replicated_reduce_scatter(
                    W_grad, 'add', default_rg, True)

            # Update W in-place
            ops.var_updates.accumulate_(W_loaded, W_grad, -1.0)
            if remote:
                # Need to store the updated weight in remote memory
                ops.remote_store(buffer1, 0, W_loaded)
    return weights_array_init, W_remote, inputs, y_d2h, X_data, label_data


def replicate(W: np.ndarray, groups: List[List[int]],
              replication_factor: int) -> np.ndarray:
    """
    Take a tensor that only returned one replica per group and replicate
    it to how it would look if it returned all replicas per group.

    Args:
        W: the numpy array to replicate
        groups: a list of indices of replicas for each group
        replication_factor: the replication factor
    Returns:
        The expanded data array, so that it contains an entry for every replica
    """
    if len(groups) == 1:
        # Need to add a replica dimension in case there's only one group
        W = np.array([W])
    new_W = np.zeros([replication_factor, *W.shape[1:]]).astype(np.float32)
    for group_index in range(len(groups)):
        group = groups[group_index]
        for index in group:
            new_W[index] = W[group_index]
    return new_W


def dereplicate(data: np.ndarray, groups: List[List[int]]) -> np.ndarray:
    """
    Opposite operation to replicate.

    Given a tensor that contains all it's replicas return the first replica
    of each group, so that they coincide with a grouped tensor that returns
    one replica per group.

    Args:
        data: the numpy array to dereplicate
        groups: a list of indices of replicas for each group
    Returns:
        The pruned data array, so that it only contains the first replica per group
    """
    new_data = np.zeros([len(groups), *data.shape[1:]]).astype(np.float32)
    for group_index in range(len(groups)):
        group = groups[group_index]
        # Fetch the first element of the group
        new_data[group_index] = data[group[0]]
    return new_data


def get_torch_grads(Xs: np.ndarray, Ws: np.ndarray, labels: np.ndarray,
                    Ys: np.ndarray) -> np.ndarray:
    """
    Simulate create_simple_model in torch, returning the gradients of W.

    This is done once for each replica.

    Args:
        Xs: list of input values (X)
        Ws: list of weight values (W)
        labels: list of label values
        Ys: list of output values (y = X @ W). Used as a sanity check for torch output.
    Returns:
        The W gradients that are produced from running the model
    """
    grads = []
    for i in range(len(Xs)):
        X = torch.tensor(Xs[i])
        W = torch.tensor(Ws[i], requires_grad=True)
        label = torch.tensor(labels[i])
        y = X @ W.T

        # Sanity check that the model is correct
        assert np.allclose(y.detach().numpy(), Ys[i])
        softmax = torch.nn.functional.log_softmax(y, dim=1)

        nll = torch.nn.functional.nll_loss(softmax, label)

        nll.backward()
        grads.append(W.grad.numpy())
    return np.array(grads).astype(np.float32)


def accumulate_groups(data: np.ndarray, groups: List[List[int]]) -> np.ndarray:
    """Replace the value of each replica by the sum of values of that group

    Args:
        data (np.ndarray): the data to be accumulated
        groups (List[List[int]]): groups to accumulate over

    Returns:
        np.ndarray: the accumulated data
    """
    new_data = data.copy()
    for group in groups:
        group_sum = sum(data[i] for i in group)
        for index in group:
            new_data[index] = group_sum
    return new_data


@pytest.mark.parametrize("config", configs)
@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("rts", [False, True])
@pytest.mark.parametrize("retrieval_mode", ["all_replicas", "one_per_group"])
def test_grouped_initialization(config, remote: bool, rts: bool,
                                retrieval_mode: str):
    """
    Run a simple model in PopART with the specified configs,
    and compare it's output to the same model run in torch.

    This test vaguely follows the test_grouped_initialization
    test from grouped_initialization_test.py.
    """
    # No point in testing inference, since training does the same, just more
    session_type = "training"

    # Create Ir
    ir = popxl.Ir()

    # Set up number of replicas
    ir.replication_factor = config["replicas"]

    # Initialise replica grouping (equivalent of VariableSettings in PopART)
    if config["inputs"] is None:
        # No replica grouping for the weights
        rg = None
    else:
        rg = ir.replica_grouping(**config["inputs"])
    # Create a simple linear model
    w_data, w, inputs, y_d2h, X_data, label_data = create_simple_model(
        ir, session_type, rg, retrieval_mode, rts, remote)

    # Run a session on an ipu device
    with popxl.Session(ir, "ipu_hw") as session:
        outputs = session.run(inputs)

    # Retrieve the updated weight value
    final_weight = session.get_tensor_data(w)

    if rg is None:
        # Model already run, so it's safe to initialize rg
        rg = ir.replica_grouping()
    groups = rg._to_variable_settings(retrieval_mode).groups(
        ir.replication_factor)

    # Get forward pass output
    y = outputs[y_d2h]

    W_replicated = replicate(w_data, groups, ir.replication_factor)
    for i in range(ir.replication_factor):
        assert np.allclose(X_data[i] @ W_replicated[i].T, y[i])

    if session_type == "inference":
        # Check that weights don't change during inference
        assert np.allclose(final_weight, w_data)
    else:
        # Reshape in case there's no replication factor dimension
        w_data = w_data.reshape((-1, C, B))
        final_weight = final_weight.reshape((-1, C, B))

        # Compare our model to a torch model
        grads = get_torch_grads(X_data, W_replicated, label_data, y)
        if rts:
            # The variables are shared, so we sum up the gradients from all group replicas
            grads = accumulate_groups(grads, groups)
        torch_final_weight = np.array(
            [W - W_grad for W, W_grad in zip(W_replicated, grads)])
        if retrieval_mode == "one_per_group":
            torch_final_weight = dereplicate(torch_final_weight, groups)
        np.testing.assert_allclose(final_weight,
                                   torch_final_weight,
                                   rtol=1e-5,
                                   atol=1e-8)
