# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""
This example shows how to use the PopXL to create a linear model and then train it on the MNIST data set.
"""
# import begin
import argparse
from collections import namedtuple
from typing import Dict, List, Tuple, Mapping
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import popxl
import popxl.ops as ops
import popxl.transforms as transforms
from popxl.ops.call import CallSiteInfo

# import end


# dataset begin
def get_mnist_data(
    test_batch_size: int, batch_size: int
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Get the training and testing data for MNIST.

    Args:
        test_batch_size (int): The batch size for test.
        batch_size (int): The batch size for training.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: the data loaders for training data and test data.
    """
    training_data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "~/.torch/datasets",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    # Mean and std computed on the training set.
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    validation_data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "~/.torch/datasets",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=test_batch_size,
        shuffle=True,
        drop_last=True,
    )
    return training_data, validation_data


# dataset end


def get_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the accuracy of predictions.
    """
    ind = np.argmax(predictions, axis=-1).flatten()
    labels = labels.detach().numpy().flatten()
    return np.mean(ind == labels) * 100.0


# linear begin
class Linear(popxl.Module):
    def __init__(self) -> None:
        """
        Define a linear layer in PopXL.
        """
        self.W: popxl.Tensor = None
        self.b: popxl.Tensor = None

    def build(
        self, x: popxl.Tensor, out_features: int, bias: bool = True
    ) -> Tuple[popxl.Tensor, ...]:
        """
        Override the `build` method to build a graph.
        """
        self.W = popxl.graph_input((x.shape[-1], out_features), popxl.float32, "W")
        y = x @ self.W
        if bias:
            self.b = popxl.graph_input((out_features,), popxl.float32, "b")
            y = y + self.b

        y = ops.gelu(y)
        return y


# linear end

Trainable = namedtuple("Trainable", ["var", "shards", "full", "remote_buffer"])


# network begin
def create_network_fwd_graph(
    ir, x, opts
) -> Tuple[
    Tuple[popxl.Tensor], Dict[str, popxl.Tensor], List[popxl.Graph], Tuple[CallSiteInfo]
]:
    """
    Define the network architecture.

    Args:
        ir (popxl.Ir): The ir to create model in.
        x (popxl.Tensor): The input tensor of this model.
        opts (Namespace): options returned from args parser.

    Returns:
        Tuple[Tuple[popxl.Tensor], Dict[str, popxl.Tensor],
        List[popxl.Graph], Tuple[CallSiteInfo]]:
        The info needed to calculate the gradients later
    """
    # Linear layer 0
    x = x.reshape((-1, 28 * 28))
    W0_data = np.random.normal(0, 0.02, (x.shape[-1], 32)).astype(np.float32)
    # rts_W0 begin
    if ir.replication_factor > 1 and opts.rts:
        W0_remote, W0_shards = popxl.replica_sharded_variable(
            W0_data, dtype=popxl.float32, name="W0"
        )
        W0 = ops.collectives.replicated_all_gather(W0_shards)
        trainable_w0 = Trainable(W0_remote, W0_shards, W0, None)
    else:
        W0 = popxl.variable(W0_data, name="W0")
        trainable_w0 = Trainable(W0, None, None, None)
    # rts_W0 end
    b0_data = np.random.normal(0, 0.02, (32)).astype(np.float32)
    b0 = popxl.variable(b0_data, name="b0")
    trainable_b0 = Trainable(b0, None, None, None)

    # Linear layer 1
    W1_data = np.random.normal(0, 0.02, (32, 10)).astype(np.float32)

    # rts_W1 begin
    if ir.replication_factor > 1 and opts.rts:
        # create remote buffer that match shard shape and dtype
        var_shard_shape: Tuple[int, ...] = (W1_data.size // ir.replication_factor,)
        buffer = popxl.remote_buffer(
            var_shard_shape, popxl.float32, entries=ir.replication_factor
        )
        # create remote rts variable
        W1_remote = popxl.remote_replica_sharded_variable(W1_data, buffer, 0)
        # load the remote rts variable from each shard
        W1_shards = ops.remote_load(buffer, 0)
        # gather all the shards to get the full weight
        W1 = ops.collectives.replicated_all_gather(W1_shards)
        trainable_w1 = Trainable(W1_remote, W1_shards, W1, buffer)
    else:
        W1 = popxl.variable(W1_data, name="W1")
        trainable_w1 = Trainable(W1, None, None, None)
    # rts_W1 end
    b1_data = np.random.normal(0, 0.02, (10)).astype(np.float32)
    b1 = popxl.variable(b1_data, name="b1")
    trainable_b1 = Trainable(b1, None, None, None)
    # Create graph to call for linear layer 0
    linear_0 = Linear()
    linear_graph_0 = ir.create_graph(linear_0, x, out_features=32)

    # Call the linear layer 0 graph
    fwd_call_info_0 = ops.call_with_info(
        linear_graph_0, x, inputs_dict={linear_0.W: W0, linear_0.b: b0}
    )
    # Output of linear layer 0
    x1 = fwd_call_info_0.outputs[0]

    # Create graph to call for linear layer 1
    linear_1 = Linear()
    linear_graph_1 = ir.create_graph(linear_1, x1, out_features=10)

    # Call the linear layer 1 graph
    fwd_call_info_1 = ops.call_with_info(
        linear_graph_1, x1, inputs_dict={linear_1.W: W1, linear_1.b: b1}
    )
    # Output of linear layer 1
    y = fwd_call_info_1.outputs[0]

    outputs = (x1, y)
    params = {
        "W0": trainable_w0,
        "W1": trainable_w1,
        "b0": trainable_b0,
        "b1": trainable_b1,
    }
    linear_layers = [linear_0, linear_1]
    fwd_call_infos = (fwd_call_info_0, fwd_call_info_1)

    return outputs, params, linear_layers, fwd_call_infos


# network end


def calculate_grads(
    dy, outputs, params, linears, fwd_call_infos
) -> Dict[str, popxl.Tensor]:
    """
    Calculate the gradients w.r.t. weights and bias.
    """
    # grad_1 begin
    # Obtain graph to calculate gradients from autodiff
    bwd_graph_info_1 = transforms.autodiff(fwd_call_infos[1].called_graph)

    # Get activations for layer 1 from forward call info
    activations_1 = bwd_graph_info_1.inputs_dict(fwd_call_infos[1])

    # Get the gradients dictionary by calling the gradient graphs with ops.call_with_info
    grads_1_call_info = ops.call_with_info(
        bwd_graph_info_1.graph, dy, inputs_dict=activations_1
    )
    # Find the corresponding gradient w.r.t. the input, weights and bias
    grads_1 = bwd_graph_info_1.fwd_parent_ins_to_grad_parent_outs(
        fwd_call_infos[1], grads_1_call_info
    )
    x1 = outputs[0]
    if params["W1"].full is not None:
        W1 = params["W1"].full
    else:
        W1 = params["W1"].var
    b1 = params["b1"].var
    grad_x_1 = grads_1[x1]
    grad_w_1 = grads_1[W1]
    if params["W1"].shards is not None:
        grad_w_1 = ops.collectives.replica_sharded_slice(grad_w_1)
    grad_b_1 = grads_1[b1]
    # grad_1 end
    # grad_0 begin
    # Use autodiff to obtain graph that calculate gradients, specify which graph inputs need gradients
    bwd_graph_info_0 = transforms.autodiff(
        fwd_call_infos[0].called_graph, grads_required=[linears[0].W, linears[0].b]
    )
    # Get activations for layer 0 from forward call info
    activations_0 = bwd_graph_info_0.inputs_dict(fwd_call_infos[0])
    # Get the required gradients by calling the gradient graphs with ops.call
    grad_w_0, grad_b_0 = ops.call(
        bwd_graph_info_0.graph, grad_x_1, inputs_dict=activations_0
    )
    if params["W0"].shards is not None:
        grad_w_0 = ops.collectives.replica_sharded_slice(grad_w_0)
    # grad_0 end
    grads = {"W0": grad_w_0, "b0": grad_b_0, "W1": grad_w_1, "b1": grad_b_1}

    return grads


# update begin
def update_weights_bias(opts, grads, params) -> None:
    """
    Update weights and bias by W += - lr * grads_w, b += - lr * grads_b.
    """
    for k, v in params.items():
        if v.shards is not None:
            # rts variable
            ops.scaled_add_(v.shards, grads[k], b=-opts.lr)
            if v.remote_buffer is not None:
                ops.remote_store(v.remote_buffer, 0, v.shards)
        else:
            # not rts variable
            ops.scaled_add_(v.var, grads[k], b=-opts.lr)


# update end


def build_train_ir(
    opts,
) -> Tuple[popxl.Ir, Tuple[popxl.HostToDeviceStream], popxl.DeviceToHostStream,]:
    """
    Build the IR for training.
        - load input data
        - build the network forward pass
        - calculating the gradients, and
        - finally update weights and bias
        - store output data
    """
    # create ir begin
    ir = popxl.Ir()
    ir.num_host_transfers = 1
    ir.replication_factor = opts.replication_factor
    with ir.main_graph, popxl.in_sequence():
        # create ir end
        # h2d begin
        # Host load input and labels
        img_stream = popxl.h2d_stream(
            [opts.batch_size, 28, 28], popxl.float32, name="input_stream"
        )
        x = ops.host_load(img_stream, "x")

        label_stream = popxl.h2d_stream(
            [opts.batch_size], popxl.int32, name="label_stream"
        )
        labels = ops.host_load(label_stream, "labels")
        # h2d end

        # Build forward pass graph
        outputs, params, linears, fwd_call_infos = create_network_fwd_graph(ir, x, opts)
        # loss begin
        # Calculate loss and initial gradients
        probs = ops.softmax(outputs[1], axis=-1)
        loss, dy = ops.nll_loss_with_softmax_grad(probs, labels)
        # loss end
        # Build backward pass graph to calculate gradients
        grads = calculate_grads(dy, outputs, params, linears, fwd_call_infos)

        # Update weights and bias
        update_weights_bias(opts, grads, params)
        # out begin
        # Host store to get loss
        loss_stream = popxl.d2h_stream(loss.shape, loss.dtype, name="loss_stream")
        ops.host_store(loss_stream, loss)
        # out end
    return ir, (img_stream, label_stream), loss_stream, params.values()


def build_test_ir(
    opts,
) -> Tuple[popxl.Ir, Tuple[popxl.HostToDeviceStream], popxl.DeviceToHostStream,]:
    """
    Build the IR for testing.
    """
    ir = popxl.Ir()
    ir.num_host_transfers = 1
    ir.replication_factor = opts.replication_factor
    with ir.main_graph, popxl.in_sequence():
        # Host load input and labels
        img_stream = popxl.h2d_stream(
            [opts.test_batch_size, 28, 28], popxl.float32, name="input_stream"
        )
        x = ops.host_load(img_stream, "x")

        # Build forward pass graph
        outputs, params, _, _ = create_network_fwd_graph(ir, x, opts)

        # Host store to get loss
        out_stream = popxl.d2h_stream(
            outputs[1].shape, outputs[1].dtype, name="loss_stream"
        )
        ops.host_store(out_stream, outputs[1])

    return ir, img_stream, out_stream, params.values()


# train begin
def train(train_session, training_data, opts, input_streams, loss_stream) -> None:
    nb_batches = len(training_data)
    for epoch in range(1, opts.epochs + 1):
        print("Epoch {0}/{1}".format(epoch, opts.epochs))
        bar = tqdm(training_data, total=nb_batches)
        # train_session_inputs begain
        for data, labels in bar:
            if opts.replication_factor > 1:
                data = data.reshape((opts.replication_factor, opts.batch_size, 28, 28))
                labels = labels.reshape((opts.replication_factor, opts.batch_size))
                inputs: Mapping[popxl.HostToDeviceStream, np.ndarray] = dict(
                    zip(input_streams, [data.float().numpy(), labels.int().numpy()])
                )
            else:
                inputs: Mapping[popxl.HostToDeviceStream, np.ndarray] = dict(
                    zip(
                        input_streams,
                        [data.squeeze().float().numpy(), labels.int().numpy()],
                    )
                )
            # train_session_inputs end
            outputs = train_session.run(inputs)
            loss = outputs[loss_stream]
            bar.set_description(f"Average loss: {np.mean(loss):.4f}")


# train end


def test(test_session, test_data, opts, input_streams, out_stream) -> None:
    nr_batches = len(test_data)
    sum_acc = 0.0
    with torch.no_grad():
        for data, labels in tqdm(test_data, total=nr_batches):
            if opts.replication_factor > 1:
                data = data.reshape(
                    (opts.replication_factor, opts.test_batch_size, 28, 28)
                )
                inputs: Mapping[popxl.HostToDeviceStream, np.ndarray] = {
                    input_streams: data.float().numpy()
                }
            else:
                inputs: Mapping[popxl.HostToDeviceStream, np.ndarray] = {
                    input_streams: data.squeeze().float().numpy()
                }
            output = test_session.run(inputs)
            sum_acc += get_accuracy(output[out_stream], labels)
    print("Accuracy on test set: {:0.2f}%".format(sum_acc / len(test_data)))


def get_test_var_values(
    test_variables, trained_weights_data_dict
) -> Dict[popxl.tensor.Variable, np.ndarray]:
    test_weights_data_dict = {}
    for train_var, value in trained_weights_data_dict.items():
        for test_var in test_variables:
            if train_var.name == test_var.var.name:
                test_weights_data_dict[test_var.var] = value
                break
    return test_weights_data_dict


def run_train(training_data, opts):
    # Build the ir for training
    train_ir, input_streams, loss_stream, train_variables = build_train_ir(opts)
    # session begin
    if opts.ipu:
        train_session = popxl.Session(train_ir, "ipu_hw")
    else:
        train_session = popxl.Session(train_ir, "ipu_model")
    with train_session:
        train(train_session, training_data, opts, input_streams, loss_stream)
    # session end
    trained_weights_data_dict = train_session.get_tensors_data(
        [v.var for v in train_variables]
    )

    print("Training complete.")
    return trained_weights_data_dict


def run_test(test_data, trained_weights_data_dict, opts):
    # test begin
    # Build the ir for testing
    test_ir, test_input_streams, out_stream, test_variables = build_test_ir(opts)
    if opts.ipu:
        test_session = popxl.Session(test_ir, "ipu_hw")
    else:
        test_session = popxl.Session(test_ir, "ipu_model")
    # Get test variable values from trained weights
    test_weights_data_dict = get_test_var_values(
        test_variables, trained_weights_data_dict
    )
    # Copy trained weights to the test ir
    test_session.write_variables_data(test_weights_data_dict)
    with test_session:
        test(test_session, test_data, opts, test_input_streams, out_stream)
    # test end
    print("Testing complete.")


def main() -> None:
    opts = parser()
    # Get the data for training and validation
    training_data, test_data = get_mnist_data(
        opts.test_batch_size * opts.replication_factor,
        opts.batch_size * opts.replication_factor,
    )
    trained_weights_data_dict = run_train(training_data, opts)
    if opts.test:
        run_test(test_data, trained_weights_data_dict, opts)


def parser():
    parser = argparse.ArgumentParser(
        description="MNIST training in PopXL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training on a single replica.",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument(
        "--ipu", action="store_true", help="Run on available IPU hardware device."
    )
    parser.add_argument(
        "--replication-factor", type=int, default=1, help="Number of replications"
    )
    parser.add_argument("--rts", action="store_true", help="Enable RTS for variables.")
    parser.add_argument("--test", action="store_true", help="Test the trained model.")
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=80,
        help="Batch size for testing on a single replica.",
    )

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    main()
