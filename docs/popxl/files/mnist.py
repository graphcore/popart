# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""
This example shows how to use the PopXL to create a linear model and then train it on the MNIST data set.
"""
# import begin
import argparse
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
    Get the training and testing data for mnist.
    """
    training_data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            '~/.torch/datasets',
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                # Mean and std computed on the training set.
                torchvision.transforms.Normalize((0.1307, ), (0.3081, )),
            ])),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

    validation_data = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        '~/.torch/datasets',
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307, ), (0.3081, )),
        ])),
                                                  batch_size=test_batch_size,
                                                  shuffle=True,
                                                  drop_last=True)
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

    def build(self, x: popxl.Tensor, out_features: int,
              bias: bool = True) -> Tuple[popxl.Tensor, ...]:
        """
        Override the `build` method to build a graph.
        """
        self.W = popxl.graph_input((x.shape[-1], out_features), popxl.float32,
                                   "W")
        y = x @ self.W
        if bias:
            self.b = popxl.graph_input((out_features, ), popxl.float32, "b")
            y = y + self.b

        y = ops.gelu(y)
        return y


# linear end


# network begin
def create_network_fwd_graph(
        ir, x) -> Tuple[Tuple[popxl.Tensor], Dict[str, popxl.Tensor], List[
            popxl.Graph], Tuple[CallSiteInfo]]:
    """
    Define the network architecture.

    Args:
        ir (popxl.Ir): The ir to create model in.
        x (popxl.Tensor): The input tensor of this model.

    Returns:
        Tuple[Tuple[popxl.Tensor], Dict[str, popxl.Tensor], List[popxl.Graph], Tuple[CallSiteInfo]]: The info needed to calculate the gradients later
    """
    # Linear layer 0
    x = x.reshape((-1, 28 * 28))
    W0_data = np.random.normal(0, 0.02, (x.shape[-1], 32)).astype(np.float32)
    W0 = popxl.variable(W0_data, name="W0")
    b0_data = np.random.normal(0, 0.02, (32)).astype(np.float32)
    b0 = popxl.variable(b0_data, name="b0")

    # Linear layer 1
    W1_data = np.random.normal(0, 0.02, (32, 10)).astype(np.float32)
    W1 = popxl.variable(W1_data, name="W1")
    b1_data = np.random.normal(0, 0.02, (10)).astype(np.float32)
    b1 = popxl.variable(b1_data, name="b1")

    # Create graph to call for linear layer 0
    linear_0 = Linear()
    linear_graph_0 = ir.create_graph(linear_0, x, out_features=32)

    # Call the linear layer 0 graph
    fwd_call_info_0 = ops.call_with_info(linear_graph_0,
                                         x,
                                         inputs_dict={
                                             linear_0.W: W0,
                                             linear_0.b: b0
                                         })
    # Output of linear layer 0
    x1 = fwd_call_info_0.outputs[0]

    # Create graph to call for linear layer 1
    linear_1 = Linear()
    linear_graph_1 = ir.create_graph(linear_1, x1, out_features=10)

    # Call the linear layer 1 graph
    fwd_call_info_1 = ops.call_with_info(linear_graph_1,
                                         x1,
                                         inputs_dict={
                                             linear_1.W: W1,
                                             linear_1.b: b1
                                         })
    # Output of linear layer 1
    y = fwd_call_info_1.outputs[0]

    outputs = (x1, y)
    params = {"W0": W0, "W1": W1, "b0": b0, "b1": b1}
    linears = [linear_0, linear_1]
    fwd_call_infos = (fwd_call_info_0, fwd_call_info_1)

    return outputs, params, linears, fwd_call_infos


# network end


def calculate_grads(dy, outputs, params, linears,
                    fwd_call_infos) -> Dict[str, popxl.Tensor]:
    """
    Calculate the gradients w.r.t. weights and bias.
    """
    # grad_1 begin
    # Obtain graph to calculate gradients from autodiff
    bwd_graph_info_1 = transforms.autodiff(fwd_call_infos[1].called_graph)

    # Get activations for layer 1 from forward call info
    activations_1 = bwd_graph_info_1.inputs_dict(fwd_call_infos[1])

    # Get the gradients dictionary by calling the gradient graphs with ops.call_with_info
    grads_1_call_info = ops.call_with_info(bwd_graph_info_1.graph,
                                           dy,
                                           inputs_dict=activations_1)
    # Find the corresponding gradient w.r.t. the input, weights and bias
    grads_1 = bwd_graph_info_1.fwd_parent_ins_to_grad_parent_outs(
        fwd_call_infos[1], grads_1_call_info)
    x1 = outputs[0]
    W1 = params["W1"]
    b1 = params["b1"]
    grad_x_1 = grads_1[x1]
    grad_w_1 = grads_1[W1]
    grad_b_1 = grads_1[b1]
    # grad_1 end
    # grad_0 begin
    # Use autodiff to obtain graph that calculate gradients, specify which graph inputs need gradients
    bwd_graph_info_0 = transforms.autodiff(
        fwd_call_infos[0].called_graph,
        grads_required=[linears[0].W, linears[0].b])
    # Get activations for layer 0 from forward call info
    activations_0 = bwd_graph_info_0.inputs_dict(fwd_call_infos[0])
    # Get the required gradients by calling the gradient graphs with ops.call
    grad_w_0, grad_b_0 = ops.call(bwd_graph_info_0.graph,
                                  grad_x_1,
                                  inputs_dict=activations_0)
    # grad_0 end
    grads = {"W0": grad_w_0, "b0": grad_b_0, "W1": grad_w_1, "b1": grad_b_1}

    return grads


# update begin
def update_weights_bias(opts, grads, params) -> None:
    """
    Update weights and bias by W += - lr * grads_w, b += - lr * grads_b.
    """
    for k, v in params.items():
        ops.scaled_add_(v, grads[k], b=-opts.lr)


# update end


def build_train_ir(
        opts) -> Tuple[popxl.Ir, Tuple[popxl.HostToDeviceStream], popxl.
                       DeviceToHostStream, ]:
    """
    Build the IR for training.
        - load input data
        - buid the network forward pass
        - calculating the gradients, and
        - finally update weights and bias
        - store output data
    """
    # create ir begin
    ir = popxl.Ir()
    ir.num_host_transfers = 1
    ir.replication_factor = 1
    with ir.main_graph, popxl.in_sequence():
        # create ir end
        # h2d begin
        # Host load input and labels
        img_stream = popxl.h2d_stream([opts.batch_size, 28, 28],
                                      popxl.float32,
                                      name="input_stream")
        x = ops.host_load(img_stream, "x")

        label_stream = popxl.h2d_stream([opts.batch_size],
                                        popxl.int32,
                                        name="label_stream")
        labels = ops.host_load(label_stream, "labels")
        # h2d end

        # Build forward pass graph
        outputs, params, linears, fwd_call_infos = create_network_fwd_graph(
            ir, x)
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
        loss_stream = popxl.d2h_stream(loss.shape,
                                       loss.dtype,
                                       name="loss_stream")
        ops.host_store(loss_stream, loss)
        # out end
    return ir, (img_stream, label_stream), loss_stream, params.values()


def build_test_ir(
        opts) -> Tuple[popxl.Ir, Tuple[popxl.HostToDeviceStream], popxl.
                       DeviceToHostStream, ]:
    """
    Build the IR for testing.
    """
    ir = popxl.Ir()
    ir.num_host_transfers = 1
    ir.replication_factor = 1
    with ir.main_graph, popxl.in_sequence():
        # Host load input and labels
        img_stream = popxl.h2d_stream([opts.test_batch_size, 28, 28],
                                      popxl.float32,
                                      name="input_stream")
        x = ops.host_load(img_stream, "x")

        # Build forward pass graph
        outputs, params, linears, fwd_call_infos = create_network_fwd_graph(
            ir, x)

        # Host store to get loss
        out_stream = popxl.d2h_stream(outputs[1].shape,
                                      outputs[1].dtype,
                                      name="loss_stream")
        ops.host_store(out_stream, outputs[1])

    return ir, img_stream, out_stream, params.values()


# train begin
def train(train_session, training_data, opts, input_streams,
          loss_stream) -> None:
    nb_batches = len(training_data)
    for epoch in range(1, opts.epochs + 1):
        print("Epoch {0}/{1}".format(epoch, opts.epochs))
        bar = tqdm(training_data, total=nb_batches)
        for data, labels in bar:
            inputs: Mapping[popxl.HostToDeviceStream, np.ndarray] = dict(
                zip(input_streams,
                    [data.squeeze().float().numpy(),
                     labels.int().numpy()]))

            outputs = train_session.run(inputs)
            loss = outputs[loss_stream]
            bar.set_description(f"Average loss: {np.mean(loss):.4f}")


# train end


def test(test_session, test_data, input_streams, out_stream) -> None:
    nr_batches = len(test_data)
    sum_acc = 0.0
    with torch.no_grad():
        for data, labels in tqdm(test_data, total=nr_batches):
            inputs: Mapping[popxl.HostToDeviceStream, np.ndarray] = {
                input_streams: data.squeeze().float().numpy()
            }
            output = test_session.run(inputs)
            sum_acc += get_accuracy(output[out_stream], labels)
    print("Accuracy on test set: {:0.2f}%".format(sum_acc / len(test_data)))


def get_test_var_values(test_variables, trained_weights_data_dict
                        ) -> Dict[popxl.tensor.Variable, np.ndarray]:
    test_weights_data_dict = {}
    for train_var, value in trained_weights_data_dict.items():
        for test_var in test_variables:
            if train_var.name == test_var.name:
                test_weights_data_dict[test_var] = value
                break
    return test_weights_data_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description='MNIST training in Popart',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help=
        "Set the Batch size. This must be a multiple of the replication factor."
    )
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=80,
                        help='batch size for testing')
    parser.add_argument('--epochs',
                        type=int,
                        default=1,
                        help="Number of epochs to train for.")
    parser.add_argument('--lr',
                        type=float,
                        default=0.005,
                        help="Learning rate.")

    opts = parser.parse_args()
    # Get the data for training and validation
    training_data, test_data = get_mnist_data(opts.test_batch_size,
                                              opts.batch_size)

    # Build the ir for training
    train_ir, input_streams, loss_stream, train_variables = build_train_ir(
        opts)
    # session begin
    train_session = popxl.Session(train_ir, 'ipu_model')
    train(train_session, training_data, opts, input_streams, loss_stream)
    # session end
    trained_weights_data_dict = train_session.get_tensors_data(train_variables)
    train_session.device.detach()
    print("Training complete.")

    # test begin
    # Build the ir for testing
    test_ir, test_input_streams, out_stream, test_variables = build_test_ir(
        opts)
    test_session = popxl.Session(test_ir, 'ipu_model')
    # Get test variable values from trained weights
    test_weights_data_dict = get_test_var_values(test_variables,
                                                 trained_weights_data_dict)
    # Copy trained weights to the test ir
    test_session.write_variables_data(test_weights_data_dict)

    test(test_session, test_data, test_input_streams, out_stream)
    test_session.device.detach()
    # test end
    print("Testing complete.")


if __name__ == "__main__":
    main()
