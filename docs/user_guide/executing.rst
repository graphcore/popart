Executing graphs
----------------

The Session class runs graphs on the IPU hardware.

Data feeds can be from single python or numpy arrays, from python iterators
producing many tensors, and from specialized high-performance data feed objects.

The graph can be executed in inference, evaluation or training modes.

In inference, only the forward pass will be executed. The user is
responsible for ensuring that the forward graph finishes with the appropriate
operation for an inference.  If the ONNX file does not contain a Softmax on
the end, then the user should use the ``builder`` class to append a Softmax.

In evaluation, the forward pass and the losses will be executed, and the
final loss value will be returned.

In training, a full forward pass, loss calculation and backward pass will be
done.

