Using ONNX with Poplar
======================

Introduction
------------

PopONNX is part of the Poplar set of tools for designing and running algorithms
on networks of Graphcore IPU processors.

It has three main features:

1) It can import ONNX graphs into a runtime environment.
2) It runs imported graphs in inference, evaluation or training modes, by
building a Poplar engine, hooking up data feeds and schduling the execution
of the Engine.
3) It provdes a simple interface for constructing ONNX graphs without needing
a third party framework.

APIs are available for C++ and python.


Supported operations
----------

PopOnnx is compatible with ONNX
[1.3](https://github.com/onnx/onnx/blob/master/docs/Versioning.md).
It supports ONNX
[ai.onnx](https://github.com/onnx/onnx/blob/master/docs/Operators.md) operator
set version 9.


References
--------

- https://onnx.ai/
- https://pytorch.org/docs/stable/index.html
- Poplar user guide


Importing graphs
--------------

The Net class is the runtime environment for executing graphs on the IPU hardware.
It can read an ONNX graph from a serialized ONNX model protobuf (ModelProto),
either directly from disk or from memory.

Some metadata must be supplied to augment the data present in the ONNX graph.

Example:

::

  import poponnx
  import torch.onnx
  import torchvision

  in = Variable(torch.randn(10, 3, 224, 224))
  model = torchvision.models.alexnet(pretrained=True)

  torch.onnx.export(model, in, "alexnet.onnx")

  # Create a runtime environment
  dataFeed = poponnx.DataFlow(500, 4, ["l1LossVal", "out", "image0"],
                              poponnx.AnchorReturnType.ALL)
  losses = [poponnx.L1Loss("out", "l1LossVal", 0.1)]
  optimizer = poponnx.ConstSGD(0.001)

  net = poponnx.Net("alexnet.onnx", dataFeed, losses, optimizer)


Building graphs without a third party framework
----------------------------------

There is a Builder class for constucting ONNX graphs without needing a third
party framework.

::

  import poponnx

  # Build a simple graph
  i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
  i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))

  o = builder.add(i1, i2)

  builder.addOutputTensor(o)

  proto = builder.getModelProto()


  # Create a runtime environment
  dataFeed = poponnx.DataFlow(500, 4, ["l1LossVal", "out", "image0"],
                              poponnx.AnchorReturnType.ALL)
  losses = [poponnx.L1Loss("out", "l1LossVal", 0.1)]
  optimizer = poponnx.ConstSGD(0.001)

  net = poponnx.Net(proto, dataFeed, losses, optimizer)

Executing graphs
--------------

The Net class runs graphs on the IPU hardware.

Data feeds can be from single python or numpy arrays, from python iterators
producing many tensors, and from specialized high-performance data feed objects.

Example

::

  TBD

