Importing graphs
----------------

The Session class is the runtime environment for executing graphs on the IPU
hardware. It can read an ONNX graph from a serialized ONNX model protobuf
(ModelProto), either directly from disk or from memory.

Some metadata must be supplied to augment the data present in the ONNX graph.

Example of importing a graoh for inference:

::

  import poponnx
  import torch.onnx
  import torchvision

  in = Variable(torch.randn(10, 3, 224, 224))
  model = torchvision.models.alexnet(pretrained=True)

  torch.onnx.export(model, in, "alexnet.onnx")

  # Create a runtime environment
  anchors = {"out" : poponnx.AnchorReturnType("ALL")}
  dataFeed = poponnx.DataFlow(100, anchors)

  session = poponnx.Session("alexnet.onnx", dataFeed)

The Session class takes the name of a protobuf file, or the protobuf
itself.  It also takes a DataFlow object which has some information about
how to execute the graph; the number of times to execute the graph in one
execution of the backend, and the names of the tensors in the graph to return
to the user.

Other parameters to the Session object describe the types of loss to apply to
the network, and the optimizer to use, for when the user wishes to train the
network instead of performing inference.

::

  import poponnx
  import torch.onnx
  import torchvision

  in = Variable(torch.randn(10, 3, 224, 224))
  model = torchvision.models.alexnet(pretrained=True)

  torch.onnx.export(model, in, "alexnet.onnx")

  # Create a runtime environment
  anchors = {"out" : poponnx.AnchorReturnType("ALL")}
  dataFeed = poponnx.DataFlow(100, anchors)

  session = poponnx.Session("alexnet.onnx", dataFeed)

