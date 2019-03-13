Importing graphs
----------------

The `Session` class is the runtime environment for executing graphs on the IPU
hardware. It can read an ONNX graph from a serialized ONNX model protobuf
(ModelProto), either directly from disk or from memory. A `Session` can either be 
constructed as an `InferenceSession` or `TrainingSession`

Some metadata must be supplied to augment the data present in the ONNX graph.

In this example of importing a graph for inference, the `torchvision` package
is used to create a pre-trained AlexNet graph, with a 4x3x244x244 input. The
`torchvision` graph has an ONNX output called `out`, and the `DataFlow` object
contains an entry to fetch that anchor.

::

  import poponnx
  import torch.onnx
  import torchvision

  input_ = torch.FloatTensor(torch.randn(4, 3, 224, 224))
  model = torchvision.models.alexnet(pretrained=True)

  torch.onnx.export(model, input_, "alexnet.onnx")

  # Create a runtime environment
  anchors = {"output" : poponnx.AnchorReturnType("ALL")}
  dataFeed = poponnx.DataFlow(100, anchors)

  session = poponnx.InferenceSession("alexnet.onnx", dataFeed)


The `Session` class takes the name of a protobuf file, or the protobuf
itself.  It also takes a `DataFlow` object which has some information about
how to execute the graph; the number of times to repeat the graph in one
execution of the backend, and the names of the tensors in the graph to return
to the user.

Other parameters to the `Session` object describe the types of loss to apply to
the network and the optimizer to use, for when the user wishes to train the
network instead of performing inference.

::

  import poponnx
  import torch.onnx
  import torchvision

  input_ = torch.randn(4, 3, 224, 224)
  model = torchvision.models.alexnet(pretrained=False)

  torch.onnx.export(model, input_, "alexnet.onnx")

  # Create a runtime environment
  anchors = {"output" : poponnx.AnchorReturnType("ALL")}
  dataFeed = poponnx.DataFlow(100, anchors)

  losses = [poponnx.NllLoss("output", "labels", "loss")]
  optimizer = poponnx.ConstSGD(0.001)

  # We need to describe the labels input shape
  inputShapeInfo = poponnx.InputShapeInfo()
  inputShapeInfo.add("labels", poponnx.TensorInfo("INT32", [4]))

  session = poponnx.TrainingSession("alexnet.onnx",
                                    dataFeed=dataFeed,
                                    losses=losses,
                                    optimizer=optimizer,
                                    inputShapeInfo=inputShapeInfo)

In this case, when the `Session` object is asked to train the graph, an `NllLoss`
node will be added to the end of the graph, and a `ConstSGD` optimizer will
be used to optimize the parameters in the network.

