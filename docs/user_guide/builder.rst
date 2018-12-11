Building graphs without a third party framework
----------------------------------

There is a Builder class for constucting ONNX graphs without needing a third
party framework.

In this example, a simple addition is prepared for execution.

::

  import poponnx

  # Build a simple graph
  i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
  i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))

  o = builder.add(i1, i2)

  builder.addOutputTensor(o)

  proto = builder.getModelProto()

  # Create a runtime environment
  anchors = {o : poponnx.AnchorReturnType("ALL")}
  dataFeed = poponnx.DataFlow(500, 4, anchors)

  session = poponnx.Session(proto, dataFeed)

