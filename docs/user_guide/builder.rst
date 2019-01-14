Building graphs without a third party framework
-----------------------------------------------

There is a Builder class for constructing ONNX graphs without needing a third
party framework.

In this example, a simple addition is prepared for execution.

::

  import poponnx

  builder = poponnx.Builder()

  # Build a simple graph
  i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
  i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))

  o = builder.add([i1, i2])

  builder.addOutputTensor(o)

  proto = builder.getModelProto()

  # Create a runtime environment
  anchors = {o : poponnx.AnchorReturnType("ALL")}
  dataFeed = poponnx.DataFlow(1, anchors)

  session = poponnx.Session(proto, dataFeed)

Adding operations to the graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The builder adds operations to the graph by calling one of the many
operation methods.  For instance `acos` with add an arc-cosine ONNX operation
to the graph.  Each of these methods follows a common signature, for
instance:

::

  out = builder.acos([in], "debug-name")

They take a list of arguments which are the input tensor names, and an optional
to assign to the node.  This name is passed to the poplar nodes.  It returns
the name of the tensor which is an output of the newly added node.

In some cases other arguments are required, for instance:

::

  out = builder.gather([input, indices], axis=[1], debugPrefix="My-Gather")

Adding parameters to the graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters, for instance the weights of a convolution, are represented as
initialized inputs to the graph.  They can be added with the
`addInitializedInputTensor` method:

::

  w_data = np.random.rand(64, 4, 3, 3).astype(np.float16)
  w1 = builder.addInitializedInputTensor(w_data)

Setting outputs
~~~~~~~~~~~~~~~

The outputs of the graph should be marked appropriately, using the
`addOutputTensor` method:

::

  builder.addOutputTensor(o)

Setting the IPU number for operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When creating a graph which will run on a multiple IPU system, nodes need
to be marked with an annotation to describe which IPU they will run upon.

For instance, to place a specific convolution onto IPU 1:

::

  we = b.addInitializedInputTensor(np.zeros([32, 4, 3, 3], np.float16))
  bi = b.addInitializedInputTensor(np.zeros([32], np.float16))
  o = b.convolution([x, we, bi], [1, 1], [1, 1, 1, 1], [1, 1])
  b.virtualGraph(o, 1)

A context manager is available for placing mutiple operations onto a
specific IPU together:

::

  builder = poponnx.Builder()

  i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
  i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
  i3 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
  i4 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))

  with builder.virtualGraph(0):
    o1 = builder.add([i1, i2])
    o2 = builder.add([i3, i4])

  with builder.virtualGraph(1):
    o = builder.add([o1, o2])


