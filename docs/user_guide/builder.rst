Building graphs without a third party framework
-----------------------------------------------

There is a `Builder` class for constructing ONNX graphs without needing a third
party framework.

In this example, a simple addition is prepared for execution.

::

  import poponnx

  builder = poponnx.Builder()

  # Build a simple graph
  i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
  i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))

  o = builder.aiOnnx.add([i1, i2])

  builder.addOutputTensor(o)

  proto = builder.getModelProto()

  # Create a runtime environment
  anchors = {o : poponnx.AnchorReturnType("ALL")}
  dataFeed = poponnx.DataFlow(1, anchors)
  device = poponnx.DeviceManager().createCpuDevice()

  session = poponnx.InferenceSession(proto, dataFeed, device)

Adding operations to the graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The builder adds operations to the graph by calling one of the many
operation methods.  For instance `acos` with add an arc-cosine ONNX operation
to the graph.  Each of these methods follows a common signature, for
instance:

::

  output = builder.aiOnnx.acos([input], "debug-name")

They take a list of arguments which are the input tensor names, and an optional
string to assign to the node.  This name is passed to the Poplar nodes.  It returns
the name of the tensor which is an output of the newly added node.

In some cases other arguments are required, for instance:

::

  output = builder.aiOnnx.gather(['input', 'indices'], axis=1, debugPrefix="My-Gather")

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

  builder.addOutputTensor(output)

Setting the IPU number for operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When creating a graph which will run on a multiple IPU system, nodes need
to be marked with an annotation to describe which IPU they will run upon.

For instance, to place a specific convolution onto IPU 1:

::

  we = builder.addInitializedInputTensor(np.zeros([32, 4, 3, 3], np.float16))
  bi = builder.addInitializedInputTensor(np.zeros([32], np.float16))
  o = builder.aiOnnx.conv([x, we, bi], [1, 1], [1, 1, 1, 1], [1, 1])
  builder.virtualGraph(o, 1)


A context manager is available for placing multiple operations onto a
specific IPU together:

::

  builder = poponnx.Builder()

  i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
  i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
  i3 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
  i4 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))

  with builder.virtualGraph(0):
      o1 = builder.aiOnnx.add([i1, i2])
      o2 = builder.aiOnnx.add([i3, i4])

  with builder.virtualGraph(1):
      o = builder.aiOnnx.add([o1, o2])

Alternatively, for automatic placement of nodes on available IPUs, use the
session option `autoVirtualGraph`.  See `SessionOptions`.
