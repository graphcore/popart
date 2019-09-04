.. _popart_building:

Building graphs in PopART
-------------------------

PopART has a ``Builder`` class for constructing ONNX graphs without needing a third
party framework.

In the example below, a simple addition is prepared for execution. The steps involved are
described in the following sections and in :any:`popart_executing`.

.. code-block:: python

  import popart

  builder = popart.Builder()

  # Build a simple graph
  i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
  i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))

  o = builder.aiOnnx.add([i1, i2])

  builder.addOutputTensor(o)

  # Get the ONNX protobuf from the builder to pass to the Session
  proto = builder.getModelProto()

  # Create a runtime environment
  anchors = {o : popart.AnchorReturnType("ALL")}
  dataFeed = popart.DataFlow(1, anchors)
  device = popart.DeviceManager().createCpuDevice()

  # Create the session from the graph, data feed and device information
  session = popart.InferenceSession(proto, dataFeed, device)

The DataFlow object is described in more detail in :any:`popart_executing`.

Adding operations to the graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The builder adds operations to the graph by calling one of the many
operation methods.  Each of these methods has a common signature.
For example, ``relu`` will add an ONNX Relu operation
to the graph:

.. code-block:: python

  output = builder.aiOnnx.relu([input], "debug-name")

They take a list of arguments which are the input tensor names, and an optional
string to assign to the node. This name is passed to the Poplar nodes and used
in debugging and profiling reports.

The operation method returns the name of the tensor that is an output of the newly added node.

In some cases other arguments are required, for instance:

.. code-block:: python

  output = builder.aiOnnx.gather(['input', 'indices'], axis=1, debugPrefix="My-Gather")

Adding parameters to the graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters, for instance the weights of a convolution, are represented as
initialised inputs to the graph.  They can be added with the
``addInitializedInputTensor`` method:

.. code-block:: python

  w_data = np.random.rand(64, 4, 3, 3).astype(np.float16)
  w1 = builder.addInitializedInputTensor(w_data)

Setting outputs
~~~~~~~~~~~~~~~

The outputs of the graph should be marked appropriately, using the
``addOutputTensor`` method:

.. code-block:: python

  builder.addOutputTensor(output)

Setting the IPU number for operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When creating a graph which will run on a multiple IPU system, nodes need
to be marked with an annotation to describe which IPU they will run upon.

For instance, to place a specific convolution onto IPU 1:

.. code-block:: python

  we = builder.addInitializedInputTensor(np.zeros([32, 4, 3, 3], np.float16))
  bi = builder.addInitializedInputTensor(np.zeros([32], np.float16))
  o = builder.aiOnnx.conv([x, we, bi],
                          dilations=[1, 1],
                          pads=[1, 1, 1, 1],
                          strides=[1, 1])
  # place operation on IPU 1
  builder.virtualGraph(o, 1)


A context manager is available for placing multiple operations together onto a
specific IPU:

.. code-block:: python

  builder = popart.Builder()

  i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
  i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
  i3 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
  i4 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))

  # place two add operations on IPU 0
  with builder.virtualGraph(0):
      o1 = builder.aiOnnx.add([i1, i2])
      o2 = builder.aiOnnx.add([i3, i4])

  # place one add operation on IPU 1
  with builder.virtualGraph(1):
      o = builder.aiOnnx.add([o1, o2])

Alternatively, for automatic placement of nodes on available IPUs, use the
session option ``autoVirtualGraph``.  See ``SessionOptions`` in the PopART API document.
