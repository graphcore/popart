Executing graphs
----------------

The Session class runs graphs on an IPU device.  The constructor of the
session gives the ONNX protobuf or file, and several other parameters to
configure the session as a whole.

Constructing the session
========================

The session constructor takes at least the ONNX graph parameter, and a
parameter `dataFlow` which directs the basic data flow in the graph.

::

  df = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})
  s = poponnx.Session("onnx.pb", dataFlow=df)

Other parameters are required for constructing a session to train a
graph, or to control more specific features of the compilation.

Training specific parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a network is being prepared for training, one or more loss function
nodes will be appended to the graph.  These are provided by the `losses`
parameter.  This is a list of loss operations.

When training, an optimization algorithm is also required.  This is provided
by the `optimizer` parameter.

Session control options
~~~~~~~~~~~~~~~~~~~~~~~

In some ONNX graphs, the input tensors may not have their sizes specified.
In this case, the `inputShapeInfo` parameter can be used to specify the
input shapes.  The Poplar framework uses statically allocated memory buffers
and so it needs to know the size of tensors before the compilation.

The `userOptions` parameter can pass a set of session control options,
as described in the API reference documentation.

The `patterns` parameter allows the user to select a set of graph transformation
patterns which will be applied to the graph.  Without this parameter, a default
set of optimization transformations will be applied.

Selecting a device for execution
================================

The device manager allows the selection of an IPU configuration for the execution.
The `setDevice` method is used to set the device within the session.

::

  session.setDevice(poponnx.DeviceManager().createIpuModelDevice({}))

The device manager can enumerate the available devices with the `enumerateDevices`
method. With no parameters the  `acquireAvailableDevice` method will acquire the
next available device.  With one parameter, it will select a device from the list
of IPU configurations, as given by the enumerate function, or by the `gc-info`
application.

With two parameters, it will select the first available configuration with a
given number of IPUs and a given number of tiles per IPU.

::

  # Acquire IPU configuration 5
  dev = poponnx.DeviceManager().acquireAvailableDevice(5)

::

  # Acquire a 2 IPU pair
  dev = poponnx.DeviceManager().acquireAvailableDevice(2, 1216)

The method `createIpuModelDevice` is used to create a Poplar software emulation
of an IPU device.  See the API documentation for details.  Similarly, the method
`createCpuDevice` creates a simple Poplar CPU backend.


Compiling the graph and preparing the hardware for execution
============================================================

Once the device has been selected, the graph can be compiled for it, and
loaded into the hardware.  The `prepareDevice` function is used:

::

  session.prepareDevice()


If there are any pre-defined inputs (weights, biases, ...) in the graph
then they will not be specified in the PyStepIO object.  However, before
executing the graph, they will need to the copied to the hardware.

If there are any optimizer specific parameters which can be modified,
then these must be written to the device.

::

  session.weightsFromHost()
  session.optimizerFromHost()

They can also be updated between executions.

::

  # Update learning rate parameter between training steps
  stepLr = learningRate[step]
  session.updateOptimizer(poponnx.SGD(stepLr))
  session.optimizerFromHost()

Executing a session
===================

Setting input/output data buffers for an execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `PyStepIO` class indicates input data for a specific execution.  It
takes a dictionary with the input tensor names as keys, and the python
arrays as data values.  It also takes a similar dictionary of names and
buffers for the output values.

A convenience function `initAnchorArrays` can create the output buffers
and map for the user, given the anchors (output nodes) which were
specified in the `dataFlow` object during session construction.

::

  # Create buffers to receive results from the execution
  anchors = session.initAnchorArrays()

  # Generate some random input data
  data_a = np.random.rand(1).astype(np.float32)
  data_b = np.random.rand(1).astype(np.float32)

  stepio = poponnx.PyStepIO({a: data_a, b: data_b}, anchors)


Inference
~~~~~~~~~

In inference, only the forward pass will be executed. The user is
responsible for ensuring that the forward graph finishes with the appropriate
operation for an inference.

::

  session.infer(stepio)


Evaluation
~~~~~~~~~~

In evaluation, the forward pass and the losses will be executed, and the
final loss value will be returned.

::

  session.evaluate(stepio)

Training
~~~~~~~~

In training, a full forward pass, loss calculation and backward pass will be
done.  Any pre-initialized parameters will be updated to reflect any changes
to them which the optimizer has made.

::

  session.train(stepio)


Fetching the trained parameters
===============================

The method `modelToHost` returns a model with updated weights.

::

  trained_model = session.modelToHost()


Retrieving poplar compilation and execution reports
===================================================

Poplar can provide JSON format reports on the compilation and execution of
the graphs.

`getSummaryReport` retrieves a text report of the compilation and execution of
the graph.  `getGraphReport` returns a JSON format report on the compilation of
the graph and `getExecutionReport` returns a JSON format report on all executions
of the graph since the last report was fetched.



