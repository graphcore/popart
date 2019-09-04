.. _popart_executing:

Executing graphs
----------------

The ``Session`` class is used to run graphs on an IPU device.
Before the graph can be run, the way in which data will be transferred
to and from the IPU must be specified. Then an IPU device can be selected
to execute the graph.

Setting input/output data buffers for an execution
==================================================

The ``PyStepIO`` class defines the input data for a specific execution.  It
takes a dictionary with the input tensor names as keys, and Python
arrays for the data values.  It also takes a similar dictionary of names and
buffers for the output values.

A convenience method ``initAnchorArrays`` can create the output buffers
and dictionary for you, given the anchors (output nodes) which were
specified in the ``DataFlow`` object during session construction.

.. code-block:: python

  # Create buffers to receive results from the execution
  anchors = session.initAnchorArrays()

  # Generate some random input data
  data_a = np.random.rand(1).astype(np.float32)
  data_b = np.random.rand(1).astype(np.float32)

  stepio = popart.PyStepIO({'a': data_a, 'b': data_b}, anchors)


.. TODO: Add something about the pytorch data feeder.


If there are any pre-defined inputs (weights, biases, etc.) in the graph
then they will not be specified in the ``PyStepIO`` object.  However, before
executing the graph, they will need to the copied to the hardware.
If there are any optimiser-specific parameters which can be modified,
then these must be written to the device. For example:

.. code-block:: python

  session.weightsFromHost()
  session.optimizerFromHost()

These can also be updated between executions.

.. code-block:: python

  # Update learning rate parameter between training steps
  stepLr = learningRate[step]
  session.updateOptimizer(popart.SGD(stepLr))
  session.optimizerFromHost()

Retrieving results
~~~~~~~~~~~~~~~~~~

The ``DataFlow`` class describes how to execute the graph.  The second parameter is
a description of the anchors, the results to fetch from the graph.

.. code-block:: python

  df = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})

This is a Python dictionary with keys that are the names of the tensors to retrieve
from the model. The associated values are an ``AnchorReturnType``, which is one of:

* ``popart.AnchorReturnType("ALL")``: a vector of results is returned, one for each
  iteration of the graph.
* ``popart.AnchorReturnType("EVERYN", N)``: a vector containing the tensor, but
  only for iterations which are divisible by ``N``.
* ``popart.AnchorReturnType("FINAL")``: the value of the tensor on the final
  iteration through the graph.

Selecting a device for execution
================================

The device manager allows the selection of an IPU configuration for executing the session.
The device must be passed into the session constructor.

.. code-block:: python

  df = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})
  device = popart.DeviceManager().createCpuDevice()
  s = popart.InferenceSession("onnx.pb", deviceInfo=device, dataFeed=df)

The device manager can enumerate the available devices with the ``enumerateDevices``
method. The  ``acquireAvailableDevice`` method will acquire the
next available device. The first parameter specifies how many IPUs to acquire.

.. code-block:: python

  # Acquire a two-IPU pair
  dev = popart.DeviceManager().acquireAvailableDevice(2)

Using ``acquireDeviceById`` will select a device from the list
of IPU configurations, as given by the ``enumerateDevices`` method, or by the ``gc-info``
command-line tool. This may be a single IPU or a group of IPUs.

.. code-block:: python

  # Acquire IPU configuration 5
  dev = popart.DeviceManager().acquireDeviceById(5)

The method ``createIpuModelDevice`` is used to create a Poplar software emulation
of an IPU device.  Similarly, the method ``createCpuDevice`` creates a simple Poplar CPU backend.
See the API documentation for details.


Executing a session
===================

Once the device has been selected, the graph can be compiled for it, and
loaded into the hardware.  The ``prepareDevice`` method is used for this:

.. code-block:: python

  session.prepareDevice()

To execute the session you need to call the session's ``run`` method.

.. code-block:: python

  session.run(stepio)


If the session is created for inference, the user is responsible for ensuring
that the forward graph finishes with the appropriate operation for an inference.
If losses are provided to the inference session the forward pass and the losses
will be executed, and the final loss value will be returned.


If the session was created for training, any pre-initialised parameters will be
updated to reflect the changes made to them by the optimiser.

Saving and loading a model
==========================

The method ``modelToHost`` writes a model with updated weights
to the specified file.

.. code-block:: python

  session.modelToHost("trained_model.onnx")

A file of saved parameters, for example from an earlier execution session, can
be loaded into the current session.

.. code-block:: python

  session.resetHostWeights("test.onnx")
  session.weightsFromHost()


Retrieving profiling reports
============================

Poplar can provide profiling information on the compilation and execution of
the graphs. There are three method functions of the session object for this:

* ``getSummaryReport`` retrieves a text summary of the compilation and execution of
the graph.
* ``getGraphReport`` returns a JSON format report on the compilation of
the graph
* ``getExecutionReport`` returns a JSON format report on all executions
of the graph since the last report was fetched.

Both ``getGraphReport`` and ``getExecutionReport`` can optionally return
a Concise Binary Object Representation (CBOR) formatted report.

For more information on the information returned by these functions, see the Profiling chapter of the *Poplar and Poplibs User Guide*.

Turning on execution tracing
============================

PopART contains an internal logging system that can show the progress of graph
compilation and execution.  It can be turned on by calling the ``Logger`` class.
For example:

.. code-block:: python

  popart.getLogger().setLevel("TRACE")

The logging levels, in decreasing verbosity, are:

* TRACE
* DEBUG
* INFO
* WARN
* ERR
* CRITICAL
* OFF
