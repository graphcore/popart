.. _popart_executing:

Executing graphs
----------------

The ``Session`` class is used to run graphs on an IPU device.
Before the graph can be run, the way in which data will be transferred
to and from the IPU must be specified. Then an IPU device can be selected
to execute the graph.

Setting input/output data buffers for an execution
==================================================

Input and output data is passed to and from a ``Session`` object via ``IStepIO``
objects. Each call to ``session.run(...)`` takes such a ``IStepIO`` object.
For every input tensor, this object contains a number of buffers that the
session can read input data from. And for every anchored tensor, it contains a
number of buffers to write output data to. There is more information about
anchors in :numref:`retrieving_results`.

The number and shape of these buffers depend on a variety of factors including

1) the shape of associated tensor in the ONNX model
2) | the ``DataFlow`` configuration (see next section) as passed to the
     ``Session`` object's constructor
3) the number of local replicas, and
4) | the accumulation factor.

This is explained in more detail in the
`C++ API <https://docs.graphcore.ai/projects/popart-cpp-api/>`_ documentation
under the ``IStepIO`` class (for inputs) and under the ``DataFlow`` class (for
outputs).

When using Python, the ``PyStepIO`` class is a convenient way of providing a
session with input and output buffers. For both input and output, this class
takes a dictionary with tensor names as keys and Python (or Numpy) arrays as
values. PopART splits up these arrays internally to provide the ``Session``
object with the buffers that it needs.

Note that ``Session`` has a convenience method, ``initAnchorArrays``,
that can create the output dictionary that ``PyStepIO`` needs automatically.

An alternative to ``PyStepIO`` is the
``PyStepIOCallback`` class, which you can use to implement ``IStepIO`` by means
of a callback mechanism.

The C++ equivalents of ``PyStepIO`` and ``PyStepIOCallback`` are ``StepIO`` and
``StepIOCallback``, respectively.

Below is an example of how to use ``PyStepIO``:

.. code-block:: python

  # Create buffers to receive results from the execution
  anchors = session.initAnchorArrays()

  # Generate some random input data
  data_a = np.random.rand(1).astype(np.float32)
  data_b = np.random.rand(1).astype(np.float32)

  stepio = popart.PyStepIO({'a': data_a, 'b': data_b}, anchors)

  session.run(stepio)


.. TODO: Add something about the pytorch data feeder.

If there are any pre-defined inputs (such as weights or biases) in the graph
then they will not be specified in the ``IStepIO`` object. However, before
executing the graph, they will need to the copied to the hardware.
If there are any optimiser-specific parameters which can be modified,
then these must be written to the device. For example:

.. code-block:: python

  session.weightsFromHost()


These can also be updated between executions.

.. code-block:: python

  # Update learning rate parameter between training steps
  stepLr = learningRate[step]
  session.updateOptimizerFromHost(popart.SGD(stepLr))


.. _retrieving_results:

Retrieving results
~~~~~~~~~~~~~~~~~~

The ``DataFlow`` class describes how to execute the graph. When you construct
a ``DataFlow`` class it expects two parameters:

.. code-block:: python

  df = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})

The first argument is ``batchesPerStep``. This is the the number of
batches a call to ``session.run(...)`` executes for before returning control to
the caller.

The second argument is a Python dictionary with keys that are the names of the
tensors to retrieve from the model via the ``IStepIO`` object. We call such
tensors *anchors*. The associated values are ``AnchorReturnType`` values, which
are one of:

* ``popart.AnchorReturnType("ALL")``: a vector of results is returned, one for each
  iteration of the graph.
* ``popart.AnchorReturnType("EVERYN", N)``: a vector containing the tensor, but
  only for iterations which are divisible by ``N``.
* ``popart.AnchorReturnType("FINAL")``: the value of the tensor on the final
  iteration through the graph.
* ``popart.AnchorReturnType("SUM")``: the sum of the values of the tensor
  from each iteration through the graph.

The effect of this setting on the number of output buffers is
explained in more detail in our `C++ API
<https://docs.graphcore.ai/projects/popart-cpp-api/>`_ documentation
documentation (see documentation for the ``DataFlow`` class).

Note that the set of tensors that are *anchored* may differ from those tensors
marked as ONNX model *outputs* (via ``builder.addOutputTensor(...)``).
That is, a model's output tensor need not be anchored and an anchored tensor
need not be a model output -- any tensor can be anchored.
It is the anchored tensors that are considered 'output' in the context of a
``IStepIO`` object.

Selecting a device for execution
================================

The device manager allows the selection of an IPU configuration for executing the session.
The device must be passed into the session constructor.

.. code-block:: python

  df = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})
  device = popart.DeviceManager().createCpuDevice()
  s = popart.InferenceSession("onnx.pb", deviceInfo=device, dataFlow=df)

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
See the `PopART C++ API Reference
<https://www.graphcore.ai/docs/popart-c-api-reference>`_ for details.

By default the functions ``acquireAvailableDevice`` and ``acquireDeviceById``
will attach the device immediately to the running process. You can pass the
``DeviceConnectionType.OnDemand`` option to the ``DeviceManager`` to defer the
device attachment until it is required by PopART.

.. code-block:: python

  # Acquire four IPUs on demand
  connectionType=popart.DeviceConnectionType.OnDemand
  dev = popart.DeviceManager().acquireAvailableDevice(4, connectionType=connectionType)

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

Note that if you plan to run your program in multiple processes simultaneously,
you should avoid possible race conditions by writing to different files, for
example by using temporary files.

A file of saved parameters, for example from an earlier execution session, can
be loaded into the current session.

.. code-block:: python

  session.resetHostWeights("test.onnx")
  session.weightsFromHost()


.. _popart_profiling:

Retrieving profiling reports
============================

Poplar can provide profiling information on the compilation and execution of
the graph. Profiling is not enabled by default.

To get profiling reports in PopART, you will need to enable profiling in the
Poplar engine. For example:

.. code-block:: python

  opts = popart.SessionOptions()
  opts.engineOptions = {"autoReport.all": "true"}

You can also control what information is included in the profiling report:

.. code-block:: python

  opts.reportOptions = {"showExecutionSteps": "true"}

There are two method functions of the session object to access the profiling
information:

* ``getSummaryReport`` retrieves a text summary of the compilation and execution of
  the graph.
* ``getReport`` returns a libpva `Report` object containing details of the
  compliation and execution of the graph.
  the graph

If profiling is not enabled, then the summary report will say 'Execution profiling not enabled'
and the report will contain no information in the execution.

For more information on the libpva Report, see the pva user guide and api document:
* `Libpva User Guide <https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/index.html>`_
* `Libpva C++ API Reference <https://docs.graphcore.ai/projects/poplar-api/en/latest/pva.html>`_
* `Libpva Python API Reference <https://docs.graphcore.ai/projects/poplar-api/en/latest/pva-python.html>`_.

For more information on profiling control and the information returned by these functions, see the
Profiling chapter of the
`Poplar and PopLibs User Guide
<https://www.graphcore.ai/docs/poplar-and-poplibs-user-guide>`_.

.. _popart_logging:

Turning on execution tracing
============================

PopART contains an internal logging system that can show the progress of graph
compilation and execution.

Logging information is generated from the following modules:

=========   =================================
popart      Generic PopART module, if no module specified
session     The ONNX session (the PopART API)
ir          The intermediate representation
devicex     The Poplar backend
transform   The transform module
pattern     The pattern module
builder     The builder module
op          The op module
opx         The opx module
ces         The constant expression module
python      The Python module
none        An unidentified module
=========   =================================


The logging levels, in decreasing verbosity, are shown below.

========  ============================
TRACE     The highest level, shows the
          order of method calls
DEBUG
INFO
WARN      Warnings
ERR       Errors
CRITICAL  Only critical errors
OFF       No logging
========  ============================

The default is "OFF". You can change this, and where the logging information is written to,
by setting environment variables, see :any:`popart_env_vars`.

Programming interface
~~~~~~~~~~~~~~~~~~~~~

You can also control the logging level for each module in your program.

For example, in Python:

.. code-block:: python

  # Set all modules to DEBUG level
  popart.getLogger().setLevel("DEBUG")
  # Turn off logging for the session module
  popart.getLogger("session").setLevel("OFF")

And in C++:

.. code-block:: C++

  // Set all modules to DEBUG level
  popart::logger::setLevel("popart", "DEBUG")
  // Turn off logging for the session module
  popart::logger::setLevel("session", "OFF")


Output format
~~~~~~~~~~~~~

The information is output in the following format:

.. code-block:: none

  [<timestamp>] [<module>] [<level>] <logging string>

For example:

.. code-block:: none

  [2019-10-16 13:55:05.359] [popart:devicex] [debug] Creating poplar::Tensor 1
  [2019-10-16 13:55:05.359] [popart:devicex] [debug] Creating host-to-device FIFO 1
  [2019-10-16 13:55:05.359] [popart:devicex] [debug] Creating device-to-host FIFO 1
