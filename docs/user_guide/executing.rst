.. _popart_executing:

Executing graphs
----------------

The ``Session`` class (:numref:`popart_importing`) is used to run graphs on an IPU.
Before the graph can be run, the way in which data will be transferred
to and from the IPU must be specified. Then, an IPU can be selected
to execute the graph.

Setting input and output data buffers for an execution
======================================================

This section describes how to set input and output data buffers for both C++ and Python.

.. tabs::

   .. group-tab:: **Python**

      When using Python, the :py:class:`~popart.PyStepIO` class is a convenient way of providing
      a session with input and output buffers. For both input and output, this
      class takes a dictionary with tensor names as keys and Python (or Numpy)
      arrays as values. PopART splits up these arrays internally to provide the
      ``Session`` object with the buffers that it needs.

      Note that the ``Session`` class has a convenience method, :py:func:`~popart.Session.initAnchorArrays`, that
      can create the output dictionary that :py:class:`~popart.PyStepIO` needs automatically.

      An alternative to :py:class:`~popart.PyStepIO` is the :py:class:`~popart.PyStepIOCallback` class, which
      you can use to implement :py:class:`~popart.PyStepIO` by means of a callback mechanism.

      The C++ equivalents of :py:class:`~popart.PyStepIO` and :py:class:`~popart.PyStepIOCallback` are
      :cpp:class:`~popart::StepIO` and :cpp:class:`~popart::StepIOCallback`, respectively.

      Below is an example of how to use :py:class:`~popart.PyStepIO`:

      .. code-block:: python

        # Create buffers to receive results from the execution
        anchors = session.initAnchorArrays()

        # Generate some random input data
        data_a = np.random.rand(1).astype(np.float32)
        data_b = np.random.rand(1).astype(np.float32)

        stepio = popart.PyStepIO({'a': data_a, 'b': data_b}, anchors)

        session.run(stepio)


      If there are any pre-defined inputs (such as weights or biases) in the
      graph then they will not be specified in the ``IStepIO`` object. However,
      before executing the graph, they will need to the copied to the device. If
      there are any optimiser-specific parameters which can be modified, then
      these must be written to the device. For example:

      .. code-block:: python

        session.weightsFromHost()


      These can also be updated between executions.

      .. code-block:: python

        # Update learning rate parameter between training steps
        stepLr = learningRate[step]
        session.updateOptimizerFromHost(popart.SGD(stepLr))


   .. group-tab:: **C++**

      Input and output data is passed to and from a :cpp:class:`~popart::Session` object via
      :cpp:class:`~popart::IStepIO` objects. Each call to :cpp:func:`Session.run() <popart::Session::run>` takes such an
      :cpp:class:`~popart::IStepIO` object. For every input tensor, the :cpp:class:`~popart::IStepIO` object
      contains a number of buffers that the session can read input data from,
      and for every anchor tensor, it contains a number of buffers to write
      output data to. Anchor tensors are described in detail in
      :numref:`retrieving_results`.

      The number and shape of these input and output buffers depend on a variety
      of factors including:

        1. The shape of the associated input and output tensor in the ONNX
           model.
        2. The :cpp:class:`~popart::DataFlow` configuration (:numref:`retrieving_results`) as
           passed to the constructor of the :cpp:class:`~popart::Session` object.
        3. The number of local replicas.
        4. The gradient accumulation factor.

      The :doc:`popart-cpp-api:index` contains more information about inputs (:cpp:class:`~popart::IStepIO` class) and outputs (:cpp:class:`~popart::DataFlow` class).

.. _retrieving_results:

Retrieving results
~~~~~~~~~~~~~~~~~~

The ``DataFlow`` class (:py:class:`Python <popart.DataFlow>`,
:cpp:class:`C++ <popart::DataFlow>`) describes how to execute the graph. When you construct
a ``DataFlow`` object, it expects two parameters, an integer (``batchesPerStep``) and a Python dictionary mapping tensor names to anchors:

.. code-block:: python

  df = popart.DataFlow(int, dict)

For example:

.. code-block:: python

  df = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})

``batchesPerStep`` is the the number of batches a call to ``Session.run(...)`` executes for before returning control to the caller. The second argument, the Python dictionary, contains keys that are the names of the tensors to retrieve from the model via the ``IStepIO`` object. These tensors are the anchor tensors. The associated values in the dictionary are defined by the ``AnchorReturnType`` class (:py:class:`Python <popart.AnchorReturnType>`,
:cpp:class:`C++ <popart::AnchorReturnType>`) and are one of:

* ``AnchorReturnTypeId.ALL``: return the tensor value for each
  iteration through the graph.
* ``AnchorReturnTypeId.EVERYN``: return the tensor value, but
  only for iterations which are divisible by ``N`` which is specified.
* ``AnchorReturnTypeId.FINAL``: return the tensor value on the
  final iteration through the graph.
* ``AnchorReturnTypeId.SUM``: return the sum of the values of
  the tensor from each iteration through the graph.

The effect of this setting on the number of output buffers is
explained in more detail in the C++ API documentation for the :cpp:class:`~popart::DataFlow` class.

Note that the set of tensors that are *anchored* may differ from those tensors
marked as ONNX model *outputs* (via ``Builder.addOutputTensor(...)``).
In other words, a model's output tensor need not be anchored and an anchored tensor need not be a model output -- any tensor can be anchored.
It is the anchored tensors that are considered to be an 'output' in the context of a ``IStepIO`` object.

Session options
===============

In this section we detail a number of selected session options. Refer to the C++ API reference document for information on all session options in the :cpp:class:`~popart::SessionOptions` class.

Stochastic rounding
~~~~~~~~~~~~~~~~~~~

You can enable
:ref:`stochastic rounding <ai-float-white-paper:deterministic versus stochastic rounding>` in PopART by setting the following session option:

.. code-block:: python

  opts = popart.SessionOptions()
  opts.enableStochasticRounding = True

.. note::
   Enabling stochastic rounding in PopART will result in the Poplar engine
   option ``target.deterministicWorkers`` being set to ``true`` (otherwise it
   will default to ``false``). You can override this engine option with
   the PopART session option ``SessionOptions.engineOptions``
   (:py:attr:`Python <popart.SessionOptions.engineOptions>`,
   :cpp:var:`C++ <popart::SessionOptions::engineOptions>`).

Selecting a device for execution
================================

The device manager allows the selection of an IPU configuration for executing
the session. The device must be passed into the ``Session`` class constructor.

.. code-block:: python

  df = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})
  device = popart.DeviceManager().createCpuDevice()
  s = popart.InferenceSession("onnx.pb", deviceInfo=device, dataFlow=df)

The ``DeviceManager`` class (:py:class:`Python <popart.DeviceManager>`,
:cpp:class:`C++ <popart::DeviceManager>`) can enumerate the available devices
with the ``enumerateDevices`` method (:py:func:`Python <popart.DeviceManager.enumerateDevices>`,
:cpp:func:`C++ <popart::DeviceManager::enumerateDevices>`). The ``acquireAvailableDevice`` method (:py:func:`Python <popart.DeviceManager.acquireAvailableDevice>`,
:cpp:func:`C++ <popart::DeviceManager::acquireAvailableDevice>`) will
acquire the next available device. The parameter specifies how many IPUs to
acquire.

.. code-block:: python

  # Acquire a two-IPU pair
  dev = popart.DeviceManager().acquireAvailableDevice(2)

Using ``acquireDeviceById`` (:py:func:`Python <popart.DeviceManager.acquireDeviceById>`,
:cpp:func:`C++ <popart::DeviceManager::acquireDeviceById>`) will select a device from the list
of IPU configurations based on its Id as returned by ``enumerateDevices``, or by the ``gc-info`` command-line tool. This may be a single IPU or a group of IPUs.

.. code-block:: python

  # Acquire IPU configuration 5
  dev = popart.DeviceManager().acquireDeviceById(5)

The method ``createIpuModelDevice`` (:py:func:`Python <popart.DeviceManager.createIpuModelDevice>`,
:cpp:func:`C++ <popart::DeviceManager::createIpuModelDevice>`) is used to create a Poplar software
emulation of an IPU.  Similarly, the method ``createCpuDevice`` (:py:func:`Python <popart.DeviceManager.createCpuDevice>`,
:cpp:func:`C++ <popart::DeviceManager::createCpuDevice>`) creates a simple
Poplar CPU backend.

By default the methods ``acquireAvailableDevice`` and ``acquireDeviceById``
will attach the device immediately to the running process. You can pass the
``DeviceConnectionType.OnDemand`` option (:py:func:`Python <popart.DeviceConnectionType.OnDemand>`,
:cpp:any:`C++ <popart::DeviceConnectionType::OnDemand>`) to the ``DeviceManager`` object to defer the
device attachment until it is required by PopART.

.. code-block:: python

  # Acquire four IPUs on demand
  connectionType=popart.DeviceConnectionType.OnDemand
  dev = popart.DeviceManager().acquireAvailableDevice(4, connectionType=connectionType)

Executing a session
===================

Once the device has been selected, the graph can be compiled for it, and
loaded into the device.  The ``prepareDevice`` method (:py:func:`Python <popart.TrainingSession.prepareDevice>`,
:cpp:func:`C++ <popart::Session::prepareDevice>`) in the ``Session`` class is used for this:

.. code-block:: python

  session.prepareDevice()

To execute the session you need to call the ``Session`` object's ``run`` method.

.. code-block:: python

  session.run(stepio)

If the session is created for inference, the user is responsible for ensuring
that the forward graph finishes with the appropriate operation for an inference.
If losses are provided to the inference session, the forward pass and the losses
will be executed, and the final loss value will be returned.


If the session was created for training, any pre-initialised parameters will be
updated to reflect the changes made to them by the optimiser.

Saving and loading a model
==========================

The ``Session`` class method ``modelToHost`` (:py:func:`Python <popart.TrainingSession.modelToHost>`,
:cpp:func:`C++ <popart::Session::modelToHost>`) writes a model with updated weights
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

There ``Session`` class contains two methods to access the profiling
information:

* ``getSummaryReport``
  (:py:func:`Python <popart.TrainingSession.getSummaryReport>`,
  :cpp:func:`C++ <popart::Session::getSummaryReport>`) retrieves a text summary
  of the compilation and execution of the graph.
* ``getReport`` (:py:func:`Python <popart.TrainingSession.getReport>`,
  :cpp:func:`C++ <popart::Session::getReport>`) returns a libpva ``Report``
  object containing details of the compilation and execution of the graph.

If profiling is not enabled, then the summary report will say 'Execution
profiling not enabled' and the report will contain no information on the
execution.

For more information on the libpva ``Report`` class, see the user guide and API
document:

* :doc:`libpva:index`
* :ref:`Libpva C++ API Reference <libpva:popvision analysis library c++ api>`
* :ref:`Libpva Python API Reference <libpva:popvision analysis library python api>`

For more information on profiling control and the information returned by these
methods, see the :ref:`poplar-user-guide:profiling` chapter of the
:doc:`poplar-user-guide:index`.

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

The default is ``OFF``. You can change this, and where the logging information is
written to, by setting environment variables as described in
:numref:`popart_env_vars`.

Programming interface
~~~~~~~~~~~~~~~~~~~~~

You can also control the logging level for each module in your program.

Examples are shown for Python and C++.

.. tabs::

   .. group-tab:: **Python**

    .. code-block:: python

      # Set all modules to DEBUG level
      popart.getLogger().setLevel("DEBUG")
      # Turn off logging for the session module
      popart.getLogger("session").setLevel("OFF")

   .. group-tab:: **C++**

    .. code-block:: C++

      // Set all modules to DEBUG level
      popart::logger::setLevel("popart", "DEBUG")
      // Turn off logging for the session module
      popart::logger::setLevel("session", "OFF")


Output format
~~~~~~~~~~~~~

The log information is output in the following format:

.. code-block:: none

  <timestamp> <namespace> <process_id>.<subprocess_id> <log_level_initial>: <log_message>

where ``<log_level_initial>`` has the following values:

========  ============================
Value     Description
========  ============================
T         Trace message
D         Debug message
I         Info message
W         Warn message
E         Error message
C         Critical messsage
========  ============================


An example of log information is:

.. code-block:: none

    2022-05-18T12:39:14.459868Z popart:devicex 90915.91186 D: [StepIOSplitter] [Gradient___input@out:0/1 - 0/1:1,0,0,0] Not yet able to call 'outComplete' on IStepIO
    2022-05-18T12:39:14.459971Z popart:session 90915.90915 T: Session::weightsToHost
    2022-05-18T12:39:14.459981Z popart:devicex 90915.90915 D: Writing weights to host


Errors
======

The full hierarchy of errors that can be thrown from a PopART Python program is:

.. code-block:: python

  popart_exception
    popart_internal_exception
    popart_runtime_error
  poplibs_exception
  poplar_exception
    poplar_runtime_error
      poplar_application_runtime_error
      poplar_system_runtime_error
        poplar_recoverable_runtime_error
        poplar_unrecoverable_runtime_error
        poplar_unknown_runtime_error

Application errors
~~~~~~~~~~~~~~~~~~

Application errors are thrown for a bug in either the user code or in the
framework.

.. code-block:: python

  popart.popart_exception
  popart.popart_internal_exception
  popart.popart_runtime_error
  popart.poplibs_exception
  popart.poplar_application_runtime_error

System errors
~~~~~~~~~~~~~

System errors are thrown by Poplar on IPU-Machines and Pod systems.

.. code-block:: python

  popart.poplar_recoverable_runtime_error
  popart.poplar_unrecoverable_runtime_error
  popart.poplar_unknown_runtime_error

A :py:exc:`popart.poplar_recoverable_runtime_error` system error has an attribute :py:attr:`popart.poplar_recoverable_runtime_error.recoveryAction`
which contains the action required to recover from this error. This will be one
of:

.. code-block:: python

  popart.RecoveryAction.IPU_RESET
  popart.RecoveryAction.PARTITION_RESET
  popart.RecoveryAction.POWER_CYCLE

If a :py:exc:`popart.poplar_unrecoverable_runtime_error` system error is thrown, you need to
contact `Graphcore Support <https://support.graphcore.ai>`__ because this issue
could either be an SDK bug or an issue with the hardware.

An :py:exc:`popart.poplar_unknown_runtime_error` system error could be either recoverable or
unrecoverable. In this instance, try the three recovery options (``IPU_RESET``,
``PARTITION_RESET``, ``POWER_CYCLE``). If none of the recovery options
resolve the issue, then contact `Graphcore Support
<https://support.graphcore.ai>`__.
