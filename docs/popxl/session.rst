.. _sec_session:

Session
=======

The :py:class:`~popxl.Session` class represents the PopART runtime
session and lets you execute a PopXL graph of operations. You create a session
as follows:

.. literalinclude:: files/tensor_addition.py
  :language: python
  :name: session_construction_code
  :caption: Example of Session construction
  :start-after: Session begin
  :end-before: Session end

.. only:: html

    :download:`Download tensor_addition.py <files/tensor_addition.py>`

where ``ir`` is the :py:class:`~popxl.Ir` object you have created.

.. warning::
     The session takes ownership of the :py:class:`~popxl.Ir` object from
     creation onwards;  the ``Ir`` object cannot be changed after this point.

The session will prepare and compile the IR that your Python
:py:class:`~popxl.Ir` object represents. A :py:class:`~popart.popart_exception`
error will be thrown if the ``Ir`` object is not valid. This can happen if there
are cycles in the graph or if the configuration for the device(s) you have
specified is invalid.


.. _running_a_session:

Running a session
-----------------

At this point, the ``Ir`` is compiled, but we must perform some more steps
before executing it on device:
  * Attach to the device
  * Initialise the weight tensors on device (from the values stored on host when building the ``Ir``)
  * Connect the :py:class:`popxl.HostToDeviceStream`s (i.e. the inputs of your model) to buffers of the appropriate size, shape and dtype

:py:class:`popxl.Session` should be used as a context manager to achieve the first two steps; it will attach to the device and load the weights from host onto device.
Then, to execute the ``Ir`` on given input data, call :py:func:`popxl.Session.run` inside the context, passing a ``Mapping`` from :py:class:`popxl.HostToDeviceStream` to the relevant input data buffers.
Note, Calling :py:func:`popxl.Session.run` before attaching to the device will result in an error.
Finally, on exiting the context, the weights will be loaded from device back onto host, and the session will detach from the device.

.. literalinclude:: files/simple_addition.py
  :language: python
  :name: session_run_1
  :caption: Example of running with :py:func:`popxl.Session.run`
  :start-after: SessionRun begin
  :end-before: SessionRun end
Once you have constructed your session, you can run your model with the relevant inputs to return your outputs. You can do this in two ways:

1. :py:func:`outputs = session.run(inputs, device_desc) <popxl.Session.run>`.

  This runs the session with ``inputs`` and constructs ``outputs``
  in the form of NumPy ``ndarray`` objects to return to the caller. Input shapes
  will be validated and outputs will be returned in the shape inferred by the
  IR. ``device_desc`` is a string describing the type of device you will use to
  run the session as described in :numref:`device-types`.

.. note::
    It is important that the context manager keeps the host weights in sync with
    the device, as attach-detach-attach-ing can invalidate the weights on
    device. This is because Poplar may zero the memory on attach/detach, or
    another process used the IPU whilst you were detached and overwrote that
    memory.

:py:func:`popxl.Session.run` will validate that all the required input streams have been passed, and that the input buffers are of correct shape.
See :numref:`sec_session_inputs` for what the shapes should be.
It will also internally create the output buffers for you, as a ``Mapping`` from :py:class:`popxl.DeviceToHostStream` to :py:class:`np.ndarray`.
The correct shape and dtype will be inferred from the ``Ir``.

Alternatively, you can create the output buffers yourself and pass them to :py:func:`popxl.Session.run_with_outputs` to fill in for you. The
``Mapping`` you pass will be validated similarly to the inputs.
2. :py:func:`popxl.Session.run_with_outputs <run_with_outputs(inputs, outputs, device_desc)>`

  If you want to write to part of a larger array, or you already have output
  arrays constructed, use :py:func:`~popxl.Session.run_with_outputs`. You must
  first construct the output arrays with the necessary shape, then pass them to
  the session. The session runs the model and writes the values to the output
  arrays. The shapes of the output arrays will be validated against the inferred
  shape and an error is thrown if the shapes do not correspond.

  .. literalinclude:: files/simple_addition.py
    :language: python
    :name: session_run_2
    :caption: Example of running with :py:func:`popxl.Session.run_with_outputs`
    :start-after: SessionRun2 begin
    :end-before: SessionRun2 end

Finally, there is also :py:func:`popxl.Session.create_host_outputs` that will create the ``Mapping`` for you, with each stream mapped to an empty :py:class:`np.ndarray`.
This is the method used internally in :py:func:`popxl.Session.run` and provides a shortcut to constructing the output arrays required for :py:func:`popxl.Session.run_with_outputs`.


.. _sec_getting_setting_tensor_data:

Getting and setting tensor data
-------------------------------

Once you have created a session, it is possible to write to variable tensors,
and read variable tensors and constant tensors from the device. You can do this
if you want to make comparisons between trained weights and a reference, or to
update or reset weights for debugging. You can also do this if you want to save
progress on your model by storing and reloading the variable tensors.

.. literalinclude:: files/tensor_get_write.py
  :language: python
  :name: tensor_get_write
  :caption: Example of getting and setting tensor data
  :start-after: TensorData begin
  :end-before: TensorData end

.. only:: html

    :download:`Download tensor_get_write.py <files/tensor_get_write.py>`

You can get or set data for multiple tensors via the methods
:py:func:`popxl.Session.get_tensors_data` and :py:func:`popxl.Session.write_variables_data`
respectively. It will defer the transfer to the host or device until the end so
that all the reads or writes are performed in one operation.

When transfers will occur between host and device
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If attached to device, :py:func:`popxl.Session.write_variable_data` will update
both the host and device weights. If not attached, only the host weights will be
updated. The device weights will be updated on the next ``weights_from_host``.
This will happen on the next :py:class:`popxl.Session` context enter, or
when you call py:func:`popxl.Session.weights_from_host` manually.

Similarly, :py:func:`popxl.Session.get_tensor_data` will only ensure the most
up-to-date weights from device are returned if attached to device. If not
attached, the current host weights will be returned.

Furthermore, :py:func:`popxl.Session.get_tensor_data` treats the host weights as
a cache of the device weights, so will only perform the ``weights_to_host`` if
the host weights are possibly out of date. This happens if a runtime function
that can mutate the weights on device has been called (like ``run``) since the
last :py:func:`popxl.Session.weights_to_host`. This happens when calling it
manually, or possibly when exiting the :py:class:`popxl.Session` context or
calling ``get_tensor_data``. Note :py:func:`popxl.Session.write_variable_data`
does not invalidate the host weights as it updates them too.

If only the data for :py:class:`popxl.Constant` tensors are required, then there
will never be a device-to-host transfer. You can perform this operation when not
attached to a device.

The above points are demonstrated in the following example:

.. literalinclude:: files/tensor_get_write_adv.py
  :language: python
  :name: tensor_get_write_adv
  :caption: Demonstration of exactly when host-device transfers occur during tensor reading and writing.
  :start-after: TensorData begin
  :end-before: TensorData end

.. only:: html

    :download:`Download tensor_get_write_adv.py <files/tensor_get_write_adv.py>`

Manual attaching and detaching from device
------------------------------------------

.. warning::
    It is highly recommended to always use the :py:class:`popxl.Session` context
    manager to manage attaching/detaching from device. The following API is
    provided for rare cases where the user needs more precise control. DO NOT
    use it unless the context manager proves insufficient for your use case.

You have direct access to the :py:class:`popart.DeviceInfo` object through
:py:func:`popxl.Session.device`. This object has an API for managing the device.
In particular, you can manually attach or detach from the device.

To attach, you can call :py:func:`popart.DeviceInfo.attach`. This will attempt
to attach to device, returning ``True`` if possible, or ``False`` if the device
is unavailable. When running on ``cpu`` or ``ipu_model`` devices, the attach
will always be successful. When running on ``ipu_hw`` devices, the attach may
fail.

To try attach to a device until a certain timeout, throwing if not possible,
call :py:func:`popart.DeviceInfo.tryAttachUntilTimeout`. If running on
``ipu_hw``, you probably just want to call this function and not ``attach``.

To detach from device, call :py:func:`popart.DeviceInfo.detach`.

Preferable to manually detaching is to use the :py:class:`popart.DeviceInfo`
context manager. This does nothing on enter (so you are still responsible for
attaching), but always detaches on exit.

On exit of the process, it will detach from all devices. However, it is an easy
mistake to be attached whilst the process is performing long-running CPU-bound
code. This hogs the device and makes it difficult to work on a machine shared
with others, thus it is important to remember to manually detach immediately
after the necessary runtime functions have completed. This can be error-prone
due to the attach-detach-attach-ing possibly invalidating IPU memory (as
explained in the note :ref:`here<Running a session>`). For these reasons, it is
highly recommended to always use the context manager.

Nested Session Contexts
-----------------------

It is possible to nest :py:class:`popxl.Session` contexts.
Every time you go from detached to attached on entering the context, a
``weights_from_host`` will occur.
When you leave the context, only if you were attached when entering that
context, a ``weights_to_host`` and detach will occur.

The following code demonstrates the semantics:

.. literalinclude:: files/nested_session_contexts.py
  :language: python
  :name: nested_session_contexts
  :caption: Demonstration of semantics of nested ``Session`` contexts
  :start-after: Session begin
  :end-before: Session end

.. only:: html

    :download:`Download nested_session_contexts.py <files/nested_session_contexts.py>`


Number of host transfers
------------------------

The :py:attr:`~popxl.Ir.num_host_transfers` property of the
:py:class:`~popxl.Ir` class determines the number of iterations required for each
``session.run`` call. For each :py:func:`~popxl.ops.host_load` (per
tensor) operation in your model to run, you will need to increment
``num_host_transfers`` by one. This includes :py:func:`~popxl.ops.host_load` operations inside called subgraphs and repeated subgraphs. For
example, if you have two :py:func:`~popxl.ops.host_load` ops for
tensors ``x`` and ``label`` in the main graph, ``num_host_transfers`` will
be 1. However if you put these ops inside a repeat op that repeats 10 times, you
will need to set ``num_host_transfers`` to 10.

If you have different numbers of :py:func:`~popxl.ops.host_load` ops
for different tensors in your graph, you will find that some streams will
exhaust their data before others, resulting in the exhausted stream looping
around the data buffer before Python is able to provide more data. For example,
assume you have two :py:func:`~popxl.ops.repeat` ops that have host
loaded tensors inside - stream A repeats three times and stream B repeats five
times - providing three batches of data for stream A and five for stream B. This
will result in stream A exhausting its data. For every model run, both streams
will advance by one batch leading to A hitting the end of its allotted data
before B.

In this case, set ``num_host_transfers`` to ``2 *
ceil(number_of_host_load_runs)`` and provide ``ceil(number_of_host_load_runs)``
batches for each stream. In the example, this would mean a
``ir.num_host_transfers = 5 * 2 = 10`` and you would need to provide five
batches for streams A and B. You will need to keep track of how many batches
have been consumed by the model, and perhaps move data to the next model run if
it was not consumed. For example, stream A would need to move the last two
unused batches to the next model run's data. Alternatively, pad the last two
batches of data for stream A with zeros on every model run.

.. note::
  This behaviour will likely change in the future so that the correct number of
  batches of data are required per stream.

.. _sec_data_input_shape:

.. _sec_session_inputs:

Data input shape
----------------

When providing data for the session, you also need to ensure that you provide
enough data for :py:attr:`replication_factor <popxl.Ir.replication_factor>` as well as
for :py:attr:`~popxl.Ir.num_host_transfers`. The input data
will need to have a shape as follows:

.. code-block:: python

  [num_host_transfers, replication_factor, *device_data_shape]

For example, with:

.. code-block:: python

  device_data_shape = (5, 9, 9)
  num_host_transfers = 7
  replication_factor = 2

then:

.. code-block:: python

  input_shape = (7, ) + (2 , ) + (5, 9) = (7, 2,5, 9, 9).

Note, ``replication_factor`` and ``num_host_transfers`` are
independent and need to have separate dimensions in the input data, or you will
find that the data will be consumed out of order.

.. literalinclude:: files/repeat_graph_2.py
  :language: python
  :name: repeat_graph_popxl_2
  :caption: Example of num_host_transfers with a repeat op.
  :start-after: SessionRun3 begin
  :end-before: SessionRun3 end

.. only:: html

    :download:`Download repeat_graph_2.py <files/repeat_graph_2.py>`

.. _device-types:

Device Types
------------

When creating a session, you need to describe the device you are using with
``device_desc``. Possible values are:

1. ``ipu_hw``

  This indicates that you are using physical IPU hardware.

2. ``ipu_model``

  This indicates that you are using the IPU Model. The IPU Model is a simulation
  of the behaviour of the IPU hardware, but it does not completely implement
  every aspect of a real IPU. For example, the IPU Model does not fully support
  replicated graphs nor the same random number generation as the IPU hardware.
  Its arithmetic results may differ from what would be obtained by using the IPU
  hardware. It also does does not support remote storing and loading of
  variable tensors.

3. ``cpu``

  This indicates that you are using a CPU. In some use cases it is faster to use
  a CPU than the IPU Model. The ``cpu`` device type does not support remote
  storing and loading of variable tensors. The ``cpu`` device type also does not
  support replication in any use case.

.. note::
  You do not need to set the number of devices, as this is calculated
  automatically from the number of virtual graphs used and the replication
  factor. An error will be thrown if the number of devices required exceeds the
  number available.
