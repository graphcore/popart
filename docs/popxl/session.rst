.. _sec_session:

Session
=======


A :py:class:`popxl.Session` is a class that compiles and executes a :py:class:`popxl.Ir`.
You can construct one like so:

.. literalinclude:: files/tensor_addition.py
  :language: python
  :name: session_construction_code
  :caption: Example of Session construction
  :start-after: Session begin
  :end-before: Session end

.. only:: html

    :download:`Download tensor_addition.py <files/tensor_addition.py>`

Where ``ir`` is the ``popxl.Ir`` object you have created.
The second parameter is a string literal describing the type of device ``session``
will run on. The possible values of this parameter are further described in :numref:`device-types`.

.. warning::
    The ``Session`` takes unique ownership of the :py:class:`popxl.Ir` object.
    From construction onwards, the ``Ir`` cannot be changed after this point.
    You also cannot create another ``Session`` from the same ``Ir``.

Constructing the session will compile the IR that your Python ``popxl.Ir`` class
represents. A `popart.popart_exception` error will be thrown if it is not valid.
This could happen, for example, if there are cycles in the graph, or if the
configuration for the device(s) you have specified is invalid.


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

  .. literalinclude:: files/simple_addition.py
    :language: python
    :name: session_run_2
    :caption: Example of running with :py:func:`popxl.Session.run_with_outputs`
    :start-after: SessionRun2 begin
    :end-before: SessionRun2 end

Finally, there is also :py:func:`popxl.Session.create_host_outputs` that will create the ``Mapping`` for you, with each stream mapped to an empty :py:class:`np.ndarray`.
This is the method used internally in :py:func:`popxl.Session.run` and provides a shortcut to constructing the output arrays required for :py:func:`popxl.Session.run_with_outputs`.


Getting / setting tensor data
-----------------------------

Once you have created a session, it is possible to write to variables, and read variables and constants from
the device. You can do this if you want to make comparisons of trained weights versus a known reference, or
to update or reset weights for debugging. You can also do this if you want to save progress on your
model by storing and reloading the variables.

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


num_host_transfers
------------------

``num_host_transfers`` is a property of the ``Ir`` class that determines the amount of data required for each ``session.run`` call.
For each ``host_load`` (per tensor) operation in your model to run, you will need to increment the ``num_host_transfers`` by one. This includes
``host_load`` operations inside called subgraphs and repeated subgraphs.
For example, if you have two ``host_load`` ops for tensors ``x`` and ``label`` in the main graph, your ``num_host_transfers``
will be 1. However if you put these ops inside a repeat op that repeats 10 times, you will need to set ``num_host_transfers`` to 10.

If you have different numbers of host load ops for different tensors in your graph, you will find that some streams will
exhaust their data before others, resulting in the exhausted stream looping round the data buffer before Python is able to provide more data.
For example, if you have 2 repeat ops that have host loaded tensors inside, one repeating 3 times (stream A),
the other 5 times (stream B), providing 3 batches of data for stream A and 5 for stream B will result in A exhausting it's data. For every
model run, both streams will advance by one batch; leading to A hitting the end of it's allotted data before B.

In this case, set your ``num_host_transfers`` to ``2 * ceil(number_of_host_load_runs)`` and provide ``ceil(number_of_host_load_runs)`` batches
for each stream. In the example, this would mean a ``ir.num_host_transfers = 5 * 2 = 10`` and  you would need to provide 5 batches for stream A and B.
You will need to keep track of how many batches have been consumed by the model, and perhaps move data to the next model run if it
was not consumed. For example stream A would need to move the last 2 unused batches to the next model run's data. Alternatively
pad the last 2 batches of data for stream A with zeros on every model run.

.. note::
  This behaviour will likely change in the future so that the correct number of batches of data are
  required per stream.

.. _sec_session_inputs:

Data input shape
----------------

When providing data for the session, you also need to ensure that you provide enough data for the
``replication_factor`` as well as the ``num_host_transfers``. The data shape provided for input will need
to be of shape:

.. code-block:: python

  [num_host_transfers, replication_factor, *device_data_shape]

For example, ``device_data_shape = (5, 9, 9)``, ``num_host_transfers = 7``, ``replication_factor = 2``
then ``input_shape = (7, ) + (2 , ) + (5, 9) = (7, 2, 5, 9, 9)``. Note, ``replication_factor`` and ``num_host_transfers``
are independent and need to have separate dimensions in the input data, or you will find data will be consumed output
of order.

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

When creating a session, you need to provide a ``device_desc`` argument to describe the device type you are using.
It can be one of ``ipu_hw``, ``ipu_model`` or ``cpu``:

1. ``ipu_hw``

  Physical IPU hardware.

2. ``ipu_model``

  The IPU Model is a simulation of the behaviour of the IPU hardware. It does not completely implement every
  aspect of a real IPU. For example, the IPU Model does not fully support replicated graphs
  nor the same random number generation as the hardware. Its arithmetic results may differ from what would be
  obtained by using the IPU hardware. It also does does not support remote storing and loading of variables.

3. ``cpu``

  Run using the CPU. In some use cases it is faster than the IPU model. In addition to not
  supporting remote storing and loading of variables as per the ``ipu_model``, the ``cpu`` device does
  not support replication in any use case.

.. note::
  You do not need to set the number of devices, as this is calculated automatically using the number
  of virtual graphs used and the replication factor. An error will be thrown if the number required
  exceeds the number available.
