.. _sec_session:

Session
=======


A session is a class that represents the PopART runtime session and lets you execute a PopXL
graph of operations. You create a session with a constructed :py:class:`popxl.Ir` object as described previously:

.. literalinclude:: ../user_guide/files/tensor_addition_popxl.py
  :language: python
  :name: session_construction_code
  :caption: Example of Session construction
  :start-after: Session begin
  :end-before: Session end

.. only:: html

    :download:`Download tensor_addition_popxl.py <../user_guide/files/tensor_addition_popxl.py>`

Where ``ir`` is the ``popxl.Ir`` object you have created.

.. warning::
     The session takes ownership of the ``popxl.Ir`` object from creation onwards, the ``Ir`` cannot be changed after this point.

The session will prepare and compile the IR that your Python ``popxl.Ir`` class represents.
A `popart.popart_exception` error will be thrown if it is not valid. This can
happen if there are cycles in the graph or if the configuration for the device(s) you have specified is invalid.


Running a session
-----------------

Once you have constructed your session, you can run your model with the relevant inputs to return your outputs.
You have two choices in doing this:

1. ``outputs = session.run(inputs, device_desc)``

  This runs the session with provided ``inputs`` and constructs ``outputs`` in the form of
  ``np.ndarray`` objects to return back to the caller. Input shapes will be validated and outputs will be
  returned in the shape inferred by the IR. ``device_desc`` is a string describing the type of device
  you will use to run the session with, this is described in :numref:`device-types`.

  .. literalinclude:: ../user_guide/files/simple_addition_popxl.py
    :language: python
    :name: session_run_1
    :caption: Example of running a session
    :start-after: SessionRun begin
    :end-before: SessionRun end


2. ``session.run_with_outputs(inputs, outputs, device_desc)``

  If you want to write to part of a larger array, or you already have output arrays constructed, use
  ``session.run_with_outputs``. You must first construct the output arrays to be written to, in the necessary
  shape, then pass it to the session. This runs the model and writes the values to the provided array.
  The shapes of the provided arrays will be validated against the inferred shape and an error thrown if incorrect.

  .. literalinclude:: ../user_guide/files/simple_addition_popxl.py
    :language: python
    :name: session_run_2
    :caption: Example 2 of running a session
    :start-after: SessionRun2 begin
    :end-before: SessionRun2 end

  There is also the method ``create_host_outputs`` which will construct these outputs. This is the method used
  internally in ``session.run`` and provides a shortcut to constructing the arrays required for ``session.run_with_outputs``.

Getting / setting tensor data
-----------------------------

Once you have created a session, it is possible to write to variables, and read variables and constants from
the device. You can do this if you want to make comparisons of trained weights versus a known reference, or
to update or reset weights for debugging. You can also do this if you want to save progress on your
model by storing and reloading the variables.

.. literalinclude:: ../user_guide/files/tensor_get_write_popxl.py
  :language: python
  :name: tensor_get_write
  :caption: Example of getting and writing tensors
  :start-after: TensorData begin
  :end-before: TensorData end

.. only:: html

    :download:`Download tensor_get_write_popxl.py <../user_guide/files/tensor_get_write_popxl.py>`

You can get or set data for multiple Variables via the methods ``get_tensors_data`` and ``write_variables_data``
respectively. It will defer the transfer to the host or IPU so that all the reads or writes are performed in one operation.

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

.. literalinclude:: ../user_guide/files/repeat_graph_popxl_2.py
  :language: python
  :name: repeat_graph_popxl_2
  :caption: Example of num_host_transfers with a repeat op.
  :start-after: SessionRun3 begin
  :end-before: SessionRun3 end

.. only:: html

    :download:`Download repeat_graph_popxl_2.py <../user_guide/files/repeat_graph_popxl_2.py>`

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
