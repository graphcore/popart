Glossary
========

Sample
~~~~~~
The smallest division of a data set.

Micro-batch size
~~~~~~~~~~~~~~~~
The number of samples processed in a single execution of a graph on a single device.
Also referred to as the machine batch size.
The micro-batch shape, or the shape of input data as defined in the ONNX model,
is therefore [micro_batch_size, *sample_shape]

Replication factor
~~~~~~~~~~~~~~~~~~
The number of graphs to be run in parallel over multiple devices.
The weight gradients from each device will be accumulated before a weight update.
Also referred to as "device replication factor" or "spatial replication factor".
This is sometimes called data-parallel execution.

Accumulation factor
~~~~~~~~~~~~~~~~~~~
The weight gradients will be accumulated over this number
of micro-batches in series before a weight update.
Also referred to as "temporal replication factor".

Accumulation can be thought of as doing replication on a single device.

Batch size
~~~~~~~~~~
This is defined as ``micro-batch size * replication factor * accumulation
factor``.
This is the number of samples per weight update.

Batches per step
~~~~~~~~~~~~~~~~
The number of batches to run in a single call to ``Session::run``.

Step size
~~~~~~~~~
This is defined as ``batch size * batches per step``.
This is the number of samples per step.

Input data shape
~~~~~~~~~~~~~~~~
Inputs to a ``session.run()`` call are read in with the assumption that data is
arranged in the shape:

[batches_per_step, accl_factor, repl_factor, micro_batch_size, *sample_shape]

However, there is no constraint of the shape of the input array, except that it
has the correct number of elements.
