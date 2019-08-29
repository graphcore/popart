Glossary
========

Sample
~~~~~~
The smallest division of a data set. 

Micro-batch size
~~~~~~~~~~~~~~~~
The number of samples processed in a single execution of a graph on a single device.
Also referred to as the machine batch size.

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
This is defined as `micro-batch size * replication factor * accumulation factor`.
This is the number of samples per weight update.

Batches per step
~~~~~~~~~~~~~~~~
The number of batches to run in a single call to `Session::run`.

Step size
~~~~~~~~~
This is defined as `batch size * batches per step`.
This is the number of samples per step.
