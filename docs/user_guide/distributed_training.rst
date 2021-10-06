.. _popart_distributed_training:

Distributed training with Horovod
=================================

In order to scale out training with PopART across multiple machines we use
`Horovod <https://github.com/horovod/horovod/>`_ to setup and run collective
operations. There is support for the ``Broadcast`` and ``AllReduce`` collective operations.
The ``Broadcast`` operation is typically run at the start of a training to initialise the weights to have the
same values across the instances. Gradients produced during the backwards pass
will be aggregated and averaged across the instances by running the
``AllReduce`` operation. This ensures that each rank applies the same gradients
to its weights during the weight update step.

How to modify a PopART program for distributed training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Import the Horovod PopART extension:

.. code-block:: python

  import horovod.popart as hvd


Initialise the Horovod runtime:

.. code-block:: python

  hvd.init()

Initialise the Horovod ``DistributedOptimizer`` object. The constructor takes
the PopART optimiser, training session and session options objects as arguments.
The ``DistributedOptimizer`` object will add operations to copy gradients into
and out of the IPU and run the Horovod ``AllReduce`` operation:

.. code-block:: python

  distributed_optimizer = hvd.DistributedOptimizer(optimizer, training.session, userOpts)

Insert the all reduce operation:

.. code-block:: python

  distributed_optimizer.insert_host_allreduce()


Broadcast the initial weights from the rank zero process to the other PopART instances:

.. code-block:: python

  hvd.broadcast_weights(training.session, root_rank=0)

Install
~~~~~~~
The Horovod PopART extension Python wheel can be found in the Poplar SDK downloaded from `<https://downloads.graphcore.ai/>`_. System prerequisites for installing the Horovod PopART extension can be found here: `Horovod install <https://horovod.readthedocs.io/en/latest/summary_include.html#install>`_.

Configuring and running distributed training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Running distributed training with the Horovod PopART extension can be done in the same way as with other frameworks. For instance, running distributed training across two processes on the same machine can be done with the following command:

.. code-block:: bash

  $ horovodrun -np 2 -H localhost:2 python train.py

Alternatively we can use the `Gloo <https://github.com/facebookincubator/gloo>`_ backend for the collective operations as shown below:

.. code-block:: bash

  $ horovodrun --gloo -np 2 -H localhost:2 python train.py

Additional documentation on flags that can be passed to ``horovodrun`` can be found here: `Horovod documentation <https://horovod.readthedocs.io/en/latest/>`_.


Full distributed training example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A small example illustrating how to use Horovod with PopART:

.. literalinclude:: files/simple_distributed_training.py
