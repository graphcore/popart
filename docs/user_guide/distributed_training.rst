.. _popart_distributed_training:

Distributed training with Horovod
=================================

In order to scale out training with PopART across multiple machines we use
`Horovod <https://github.com/horovod/horovod/>`_ to setup and run collective
operations. There is currently support for the following MPI-based collective
operations: ``Broadcast`` and ``AllReduce``. The ``Broadcast`` operation is
typically run at the start of a training to initialise the weights to have the
same values across the instances. Gradients produced during the backwards pass
will be aggregated and averaged across the instances by running the
``AllReduce`` operation. This ensures that each rank applies the same gradients
to its weights during the weight update step.

How to modify a PopART program for distributed training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Import the Horovod PopART extension:

.. code-block:: python

  import horovod.popart as hvd


Enable the ``hostAllReduce`` PopART session option:

.. code-block:: python


  userOpts = popart.SessionOptions()

  # Enable host side AllReduce operations in the graph
  userOpts.hostAllReduce = True


Initialise the Horovod runtime:

.. code-block:: python

  hvd.init()

Initialise the Horovod ``DistributedOptimizer`` object. The constructor takes
the PopART optimiser, training session and session options objects as arguments.
The ``DistributedOptimizer`` object will add operations to copy gradients into
and out of the IPU and run the Horovod ``AllReduce`` operation:

.. code-block:: python

  distributed_optimizer = hvd.DistributedOptimizer(optimizer, training.session, userOpts)

Broadcast the initial weights from the rank zero process to the other PopART instances:

.. code-block:: python

  hvd.broadcast_weights(training.session, root_rank=0)

Install
~~~~~~~
Requirements for installing the Horovod PopART extension can be found here: `Horovod install <https://github.com/horovod/horovod/>`_.

Configuring and running distributed training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Running distributed training with the Horovod PopART extension can be done in the same way as with other frameworks. For instance, running distributed training across two processes on the same machine can be done with the following command:

.. code-block:: bash

  $ horovodrun -np 2 -H localhost:2 python train.py

Additional documentation on running Horovod can be found here: `Horovod documentation <https://horovod.readthedocs.io/en/latest/>`_.


Full distributed training example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/distributed_training/simple_distributed_training.py
