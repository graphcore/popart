.. _sec_application_example_mnist_rts:

Application example: MNIST with replication and RTS
===================================================

In this section, we use RTS variables (:numref:`sec_rts`) based on the previous MNIST application example (:numref:`sec_application_example_mnist`).
Recall that RTS is based on replication, so first of all we need to change the code to support replication.
Then we need to change the variable tensors into RTS variable tensors.

Add support for replications
----------------------------
The `replication <https://docs.graphcore.ai/projects/ipu-programmers-guide/en/latest/algorithmic_techniques.html#replication>`_,
a data parallelism, is achieved by running the same program in parallel on multiple sets of IPUs. PopXL currently support
`local replication <https://docs.graphcore.ai/projects/popart-user-guide/en/latest/performance.html#local-replication>`_.
In local replication, replications are handled by a single host instance. We've added the command line option ``--replication-factor``
to the code, indicating the number of replications we need. Then we assign this parameter to each IR, as
shown in the code below, in ``build_train_ir`` and ``build_test_ir``.

.. code-block:: python

    ir.replication_factor = opts.replication_factor

When replication is used, the total number of data processed in each batch equals to the batch size in one replica multiplied by
the replication factor. In this case, we need to provide enough input data to run in multiple replicas each step. To have
the same training results from different replication factors, we need to guarantee that the ``batch_size * replication_factor``
stays the same. We can achieve this when preparing the dataset by simply keeping the code for batch size unchanged (when
``replication_factor`` equal to 1), and multiplying the replication factor as shown in the code below.

.. code-block:: python

    training_data, test_data = get_mnist_data(
        opts.test_batch_size * opts.replication_factor,
        opts.batch_size * opts.replication_factor)

We also need to change the data passed to each session to match the required dimension of its inputs in ``train``.
(:numref:`sec_session_inputs`)

.. literalinclude:: files/mnist_rts.py
  :language: python
  :start-after: train_session_inputs begin
  :end-before: train_session_inputs end

After making similar changes of data shape for the test session, replication is also supported in the testing of
the trained model. You can check whether the replication works by running:

.. code-block:: bash

  python mnist_rts.py --replication-factor 2 --batch-size 4

It should give similar test accuracy to the following command

.. code-block:: bash

  python mnist_rts.py --replication-factor 1 --batch-size 8

Change variable tensors to RTS variable tensors
-----------------------------------------------
We can create RTS variables for training and testing in two different ways. One which exposes the remote buffer
(using :py:func:`~popxl.remote_replica_sharded_variable`) and one which does not expose the remote buffer (using
:py:func:`~popxl.replica_sharded_variable()`). In the code we:

- Create variable tensor ``W0`` by using :py:func:`~popxl.replica_sharded_variable()`;
- Create variable tensor ``W1`` by using :py:func:`~popxl.remote_replica_sharded_variable`.

To collect all the info needed by an RTS variable, we've used the named tuple

.. code-block:: python

  Trainable = namedtuple('Trainable', ['var', 'shards', 'full', 'remote_buffer'])

The ``var`` is the remote variable tensor, the ``shards`` is the shards after remote load operation,
the ``full`` is the tensor after all gather operation, and the ``remote_buffer`` is the corresponding
remote buffer that handles the variable if known. If a variable tensor is not an RTS variable, then
``shards``, ``full``, and ``remote_buffer`` will be ``None``.

The ``Trainable`` for ``W0``, ``trainable_w0``, is created as shown in the code below:

.. literalinclude:: files/mnist_rts.py
  :language: python
  :start-after: rts_W0 begin
  :end-before: rts_W0 end

The ``Trainable`` for ``W1``, ``trainable_w1``, is created as shown in the code below:

.. literalinclude:: files/mnist_rts.py
  :language: python
  :start-after: rts_W1 begin
  :end-before: rts_W1 end

Notice that when we get the gradient for an input RTS variable tensor, the tensor is the (non-sharded) "full"
tensor which has been gathered from the shards by using :py:func:`~popxl.ops.collectives.replicated_all_gather()`.
After obtaining gradients for the "full" tensors, the gradients are then sliced by using :py:func:`~popxl.ops.collectives.replicated_reduce_scatter()`
to update each shard of the RTS variable tensor.

.. code-block:: python

    ......
    if params["W1"].shards is not None:
        grad_w_1 = ops.collectives.replica_sharded_slice(grad_w_1)

    ......
    if params["W0"].shards is not None:
        grad_w_0 = ops.collectives.replica_sharded_slice(grad_w_0)

When you update a variable tensor, and if a remote buffer is used, you also need to restore the updated value to the right place as well.

.. literalinclude:: files/mnist_rts.py
  :language: python
  :start-after: update begin
  :end-before: update end
