Performance optimisation
========================

.. TODO: Add sections on recomputation,
.. automatic virtual graphs, replication.

Pipelined execution
-------------------

Pipelining is a feature for optimising utilization of a multi-IPU system by
parallelizing the execution of model partitions, with each partition operating
on a separate mini-batch of data. We refer to these partitions here as pipeline
stages.

You can split a model into pipeline stages by annotating operations in the
ONNX model using the ``Builder`` class. There are two APIs. For instance, to
place a specific convolution onto pipeline stage 1:

.. code-block:: python

  o = builder.aiOnnx.conv([x, w])
  builder.pipelineStage(o, 1)

Or using the context manager:

.. code-block:: python

  with builder.pipelineStage(1):
      o = builder.aiOnnx.conv([x, w])

Alternatively, if you have annotated the operations with ``VirtualGraph``
attributes, then you can defer annotating pipeline stage attributes to
the ``Session`` constructor. However, it is recommended that you profile
the model to choose a partitioning with the optimal utilization.

You can enable pipelined execution by setting the session option
``enablePipelining`` to ``True``. See ``SessionOptions`` in the
`PopART C++ API Reference
<https://docs.graphcore.ai/projects/popart-cpp-api>`_.

Note that, by default, pipelining a training model with variable tensors stored
over different pipeline stages results in 'stale weights' (see Zhang et al.,
arXiv:1912.12675). This can be avoided by enabling gradient accumulation. In
this case, the pipeline is flushed before the weight update applies the
accumulated gradients.


Sync configuration
------------------

In a multi-IPU system, synchronisation (sync) signals are used to ensure that
IPUs are ready to exchange data and that data exchange is complete. These sync
signals are also used to synchronise host transfers and access to remote
buffers.

Each IPU can be allocated to one or more "sync groups". At a synchronization
point, all the IPUs in a sync group will wait until all the other IPUs in the
group are ready.

Sync groups can be used to to allow subsets of IPUs to overlap their
operations. For example, one sync group can be performing data transfers to or
from the host, while another group is processing a previous batch of data.

You can configure the sync groups using the PopART ``syncPatterns`` option
when creating a device.

For example, the following code shows how to set the sync configuration to
"ping-pong" mode.

.. code-block:: python

    sync_pattern = popart.SyncPattern.Full
    if args.execution_mode == "PHASED":
        sync_pattern = popart.SyncPattern.ReplicaAndLadder
    device = popart.DeviceManager().acquireAvailableDevice(
        request_ipus,
        pattern=sync_pattern)

Sync patterns
.............

There are three sync patterns available. These control how the IPUs are
allocated to the two sync groups, GS1 and GS2.

The sync patterns are described with reference to the diagram below, which
shows four IPUs: A, B, C and D.

.. _fig_sync_patterns:
.. figure:: images/syncpatterns.*
  :width: 90%
  :align: center
  :alt:  Sync patterns in PopART

  Sync patterns

* **Full:** All four IPUs are in both sync groups. Any communication between
  the IPUs or with the host, will require all IPUs to synchronise.

* **SinglePipeline:** One sync group contains all four of the IPUs. So any
  communication using that sync group will synchronise all the IPUs.

  The other sync group is used separately by each IPU. This means that they
  can each sync with the host independently, without syncing with each other.
  This allows any IPU to be doing host IO, for example, while others are
  processing data.

* **ReplicaAndLadder:** One sync group contains all the IPUs.
  The other sync group is used independently by sets of IPUs,
  for example A+C and B+D. This means that each subset can communicate
  independently of each other. The two groups of IPUs can then alternate
  between host I/O and processing.

For more information on how the sync groups are used by the Poplar framework,
please refer to the `Poplar and PopLibs User Guide
<https://www.graphcore.ai/docs/poplar-and-poplibs-user-guide>`_.
