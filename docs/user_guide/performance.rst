Performance optimisation
========================

.. TODO: Add sections on recomputation,
.. automatic virtual graphs.

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

Graph replication
-----------------
PopART has the ability to run multiple copies of your model in parallel
on distinct sets of IPUs. This is called *graph replication*. Informally,
replication is a means of parallelising your inference or training workloads.

When training, weight updates are coordinated between replicas to ensure
replicas benefit from each other's weight updates. A reduction is
applied on the loss gradients across replicas according to the
``ReductionType`` specified when constructing the loss operation.
The reduction and subsequent weight update both involve some communication
between replicas. This communication is managed by PopART.

When you use replication, PopART also manages the splitting and distribution of
input data, making sure the data specified in the ``StepIO`` is split evenly
between replicas. This does mean you need to provide enough input data to
satisfy all (local) replicas.

There are two tiers of replication available in PopART, *local* and
*global*, each of which we will describe below.

Note that replication is not supported on IPU model targets.

Local replication
~~~~~~~~~~~~~~~~~

Local replications are replications managed by a single PopART
process. This means local replication is limited to those IPUs that are
accessible to the host machine that PopART is running on. To enable local
replication, set session option
``enableReplicatedGraphs`` to ``True`` and set ``replicatedGraphCount`` to the
number of times you want to replicate your model. For example, to replicate
a model twice, pass the following session options to your session:

.. code-block:: python

  opts = popart.SessionOptions()
  opts.enableReplicatedGraphs = True
  opts.replicatedGraphCount = 2

Note that if one replica of your model uses, say, 3 IPUs then with a
``replicatedGraphCount`` of 2 you will need 6 IPUs to run both replicas.
Also, you will need to provide twice the volume of input data. The data returned
for each anchor will include a local replication dimension for
all values of ``AnchorReturnType``.

More details on the expected shapes of input and output data (for a given set of
session options) can be found in the `API documentation
<https://docs.graphcore.ai/projects/popart-cpp-api/>`_ of the ``IStepIO`` and
``DataFlow`` classes, respectively.

Global replication
~~~~~~~~~~~~~~~~~~

In addition to local replication, it is possible for multiple PopART processes
to work together using *global replication*. With this option, as the PopART
processes may run on separate hosts, you are not limited to using only the IPUs
that are available to a single host. It is also possible to combine local and
global replication.

To enable global replication, set ``enableDistributedReplicatedGraphs`` to
``True`` and set ``globalReplicationFactor`` to the desired total number of
replications (*including* local replication). Finally,
``globalReplicaOffset`` must be set to a different offset for each PopART
processes involved, using offsets starting from 0 and incrementing by the local
replication factor for each process.

For example, if the local replication factor is 2 and we want to replicate this
over four PopART processes then we need to configure a global replication
factor of 8. We then expect the ``globalReplicaOffset`` in the respective PopART
processes to be set to 0, 2, 4 and 6, respectively, going up in increments
equivalent to the local replication factor. As an example, the configuration
of the PopART session on the second host is shown below:

.. code-block:: python

  opts = popart.SessionOptions()
  # Local replication settings.
  opts.enableReplicatedGraphs = True
  opts.replicatedGraphCount = 2
  # Global replication settings.
  opts.enableDistributedReplicatedGraphs = True
  opts.globalReplicationFactor = 8
  opts.globalReplicaOffset = 2 # <-- Different offset for each PopART instance

Note that when local and global replication are used in tandem the data provided
to each PopART instance (in the ``IStepIO`` instance passed to ``Session::run``)
should contain only the data required for the local replicas. Moreover,
the output anchors will also only contain the output data for the local
replicas. Essentially, input and output data shapes are unaffected by global
replication settings.

More details on the input and output shapes can be found in the
`API documentation <https://docs.graphcore.ai/projects/popart-cpp-api/>`_ of the
``IStepIO`` and ``DataFlow`` classes, respectively.

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
