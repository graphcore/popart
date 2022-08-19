.. _popart_tensor_locations:

Tensor locations
================

The memory in an :term:`glossary:IPU-machine` is made up of :term:`glossary:In-Processor-Memory` (memory on the IPU) and :term:`glossary:Streaming Memory` (memory not on the IPU). For more details about the memory architecture of the IPU hardware, refer to the :ref:`IPU Programmer's Guide <ipu-programmers-guide:sec_mem_arch>`.

By default, tensors reside in the In-Processor-Memory of the the IPU, but tensor location settings allow smart offloading of tensors to the Streaming Memory when required, as well as sharding tensors across replicas in data parallel training.

Setting the tensor location does not interfere with overlapped IO settings (:numref:`popart_overlap_io`), even
though both of them can specify a tile set (:py:attr:`~popart.TileSet`) on which the tensor should reside when being loaded onto the In-Processor-Memory.

Streaming Memory
~~~~~~~~~~~~~~~~

Streaming Memory is used to temporarily store tensors not immediately
required by IPU computations. It allows larger models or :term:`batch sizes <batch size>` to fit
on the IPU, but access to this larger and slower memory pool has to be
infrequent and balanced with computation.

Whether a tensor is located in Streaming Memory (off-chip) or in In-Processor-Memory (on-chip) can be controlled by various options in :py:class:`~popart.SessionOptions`:

.. code-block:: python

  opts = popart.SessionOptions()
  opts.weightTensorLocationSettings.minElementsForOffChip
  opts.weightTensorLocationSettings.location = popart.TensorLocation(...)
  opts.optimizerStateTensorLocationSettings.minElementsForOffChip
  opts.optimizerStateTensorLocationSettings.location = popart.TensorLocation(...)
  opts.accumulatorTensorLocationSettings.minElementsForOffChip
  opts.accumulatorTensorLocationSettings.location = popart.TensorLocation(...)
  opts.activationTensorLocationSettings.minElementsForOffChip
  opts.activationTensorLocationSettings.location = popart.TensorLocation(...)

The class :py:class:`popart.TensorLocation` can also be used to customise location settings for individual tensors.

.. code-block:: python

  opts.tensorLocationSettingsOverride[name] = popart.TensorLocation(...)


The :py:class:`TensorLocation(storage, loadTileSet, storageTileSet) <popart.TensorLocation>` settings object takes up to three arguments relevant for off-chip tensors:

  * :py:class:`~popart.TensorStorage` ``storage``:

    #. ``OnChip``: Store the tensor in on-chip In-Processor-Memory. The default setting for
       all tensors. The tensor remains on the IPU.
    #. ``OffChip``: Store the tensor in off-chip Streaming Memory when not
       immediately required by IPU computations. This option may not have any
       effect if the PopART IR decides that there is no sensible time-frame when
       the tensor could be scheduled for being copied off-chip.

  * :py:class:`~popart.TileSet` ``loadTileSet``: The set of tiles that stream the data from and to
    Streaming Memory.

    #. ``IO``: Load data from Streaming Memory to the IO tiles first.
    #. ``Compute``: Load data from Streaming Memory directly to the compute tiles.

  * :py:class:`~popart.TileSet` ``storageTileSet``: The set of tiles on which the tensor preferentally
    resides when on-chip. Does not have any effect if the ``loadTileSet`` is ``Compute``.

    #. ``IO``: Data should stay on IO tiles whenever possible.
    #. ``Compute``: Data should move to compute tiles as soon as possible.

PopART will intelligently decide, based on the provided settings, when exactly a
tensor will be moved between IO tiles, compute tiles and off-chip Streaming
Memory.

If ``TileSet::IO`` is used in any location setting, a subset of IPU tiles
have to be set aside:

.. code-block:: python

   opts.numIOTiles = 128

Replicated tensor sharding
~~~~~~~~~~~~~~~~~~~~~~~~~~

`Replicated tensor sharding <https://docs.graphcore.ai/projects/ipu-programmers-guide/en/latest/algorithmic_techniques.html#replicated-tensor-sharding>`__ (RTS) is applicable to tensors that usually contain
the same information on each replica. RTS eliminates redundant data storage when
the full (unsharded) tensor does not need to be present on the IPU. If the full
tensor is needed, a replicated ``AllGather`` operation is used to recombine the
sharded tensor. Fully updated tensors that need to be sharded (and reduced)
again, require a replicated ``ReduceScatter`` operation.

RTS modifies existing optimizers in the model, and modifies or replaces the
``ReplicatedAllReduce`` which is typically applied to gradients in data parallel
training.

In PopART, collective ``ReplicatedAllReduce`` operations are present in the transformed IR graph when the model contains an optimizer that the user has set, and if replication is enabled:

.. code-block:: python

  opts.enableReplicatedGraphs = True
  opts.replicatedGraphCount = num_replicas


Only variable tensors that are assumed to be equal across replicas can be
sharded. This includes the model weights and the optimizer states
(for example momentums of stochastic gradient descent) in data parallel training
configurations.

If only weights should be sharded, then you can set:

.. code-block:: python

  opts.weightStateTensorLocationSettings.minElementsForReplicatedTensorSharding = num_replicas
  opts.weightTensorLocationSettings.location.replicatedTensorSharding = popart.ReplicatedTensorSharding.On

If optimizer states should be sharded in addition, then you can set:

.. code-block:: python

  opts.optimizerStateTensorLocationSettings.minElementsForReplicatedTensorSharding = num_replicas
  opts.optimizerStateTensorLocationSettings.location.replicatedTensorSharding = popart.ReplicatedTensorSharding.On

The size of sharded tensors on the IPU is smaller than that of the full tensor,
but they can be used normally on the host. For example, take a tensor with a
shape of ``[5,2,3]`` and with 30 elements in total. If we shard across four
replicas, each replica will have a size of :math:`\\ceil(\frac{5*2*3}{4})=8`.
However, since we have 30 elements, two replicas will contain 8 elements and the
other two will contain 7 elements and the remaining element will be padded with
a 0. Since all replicas share the same compiled binary, padded and unpadded
sharded tensors are handled in the same way. When loading sharded tensors
from the IPUs to the host, the sharded tensors are concatenated and the padding
is removed (see :cpp:class:`gcl::CollectiveBalancedReorder`).


RTS sharding domains and distributed instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For distributed instances of a PopART program, it is recommended to launch the
training application with PopRun. PopDist can then be
used to configure the per-instance replication settings automatically:

.. code-block:: python

  # Let popdist handle distributed settings, such as:
  # opts.enableReplicatedGraphs
  # opts.replicatedGraphCount
  # opts.enableDistributedReplicatedGraphs
  # opts.globalReplicaOffset
  # opts.globalReplicationFactor
  popdist.popart.configureSessionOptions(opts)

For more information about PopRun and PopDist, refer to the :doc:`poprun-user-guide:index`, which includes details about the installation of Horovod if you are using the MPI communication protocol.

When using distributed instances across two or more :term:`glossary:Pod`\s, the
GW-Link transfer speeds (for both the IPU Mk1 and Mk2 architectures) are slower
than the IPU-Link speed within the :term:`glossary:Pod`. It is therefore
beneficial to load replica sharded tensors from Streaming Memory and
``AllGather`` across the replicated instances within a :term:`glossary:Pod`
rather than across all replicas.

The sharding domain can be applied to types of tensors or individual tensors.
Tensors that are linked together (for example the optimizer state, accumulator
and weight being consumed by the same optimizer instance) should be configured
with the same replicated tensor sharding domain.

.. note:: The term

The recommended configuration for sharding optimizer states with multiple :term:`glossary:Pod`\s is:

.. code-block:: python

  # Number of local replicas
  num_local_replicas = popdist.getNumLocalReplicas()
  # Number replicas in total
  num_total_replicas = popdist.getNumTotalReplicas()

  if num_total_replicas > num_local_replicas:
      # It would not make sense to shard fewer elements
      opts.optimizerStateTensorLocationSettings.minElementsForReplicatedTensorSharding = num_local_replicas
      # Only enable sharding on the optimizer state
      opts.optimizerStateTensorLocationSettings.location.replicatedTensorSharding = popart.ReplicatedTensorSharding.On

      # Set the sharding domain
      sharding_domain = popart.CommGroup(
          popart.CommGroupType.Consecutive, num_local_replicas)

      # Ensure all related tensors have the same sharding domain set
      opts.weightTensorLocationSettings.location.shardingDomain = sharding_domain
      opts.optimizerStateTensorLocationSettings.location.shardingDomain = sharding_domain
      opts.accumulatorTensorLocationSettings.location.shardingDomain = sharding_domain

These settings will apply to all weights, optimizer states and accumulators in the model.

:py:class:`~popart.CommGroup` is used to set the sharding domain. The :py:class:`~popart.CommGroup` class is composed of the :py:class:`~popart.CommGroupType` enum,
and the size of each group. Examples of ``CommGroup`` settings are:

  - ``popart.CommGroup(popart.CommGroupType.All, 0)``:
    Default, shard the tensor across all replicas and all instances. Currently
    not supported for multiple program instances, since each host instance
    requires the full tensor. If sharding across two instances, each host would
    only have access to half the (sharded) tensor.
  - ``popart.CommGroup(popart.CommGroupType.Consecutive, num_local_replicas)``:
    Shard the tensor across all replicas owned by a single instance. Each host
    instance has access to the complete variable tensor. The size of the domain
    currently has to match ``num_local_replicas``, which means sharding across,
    for example, half the replicas managed by an instance is not supported.
