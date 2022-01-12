.. _popart_tensor_locations:

Tensor locations
================

By default, tensors reside in the on-chip SRAM of the IPU. Tensor location
settings allow smart offloading of tensors to off-chip streaming memory when
required, as well as sharding tensors across replicas in data parallel training.

Setting the tensor location does not interfere with overlapped IO settings, even
though both of them can specify a ``TileSet`` on which the tensor should reside
when being loaded onto the chip's SRAM.

Off-chip streaming memory
~~~~~~~~~~~~~~~~~~~~~~~~~

Off-chip streaming memory is used to temporarily store tensors not immediately
required by IPU computations. It allows to fit larger models or batch sizes
on the IPU, but access to this larger and slower memory pool has to be
infrequent and balanced with computation.

If a tensor is located on or off chip can be controlled by tensor type:

.. code-block:: python

  opts.weightTensorLocationSettings.minElementsForOffChip
  opts.weightTensorLocationSettings.location = popart.TensorLocation(...)
  opts.optimizerStateTensorLocationSettings.minElementsForOffChip
  opts.optimizerStateTensorLocationSettings.location = popart.TensorLocation(...)
  opts.accumulatorTensorLocationSettings.minElementsForOffChip
  opts.accumulatorTensorLocationSettings.location = popart.TensorLocation(...)
  opts.activationTensorLocationSettings.minElementsForOffChip
  opts.activationTensorLocationSettings.location = popart.TensorLocation(...)

Settings ``TensorLocation`` is also applicable to individual tensors
(by tensor name):

.. code-block:: python

  opts.tensorLocationSettingsOverride[name] = popart.TensorLocation(...)


The ``popart.TensorLocation(storage, loadTileSet, storageTileSet)`` settings
object takes up to three arguments relevant for off-chip tensors:

  * ``TensorStorage storage``:

    #. ``OnChip``: Store the tensor in on-chip memory. The default setting for
       all tensors. The tensor remains on the IPU.
    #. ``OffChip``: Store the tensor in off-chip streaming memory when not
       immediately required by IPU computations. This option may not have any
       effect if the PopART IR decides that there is no sensible time-frame when
       the tensor should be scheduled for being copied off-chip.

  * ``TileSet loadTileSet``: The set of tiles that stream the data from and to
    off-chip streaming memory.

    #. ``IO``: Load data from off-chip memory to the IO tiles first.
    #. ``Compute``: Load data from off-chip memory directly to the compute tiles.

  * ``TileSet storageTileSet``: The set of tiles on which the tensor preferentally
    resides when on-chip. Does not have any effect if the ``loadTileSet`` is ``Compute``.

    #. ``IO``: Data should stay on IO tiles whenever possible.
    #. ``Compute``: Data should move to Compute tiles as soon as possible.

PopART will intelligently decide, based on the provided settings, when exactly a
tensor will be moved between IO tiles, compute tiles and off-chip streaming
memory.

If ``TileSet::IO`` is used in any location setting, a subset of IPU tiles
have to be set aside:

.. code-block:: python

  opts = popart.SessionOptions()
  opts.numIOTiles = 128

Replicated tensor sharding (RTS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applicable to tensors that usually contain the same information on each replica.
Replicated tensor sharding eliminates redundant data storage when the full
(unsharded) tensor does not need to be present on the IPU. RTS tensors of which
each replica needs a full copy of, are recombined with a replicated
``AllGather`` operation. Fully updated tensors that need to be sharded (and
reduced) again, require a replicated ``ReduceScatter`` operation.

Replicated tensor sharding modifies existing optimizers in the model,
and modifieds or replaces the ``ReplicatedAllReduce`` which is typically
applied to gradients in data parallel training.

Collective ``ReplicatedAllReduce`` operations are present in the transformed
PopART IR graph when the model contains an optimizer that the user has set, and
if replication is enabled:

.. code-block:: python

  opts = popart.SessionOptions()
  opts.enableReplicatedGraphs = True
  opts.replicatedGraphCount = num_replicas


Only variable tensors that are assumed to be equal across replicas can be
sharded. This entails the model weights and the optimizer states
(for example momentums of stochastic gradient descent) in data parallel training
configurations.

If only weights should be sharded:

.. code-block:: python

  opts.weightStateTensorLocationSettings.minElementsForReplicatedTensorSharding = num_replicas
  opts.weightTensorLocationSettings.location.replicatedTensorSharding = popart.ReplicatedTensorSharding.On

If optimizer states should be sharded in addition:

.. code-block:: python

  opts.optimizerStateTensorLocationSettings.minElementsForReplicatedTensorSharding = num_replicas
  opts.optimizerStateTensorLocationSettings.location.replicatedTensorSharding = popart.ReplicatedTensorSharding.On

Sharded tensors have modified sizes on the IPU, but can be used as usual on the
host. If the original tensor has a shape of
``[5,2,3]`` (30 elements in total) and we shard across four replicas,
each replica will hold ``ceil(5*2*3/4)`` elements. Two replicas will have
unpadded data of shape ``[8]``, while the other two replicas contain 7 data
elements, and 1 pad (zero) element. Since all replicas share the same compiled
binary, there is no distinction between padded and unpadded sharded tensor.
When downloading sharded tensor from the IPUs to the host, the sharded tensors
are concatenated and the padding is removed
(see GCL ``CollectiveBalancedReorder``).


RTS sharding domains and distributed instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For distributed instances of a PopART program, it is recommended to launch the
training application with PopRun. PopDist can then be used to configure the
per-instance replication settings automatically:

.. code-block:: python

  # Let popdist handle distributed settings, such as:
  # opts.enableReplicatedGraphs
  # opts.replicatedGraphCount
  # opts.enableDistributedReplicatedGraphs
  # opts.globalReplicaOffset
  # opts.globalReplicationFactor
  popdist.popart.configureSessionOptions(opts)

For more information about PopRun and PopDist, refer to the `user guide <https://docs.graphcore.ai/projects/poprun-user-guide/>`__, including details about the installation of Horovod if you are using the MPI communication protocol.

When using distributed instances across two or more IPU Pods, the gateway link
transfer speeds (IPU MK1, MK2) are lower than the IPU link speed within the Pod.
It is therefore more beneficial to load replica sharded tensors from streaming
memory and ``AllGather`` across the replicated instances within a Pod rather
than across all replicas.

The sharding domain can be applied to types of tensors or individual tensors.
Tensors that are linked together (for example the optimizer state,
accumulator and weight being consumed by the same optimizer instance) should
be configured with the same replicated tensor sharding domain.

The recommended configuration for sharding optimizer states with multiple
IPU Pods is therefore:

.. code-block:: python

  num_local_replicas = popdist.getNumLocalReplicas()
  num_total_replicas = popdist.getNumTotalReplicas()

  if num_total_replicas > num_local_replicas:
      # Fewer elements would not make sense to shard
      opts.optimizerStateTensorLocationSettings.minElementsForReplicatedTensorSharding = num_local_replicas
      # Only enable sharding on the optimizer state
      opts.optimizerStateTensorLocationSettings.location.replicatedTensorSharding = popart.ReplicatedTensorSharding.On

      sharding_domain = popart.CommGroup(
          popart.CommGroupType.Consecutive, num_local_replicas)

      # Ensure all related tensors have the same sharding domain set
      opts.weightTensorLocationSettings.location.shardingDomain = sharding_domain
      opts.optimizerStateTensorLocationSettings.location.shardingDomain = sharding_domain
      opts.accumulatorTensorLocationSettings.location.shardingDomain = sharding_domain

The settings will apply to all weights, optimizer states and accumulators in the
model.


Examples for supported ``CommGroup`` settings:

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
