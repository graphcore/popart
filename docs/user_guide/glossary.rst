Glossary
========

.. glossary::
  :sorted:

  Sample
    The smallest division of a data set.

  Micro-batch size
    The number of samples processed in a single execution of a graph on a single device.
    Also referred to as the machine batch size.
    The micro-batch shape, or the shape of input data as defined in the ONNX model, is therefore ``[micro_batch_size, *sample_shape]``.

  Replication factor
    The number of graphs to be run in parallel over multiple devices.
    The weight gradients from each device will be accumulated before a weight update.
    Also referred to as "device replication factor" or "spatial replication factor".
    This is sometimes called data-parallel execution.

  Accumulation factor
    The weight gradients will be accumulated over this number of micro-batches in series before a weight update.
    Also referred to as "temporal replication factor".

    Accumulation can be thought of as doing replication on a single device.

  Batch size
    This is defined as ``micro-batch size * replication factor * accumulation factor``.
    This is the number of samples per weight update.



  Batches per step
    The number of batches to run in a single call to ``Session::run``.
    For an ``InferenceSession`` this is equal to the number of executions of the model.
    For a ``TrainingSession`` this is equal to the number of weight updates.

  Step size
    This is defined as ``batch size * batches per step``.
    This is the number of samples per step.

  Input data shape
    Inputs to a ``session.run()`` call are read in with the assumption that data is arranged in the shape:

    ``[batches_per_step, accl_factor, repl_factor, micro_batch_size, *sample_shape]``

    However, there is no constraint of the shape of the input array, except that it has the correct number of elements.

  Virtual graph
    Subdivision of a graph to a subset of IPU tiles. While ``virtualGraphId`` in PopART refers to the graph associated with an IPU, the virtual graph can be subdivided further into tile sets ``IO`` and ``Compute``.

  Off-chip streaming memory
    Large pool of memory not located on the IPU that can be used to offload tensors from the IPU.
    Tensor location settings can be used to specify which tensors should be offloaded. Decreases on-chip memory usage.

  RTS
  Replicated tensor sharding
    Eliminate storage and compute redundancy by sharding a weight, optimizer state or accumulator tensor equally across ``N`` data parallel replicas.
    When the replicas require the full tensor, ``ReplicatedAllGatherOp`` is used.
    Increases performance, especially in conjunction with off-chip remote memory, decreases on-chip memory usage.

  Pipelining
    Assigning parts of the computational graph to different pipeline stages so that they can run in parallel.

  Pipeline cycle
    - The time step with which micro-batches move through the pipeline.
    - All Pipeline Stages are executed once within one Pipeline Cycle, in parallel (except for some serialisation if multiple Pipeline Stages are mapped to a single IPU).

  Pipeline phase
    The three phases a pipeline consists of:

    - Fill-phase (ramp up, adding a new pipeline stage in each iteration)
    - Main-phase (running all pipeline stages in parallel)
    - Flush-phase (ramp down, discontinue a pipeline stage in each iteration)

  Pipline stage
    - A partition of the graph that can be (in theory) executed in parallel with any other pipeline stage (although multiple pipeline stages mapped to a single IPU will in practice run serially).
    - Each Pipeline Stage operates on a single and separate micro-batch of data.
    - Excluding inter-IPU copies and host IO, operations on a Pipeline Stage have no dependencies on other Pipeline Stages within a single Pipeline Cycle.
    - Pipeline Stages cannot span IPU boundaries.
