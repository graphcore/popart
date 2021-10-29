.. _popart_overlap_io:

Overlapping IO with compute
===========================

By default, PopART will stream data to and from the IPUs serially to computation
on the IPU. For models that are limited by IO, overlapped IO settings allow
a portion of the IPU (IO tiles) to handle streaming data while the compute tiles
work uninterrupted.

Configuring IO tiles
~~~~~~~~~~~~~~~~~~~~

Overlapping IO requires the user to set a number of tiles aside for streaming
data. The number of tiles can be adjusted to optimize transfer throughput,
and according to how much input data needs to be held on the IO tiles until
the compute tiles are ready to receive and process the data.
As a rule of thumb, the IO tiles can be adjusted to the next power of two after
dividing the size of all (overlapped) inputs and anchors divided by the
available SRAM per tile.


.. code-block:: python

  opts = popart.SessionOptions()
  opts.numIOTiles = 128

IO tiles can only be used when virtual graphs are enabled. Virtual graph modes
enable to assign tensors and operations to a subset of IPUs, and within each IPU,
to a subset of tiles (such as compute and IO tiles). The supported modes
are one of:

.. code-block:: python

  opts.virtualGraphMode = popart.VirtualGraphMode.Manual
  opts.virtualGraphMode = popart.VirtualGraphMode.Auto
  opts.virtualGraphMode = popart.VirtualGraphMode.ExecutionPhases

Finally, overlapped IO needs explicit main loops and host copy operations to be
enabled. These settings will become the default in future releases.

.. code-block:: python

  opts.enableExplicitMainLoops = True
  opts.useHostCopyOps = True

Explicit main loops will cause the PopART IR to be transformed such that the
two training loops will become visible:

  - One loop for the device iterations, with ``batches per step`` iterations.

  - One loop for the gradient accumulations, with ``accl_factor`` iterations.

The respective loops will not be present if the ``batches per step`` or
``accl_factor`` are set to less than 2 (i.e. a single iteration).

Host copy operations will represent the streaming of tensors from and to the
host as operations in the IR:

  - ``HostLoadOp`` streams data host -> IPU
  - ``HostStoreOp`` streams data IPU -> host

In this configuration, it is possible to schedule the operations in the IR
such that overlap between compute and IO occurs.

Specify overlap strategy for inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each input can be configured individually. To do so, ``popart.InputSettings``
can be set when adding an input tensor to the model:

.. code-block:: python

  x = builder.addInputTensor(
      popart.TensorInfo("FLOAT", [1, size, size]),
      popart.InputSettings(
         popart.TileSet.IO,
         popart.ExchangeStrategy.OverlapInnerLoop
      ), f"x{i}")

``popart.InputSettings`` specifies which set of tiles the tensor should be
loaded to, and what strategy should be applied. Overlap requires that we use
IO tiles.

Specify overlap strategy for anchors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each anchor can be configured individually. To do so,
``popart.AnchorReturnType`` takes the additional ``TileSet`` and
``ExchangeStrategy`` arguments:

.. code-block:: python

 dataFlow = popart.DataFlow(
     batches_per_step, {
         loss: popart.AnchorReturnType(
         "All",
         popart.TileSet.IO,
         popart.ExchangeStrategy.OverlapInnerLoop),
     })

Again, overlap requires that we use IO tiles here as well. The model will still
be adjusted and compiled if a strategy is selected without using IO tiles,
but it will not improve throughput.

Exchange strategies
~~~~~~~~~~~~~~~~~~~

Available exchange strategies:
  - ``JustInTime``: No overlap, the data is loaded when required by other
    operations.

  - ``OverlapInnerLoop``: Preload values in the previous inner loop iteration
    for the next iteration. If the inner loop iteration count is ``N``, then
    ``N-2`` data exchanges will overlap, while the first and the last will not.
    This is the recommended setting as long as
    ``accl_factor < 2 && batches per step >>> 2`` or ``accl_factor >>> 2``,
    since that will hide most exchanges while not increasing memory requirements
    and graph complexity too much.

  - ``OverlapLoops``: Preload values in the previous loop iteration for the next
    iteration in both the inner and outer loop.
    If the outer loop iteration count is ``M``, now ``M*N-2`` exchanges will
    overlap, but the IR graph becomes more complex and more memory will be
    required on the IO tiles. This is the recommended setting if
    ``accl_factor ~= 2 && batches per step >>> 2``.

  - ``OverlapStep``: Preload both inner loops and across host iterations - not
    supported yet. This will be the recommended setting if
    ``accl_factor ~= 2 && batches per step ~= 2``.
