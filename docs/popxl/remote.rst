.. _sec_remote:

Variables in Streaming Memory
=============================

When the IPU memory is insufficient, you can use :ref:`poplar-user-guide:remote
memory buffers` to store and load data in Streaming Memory. The remote buffer is
often used for the variable tensors and for the intermediate tensors. In this
section, you will see how to use the following in PopXL:

- remote buffers
- remote variable tensors
- replicated tensor sharding variables

Remote buffers
--------------

In PopXL, you can create a remote buffer in the IR by using
:py:func:`remote_buffer(tensor_shape, tensor_dtype, entries) <popxl.remote_buffer>`. The remote buffer contains a number of slots for tensors
(``entries``) with the same shape (``tensor_shape``) and data type
(``tensor_dtype``).

You can then store a tensor ``t`` at the index ``offset`` of a remote buffer
``remote_buffer`` by using the operation :py:func:`remote_store(remote_buffer, offset, t) <popxl.ops.remote_store>`. To load a tensor at the index ``offset``
of the remote buffer ``remote_buffer``, you can use
:py:func:`remote_load(remote_buffer, offset, name) <popxl.ops.remote_load>`. You
can also name the returned tensor with ``name``.

Remote variable tensors
-----------------------

Similarly to creating a variable tensor (:numref:`sec_tensors_variable`), you
can also create a variable tensor located in Streaming Memory by using :py:func:`~popxl.remote_variable`:

.. code-block:: python

    remote_variable(data: Union[HostTensor, float, int],
                    remote_buffer: RemoteBuffer,
                    offset: int = 0,
                    dtype: Optional[dtypes.dtype] = None,
                    name: Optional[str] = None,
                    downcast: bool = True)

The returned variable tensor, with value ``data``, is put at the index
``offset`` of the remote buffer ``remote_buffer``. The data type and shape of
this variable tensor needs to be compatible with those of the remote buffer.

:numref:`code_remote_variable_popxl` shows how to use remote buffers and remote
variable tensors. First, a remote buffer, ``buffer``, is created with only one
entry. Then a remote variable tensor, ``remote_x``, is created with value 1.
This variable is stored at index 0 of the ``buffer``. The value is then loaded
from the remote buffer to the IPU variable ``loaded_x``. The value of
``loaded_x`` is then updated by ``y`` with value 2. The new value of
``loaded_x`` is then stored in the same place, index 0 of ``buffer``, as
``remote_x``. You can check the value of ``remote_x`` by using
``session.get_tensor_data(remote_x)`` after you run a ``session``. Both
``loaded_x`` and ``remote_x`` have the value 3 in this example.

.. literalinclude:: files/remote_variable.py
  :language: python
  :name: code_remote_variable_popxl
  :caption: Example to use remote buffer and remote variable
  :start-after: remote_var begin
  :end-before: remote_var end
  :linenos:
  :lineno-match:

.. only:: html

    :download:`Download remote_variable.py <files/remote_variable.py>`

.. _sec_rts:

Variable tensors for replicated tensor sharding
-----------------------------------------------

You can also create a variable tensor for replicated tensor sharding (RTS) that
is split in equal shards across replicas. See the
:ref:`PopART User Guide <popart-user-guide:replicated tensor sharding (rts)>`
for more information.
Together with the allGather operation
:py:func:`~popxl.ops.collectives.replicated_all_gather`, RTS avoids storing the
same tensor for each replica. The full tensor is stored in Streaming Memory.
After the full tensor is updated on the IPU, it needs to be sharded and/or
reduced again to each replica by using the reduceScatter operation
:py:func:`~popxl.ops.collectives.replicated_reduce_scatter`.

In PopXL, each shard of an RTS variable tensor is stored in its own remote
buffer. To simplify the use of replication, each shard shares the same
representation of its remote buffer. As shown in :numref:`fig_rts`, each buffer
has the same tensor type and tensor shape in each shard. The number of shards is
the same as the number of replicas.

.. figure:: images/RTS_var.png
  :name: fig_rts
  :align: center
  :alt: illustration of rts

  An RTS variable tensor in PopXL

Note that you need to have replication enabled to create an RTS variable tensor.
You can enable replication by setting :py:attr:`~popxl.Ir.replication_factor` to
> 1 (:numref:`sec_data_input_shape`).

There are two ways to create an RTS variable tensor:

#. Store the full variable tensor in Streaming Memory. You can access the
   variable tensor through ``remote_buffer``.

   .. code-block:: python

    remote_replica_sharded_variable(data: Union[HostTensor, float, int],
                                    remote_buffer: RemoteBuffer,
                                    offset: int = 0,
                                    dtype: Optional[dtypes.dtype] = None,
                                    name: Optional[str] = None,
                                    downcast: bool = True) -> Variable

   :py:func:`~popxl.remote_replica_sharded_variable` returns an RTS variable tensor
   that has value ``data`` at the index ``offset`` of remote buffer
   ``remote_buffer``. You need to use :py:func:`~popxl.ops.remote_load` and :py:func:`~popxl.ops.remote_store`
   operations to load and store the variable tensor data to and from the IPU.

#. Store the full variable tensor in Streaming Memory, along with another tensor
   to represent its shards. The tensor representing the shards can be used
   without :py:func:`~popxl.ops.remote_load` and
   :py:func:`~popxl.ops.remote_store` since it is automatically
   loaded from or stored to Streaming Memory.

    .. code-block:: python

        replica_sharded_variable(data: Union[HostTensor, float, int],
                                 dtype: Optional[dtypes.dtype] = None,
                                 name: Optional[str] = None,
                                 downcast: bool = True) -> Tuple[Variable, Tensor]

   In :py:func:`~popxl.replica_sharded_variable`, the variable tensor is still
   created with a remote buffer, as for
   :py:func:`~popxl.remote_replica_sharded_variable`. The number of entries in this
   buffer is the number of elements in the data divided by the number of
   replicas. Each shard is then automatically loaded or stored according to the
   execution context. However, the remote buffer is hidden to provide an
   easier interface. You can use :py:func:`~popxl.remote_replica_sharded_variable` to
   have more flexibility.

The example in the code tab `Remote RTS variable
tensor` shows how to update the value of a remote RTS variable tensor created
with :py:func:`~popxl.remote_replica_sharded_variable`:

   * The remote RTS variable tensor ``remote_x`` is created with a remote buffer
     ``buffer``.
   * Remote load tensor ``remote_x`` to tensor ``loaded_x``.
   * Gather the shards of the tensor ``loaded_x`` to tensor ``full_x``.
   * Update the tensor ``full_x`` in place by adding tensor ``y``.
   * Shard tensor ``full_x`` across replicas to tensor ``updated_shard``.
   * Remote store tensor ``updated_shard`` to index 0, the same place as tensor ``remote_x``, in the remote buffer ``buffer``.

The example in the code tab `RTS variable tensor` shows how
to update the RTS variable tensor created by :py:func:`~popxl.replica_sharded_variable`. In this
example you can see the remote store and load operations are hidden.

   * A remote RTS variable tensor ``remote_x`` and its shards ``loaded_x`` are
     created without specifying a buffer.
   * Then, the shards ``loaded_x`` are updated by adding the sharded tensor
     ``y``.

.. tabs::

   .. group-tab:: **Remote RTS variable tensor**

      .. literalinclude:: files/remote_rts_var.py
         :language: python
         :caption: Example to use remote RTS variable tensor
         :start-after: remote_var begin
         :end-before: remote_var end
         :linenos:
         :lineno-match:

      .. only:: html

         :download:`Download remote_rts_var.py <files/remote_rts_var.py>`

   .. group-tab:: **RTS variable tensor**

      .. literalinclude:: files/rts_var.py
         :language: python
         :caption: Example to use RTS variable tensor
         :start-after: remote_var begin
         :end-before: remote_var end
         :linenos:
         :lineno-match:

      .. only:: html

         :download:`Download rts_var.py <files/rts_var.py>`
