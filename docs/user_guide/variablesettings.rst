Grouping graph replicas
-----------------------

This section details how to use the :py:class:`popart.VariableSettings` for
the purpose of grouping tensor weights across replicas.
For a detailed description of what a replica is, refer to the
:ref:`ipu-programmers-guide:replication section` in the :doc:`ipu-programmers-guide:index`.

Concept
~~~~~~~

When using graph replication, variables by default contain the same value on all
replicas. With the help of :py:class:`~popart.VariableSettings` we can assign
distinct tensor values to (and retrieve tensor values from) groups of
replicas, removing the limitation of assigning the same value to all replicas.

VariableSettings
~~~~~~~~~~~~~~~~

The :py:class:`~popart.VariableSettings` object is initialized with two values:
a :py:class:`~popart.CommGroup` and a :py:class:`~popart.VariableRetrievalMode`.
The ``CommGroup`` is used to set the communication groups this tensor is divided into across replicas,
and the ``VariableRetrievalMode`` lets you specify how to retrieve variables from the replicas.

The :py:class:`~popart.CommGroup` class in turn is composed of the :py:class:`~popart.CommGroupType` enum,
and the size of each group. Possible values for ``CommGroupType`` are:

    * :py:attr:`popart.CommGroupType.All`:
        This is the default group type, with this
        grouping all replicas use the same variable values. This ``CommGroupType``
        ignores group size. An example of such a grouping is in :numref:`all_consecutive`.

        .. list-table:: Replication factor 16, CommGroupType = All
            :width: 80%
            :widths: 10 40
            :header-rows: 1
            :name: all_consecutive

            * - Group
              - Replicas
            * - 0
              - 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

    * :py:attr:`popart.CommGroupType.Consecutive`:
        With this ``CommGroupType``
        replicas will be grouped together with adjacent replicas (based on replica index) with
        each group having a size equal to the size the CommGroup is instantiated with. An example
        of such a grouping is in :numref:`varset_consecutive`.

        .. list-table:: Replication factor 16, CommGroupType = Consecutive, CommGroup size = 4
            :width: 80%
            :widths: 10 40
            :header-rows: 1
            :name: varset_consecutive

            * - Group
              - Replicas
            * - 0
              - 0, 1, 2, 3
            * - 1
              - 4, 5, 6, 7
            * - 2
              - 8, 9, 10, 11
            * - 3
              - 12, 13, 14, 15

    * :py:attr:`popart.CommGroupType.Orthogonal`:
        Orthogonal groups, unlike Consecutive, assign replicas such that the first member
        of a group has replica-index same as the group-index, and following members are
        assigned with a stride from the previous equal to the number of groups.
        An example to visualize this is in :numref:`varset_orthogonal`.

        .. list-table:: Replication factor 16, CommGroupType = Orthogonal, CommGroup size = 4
            :width: 80%
            :widths: 10 40
            :header-rows: 1
            :name: varset_orthogonal

            * - Group
              - Replicas
            * - 0
              - 0, 4, 8, 12
            * - 1
              - 1, 5, 9, 13
            * - 2
              - 2, 6, 10, 14
            * - 3
              - 3, 7, 11, 15

    * :py:attr:`popart.CommGroupType.Ungrouped`:
        Ungrouped replicas imply that each replica is in their own group, see :numref:`varset_ungrouped`.

        .. list-table:: Replication factor 16,  CommGroupType = Ungrouped
            :width: 80%
            :widths: 10 40
            :header-rows: 1
            :name: varset_ungrouped

            * - Group
              - Replicas
            * - 0
              - 0
            * - 1
              - 1
            * - 2
              - 2
            * - ...
              - ...
            * - 14
              - 14
            * - 15
              - 15


Instantiating Variables with VariableSettings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before creating variables with ``VariableSettings`` a replication factor must be decided upon, as
different replication factors will change the number of communication groups requiring initialization, and thus the size of
the instantiating buffer size.

``VariableSettings`` can be added to the :py:func:`~popart.Builder.addInitializedInput` or
:py:func:`~popart.Graph.addVarInit` call when initiating your variable.

The initializer buffer used for creating these variables have to be sized such that they initialize each
group individually. This is done by adding an outer dimension to the initializing buffer equal to the
number of groups, the graph-builder will handle the rest. That is to say, a tensor with shape ``[2, 3, 4]``
with :py:class:`~popart.VariableSettings` and a ``replication_factor`` (that is the number of replicas) that results in 4 groups must
be instantiated with shape ``[4, 2, 3, 4]``, where ``[r, ...]`` instantiates the variable on replica ``r``.


Weight input/output
~~~~~~~~~~~~~~~~~~~

When using :py:class:`~popart.PyWeightsIO` to read the value of the weights, the buffer size must match
the size of the initializing data, and if :py:class:`~popart.VariableRetrievalMode` is `AllReplicas` said
outer dimension must match the replication factor.

For example: with a tensor of shape ``[2, 3, 4]``, using replication factor 4 and a
:py:class:`~popart.VariableSettings` with :py:class:`~popart.CommGroup`
(:py:attr:`~popart.CommGroupType.Consecutive`, ``2``) we need a buffer for the :py:class:`~popart.PyWeightsIO` with
the shape

  * ``[2, 2, 3, 4]`` if we use :py:attr:`popart.VariableRetrievalMode.OnePerGroup`.
  * ``[4, 2, 3, 4]`` if we use :py:attr:`popart.VariableRetrievalMode.AllReplicas`.

The on device buffer is populated when using :py:func:`popart.Session.readWeights`.

:numref:`var_settings_example_popart_io`

.. literalinclude:: ../user_guide/files/replication_popart.py
    :language: python
    :emphasize-lines: 37-41
    :caption: Creating buffers for replicas.
    :name: var_settings_example_popart_io

ONNX checkpoints
~~~~~~~~~~~~~~~~

ONNX is not by default aware of the replication factor, thus unless told specifically
the ONNX model will attempt to interpret the outermost dimension as a part of each
replica, usually breaking the model logic in the process.

To accomodate this the builder function: :py:func:`popart.Builder.embedReplicationFactor`
writes the replication factor into the Onnx model as an attribute of the graph.

The builder does not need the replication factor embedded when using
:py:func:`popart.Session.resetHostWeights`
to write a ONNX-file into a new model.
