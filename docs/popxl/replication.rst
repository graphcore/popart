.. _sec_replication_types:

Replication
===========

This chapter describes how to use replication in PopXL.

.. _sec_graph_replication:

Graph replication
-----------------

PopXL has the ability to run multiple copies of your model in parallel.
This is called graph replication. Replication is a means of parallelising your
inference or training workloads.
We call each instance of the graph a replica. The replication factor is the number of replicas in total across all replica groups (see :numref:`sec_replica_grouping`).

This can be set through :py:attr:~popxl.Ir.replication_factor`.


.. _sec_replica_grouping:

Replica grouping
----------------

In PopXL you have the ability to define a grouping of replicas when you create variables.
This grouping is used when you initialise or read the variable. Typically the variables are
initialised and read on a per-group basis. The default behaviour is all replicas belong to one group.

The grouping in question is defined by the :py:class:`~popxl.ReplicaGrouping` object, instantiated with
:py:func:`~popxl.Ir.replica_grouping`.
:py:class:`~popxl.ReplicaGrouping` is initialized with a ``group_size`` and a ``stride``.

The ``group_size`` parameter sets the number of replicas to be grouped together, and
the ``stride`` parameter sets the replica index difference between two members of a group.

.. warning::
    Limitations:

      * When `stride == 1` a requirement is :py:attr:`~popxl.Ir.replication_factor` modulo ``group_size`` equals 0.
      * When `stride != 1` a requirement is ``stride`` times ``group_size`` equals :py:attr:`~popxl.Ir.replication_factor`.

Tables :numref:`varset_consecutive`, :numref:`varset_othagonal`, and :numref:`varset_ungrouped` shows
some different way ``group_size`` and ``stride`` would part up replicas into groups.

.. list-table:: Replication factor 16, group_size = 4, and stride = 1
    :width: 80%
    :header-rows: 1
    :name: varset_consecutive

    * - Group
      - Replicas
    * - 0
      -  0,  1,  2,  3
    * - 1
      -  4,  5,  6,  7
    * - 2
      -  8,  9, 10, 11
    * - 3
      - 12, 13, 14, 15

.. list-table:: Replication factor 16, group_size = 4, and stride = 4
   :width: 80%
   :header-rows: 1
   :name: varset_othagonal

   * - Group
     - Replicas
   * - 0
     - 0, 4,  8, 12
   * - 1
     - 1, 5,  9, 13
   * - 2
     - 2, 6, 10, 14
   * - 3
     - 3, 7, 11, 15

.. list-table:: Replication factor 16, group_size = 1, and stride = 1
  :width: 80%
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

.. _sec_replica_grouping_code_examples:

Code examples
-------------

:numref:`var_settings_example` shows a simple example of the initialization of a few different groupings.


.. literalinclude:: files/replication.py
    :language: python
    :lines: 1-29
    :caption: Example of setting up different variables.
    :name: var_settings_example
    :linenos:
    :lineno-match:

:numref:`remote_var_settings_example` shows an example of using a replica
grouping on a remote variable. The IR has two replicas, and each is its own
group.

.. literalinclude:: files/remote_variable_replica_grouped.py
    :language: python
    :start-after: RemoteVarReplicaGrouped Begin
    :end-before: RemoteVarReplicaGrouped End
    :caption: Example of setting up different remote variables.
    :name: remote_var_settings_example
    :linenos:
    :lineno-match:

There are a couple of specfics to note here. Firstly, you need the ``in_sequence``
context as there is no data-flow dependence between the *inplace* add op and
``remote_store`` op on the same tensor. Secondly, we manually pass the correct
per-replica shape to ``popxl.remote_buffer``. This shape does not have the group
dimension.

.. note::
  If you consider ``v_h`` to be the data for a single variable, this is akin to
  sharding the variable over two replicas. Actually, unless you need to
  AllGather your shards and cannot forgo the CBR optimisation, it is advisable
  to just use replica groupings as shown to achieve sharding. This is because
  the API is much less brittle with respect to what you can do without errors or
  undefined behaviour.

Retrieval modes
---------------

By default, only one replica per group is returned. Usually this is sufficient as all replicas within
a group should be identical. However, if you wish to return all replicas within a group
(for example to test all grouped replicas are the same), set the ``retrieval_mode`` parameter to
``"all_replicas"``  when constructing your variable:


.. literalinclude:: files/replication.py
    :language: python
    :lines: 31-46
    :caption: Example of setting up variables with all_replicas retrieval mode.
    :name: var_settings_example_2
    :linenos:
    :lineno-match:
