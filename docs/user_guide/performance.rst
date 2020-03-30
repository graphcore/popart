Performance optimisation
========================

.. TODO: Add sections on pipelining, recomputation,
.. automatic virtual graphs, replication.

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
    if args.execution_mode == "PINGPONG":
        sync_pattern = popart.SyncPattern.PingPong
    device = popart.DeviceManager().acquireAvailableDevice(
        request_ipus,
        1216,
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

* **PingPong:** One sync group contains all the IPUs. The other sync group is
  used independently by sets of IPUs, for example A+C and B+D. This means that
  each subset can communicate independently of each other. The two groups of
  IPUs can then alternate between host I/O and processing.

For more information on how the sync groups are used by the Poplar framework,
please refer to the `Poplar and Poplibs User Guide
<https://documents.graphcore.ai/documents/UG1/latest>`_.
