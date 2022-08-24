.. _popart_env_vars:

Environment variables
=====================

There are several environment variables which you can use to control the
behaviour of PopART.

.. include:: env_vars_logging.rst


Generating DOT files
---------------------

PopART can output a graphical representation of the graph, in DOT format.

POPART_DOT_CHECKS
~~~~~~~~~~~~~~~~~~

PopART can export DOT files of your graphs at different points. By default, no DOT files are created, but you can use the ``POPART_DOT_CHECKS`` environment variable to control what DOT files to create. ``POPART_DOT_CHECKS`` is set using predefined checkpoint names. For example to create a DOT file of all checkpoints, you would set:

.. code-block:: console

  export POPART_DOT_CHECKS=ALL

The predefined checkpoint names are:

- ``FWD0``: Initial IR immediately after lowering from ONNX to the IR.
- ``FWD1``: After the pre-alias patterns have been applied to ``FWD0``.
- ``BWD0``: After growing the backward pass (including the optimiser step). Note this happens before optimiser decomposition, so the optimiser will appear as a single special op rather than the many ops that implement it.
- ``PREALIAS``: After pre-alias transforms have been applied to ``BWD0``.
- ``MAINLOOPS``: After the ``MainLoops`` transform has been applied. This transform adds explicit loop ops to the IR for device iterations (:term:`batches per step`) and gradient accumulation.
- ``FINAL``: The final IR after preparation.
- The following checkpoint names only apply if you are using *explicit* :term:`pipelining` (Note: The default is implicit pipelining). Explicit pipelining happens between ``MAINLOOPS`` and ``FINAL``:

  - ``EXPLICITPIPELINEBEFOREOUTLINE``: Before the outline stage of the transform.
  - ``EXPLICITPIPELINEAFTEROUTLINE``: After the outline stage of the transform.
  - ``EXPLICITPIPELINEAFTERDECOMPOSE``: After the decompose stage of the transform.
- ``ALL``: All checkpoints are selected.

Multiple checkpoint names can be passed, and should be separated using ":". For example to create a DOT file for the ``FWD0`` and ``PREALIAS`` checkpoints:

.. code-block:: console

  export POPART_DOT_CHECKS=FWD0:PREALIAS

The values in ``POPART_DOT_CHECKS`` will be combined with any values
that are defined in the session options.

Caching of compiled executables
-------------------------------

It can take a long time to compile a large graph into an executable for the IPU. You can enable caching of compiled executables to avoid re-compiling the same graph every time it is run.

To enable the cache, set the environment variable ``POPART_CACHE_DIR`` to the path where the compiled executables for the PopART graph will be stored. For example:

.. code-block:: console

  $ export POPART_CACHE_DIR="/tmp/cachedir"

An executable binary file with the extension ``.popef`` will be saved for each Poplar graph required to execute the PopART program.

The cache does not *manage* the files within the directory. It is your responsibility to delete out-of-date files. No index is kept of the files, so they can be deleted without risk.



Inspecting the IR
-----------------

You can also set an environment variable to inspect the IR.

POPART_IR_DUMP
~~~~~~~~~~~~~~

If set, ``POPART_IR_DUMP`` defines the name of a file where the serialised IR will be written. For example, to write the serialised IR to the file ``ir_dump.log``, set:

.. code-block:: console

  export POPART_IR_DUMP=ir_dump.log

The IR will be written either at the end of the IR preparation phase, or when an exception is thrown during the IR preparation phase.

Manipulating outliner behaviour for PopLiner
--------------------------------------------

.. note::
  It is assumed that everything in this section will only be used in conjunction
  with PopLiner - the guided partitioning tool.

POPART_POPLINER_OUTLINER_REGEX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used to help reproduce multi-IPU outlining when compiling for a single IPU. This
is helpful for use with PopLiner - the guided partitioning tool. When set, the
value will be used as a regular expression for extracting the layer from the
operation name. The layer must be numerical and must be within parenthesis in
the regular expression for the capture group. The following value is recommended
for this environment variable:

.. code-block:: console

  export POPART_POPLINER_OUTLINER_REGEX="(?:^|\/)(?:[L|l]ayer|blocks|encoder)[\/_\.]?(\d+)"

When set, you should see warning-level log messages to indicate which layer
names are found, allowing you to check that you regular expression works as
intended.

Please remember to unset this environment variable when you no longer wish to
generate a profile for PopLiner.

.. note::

  If you see an error message about remote buffers mapping to multiple
  virtual graphs when using this environment variable, you may need to set the
  encoder start IPU to 0.
