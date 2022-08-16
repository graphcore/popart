.. _popart_env_vars:

Environment variables
=====================

There are several environment variables which you can use to control the
behaviour of PopART.

Logging
-------

PopART can output information about its activity as described in :any:`popart_logging`.
You can control the default level of logging information using environment variables.

POPART_LOG_LEVEL
~~~~~~~~~~~~~~~~~

This controls the amount of information written to the log output for all modules. Finer control
can be achieved using :any:`POPART_LOG_CONFIG`.


POPART_LOG_DEST
~~~~~~~~~~~~~~~~

This variable defines the output for the logging information. The value can be "stdout", "stderr" or a file name.

The default, if not defined, is "stderr".

.. _POPART_LOG_CONFIG:

POPART_LOG_CONFIG
~~~~~~~~~~~~~~~~~

If set, this variable defines the name of a configuration file which specifies the logging level for each module.
This is a JSON format file with pairs of module:level strings.
For example, a file called ``conf.py`` can be specified by setting the environment variable:

.. code-block:: console

  export POPART_LOG_CONFIG=conf.py

To set the logging level of the devicex and session modules, ``conf.py`` would contain:

.. code-block:: json

  {
    "devicex":"INFO",
    "session":"WARN"
  }

These values override the value specified in POPART_LOG_LEVEL.


Generating DOT files
---------------------

POPART_DOT_CHECKS
~~~~~~~~~~~~~~~~~~

PopART can output a graphical representation of the graph, in DOT format.
This variable controls what DOT files which will be created.
For a DOT file to be created the variable must either match the ``check``
variable of a ``dotCheckpoint`` call in the code, or ``POPART_DOT_CHECKS``
must be set to ALL.
In the latter case a DOT file will be created for all ``dotCheckpoint``s in
the code.

The following ``check`` values are already present when preparing the IR with the traditional API:

- FWD0
- FWD1
- BWD0
- PREALIAS
- FINAL

If using the PopXL API, only FINAL will be present.
However, the user can add a ``dot_checkpoint`` using any ``check`` name anywhere during the creation of the model.
The following example shows how the user can add checkpoints after ops:

.. code-block:: python

  c = ops.add(a, b)
  ir.dot_checkpoint("ADD1")
  d = ops.mul(a, c)
  ir.dot_checkpoint("MULTIPLY")

The ``POPART_DOT_CHECKS`` may be combined using ":" as a separator.
The example below shows how to set ``POPART_DOT_CHECKS`` to export
DOT graphs for the FWD0 and FINAL stages.

.. code-block:: console

  export POPART_DOT_CHECKS=FWD0:FINAL

The values in ``POPART_DOT_CHECKS`` will be combined with any values
that are defined in the session options.



Inspecting the Ir
-----------------

POPART_IR_DUMP
~~~~~~~~~~~~~~

If set, this variable defines the name of a file where the serialised ir will be written.
The ir will be written either at the end of the ir preparation phase, or when an exception
is thrown during the ir preparation phase.
