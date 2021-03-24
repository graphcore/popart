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

PopART can output a graphical representation of the graph, in DOT format, when it
constructs the intermediate representation (IR). The stages of IR construction
where the DOT files is generated is controlled by this variable.

Supported values:

- FWD0
- FWD1
- BWD0
- PREALIAS
- FINAL

These values may be combined using ":" as a separator.
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
