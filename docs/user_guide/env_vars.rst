Environment variables
---------------------

There are several environment variables which you can use to control the
behaviour of PopART.

POPART_LOG_LEVEL
~~~~~~~~~~~~~~~~~

This controls the amount of information written to the log output.

The logging levels, in decreasing verbosity, are:

* TRACE
* DEBUG
* INFO
* WARN
* ERR
* CRITICAL
* OFF

The default is "OFF".

POPART_LOG_DEST
~~~~~~~~~~~~~~~~

This variable defines the output for the logging information. The value can be stdout, stderr or a file name.

The default, if not defined, is "stderr".

.. TODO: POPART_LOG_CONFIG
.. ~~~~~~~~~~~~~~~~~~


POPART_DOT_CHECKS
~~~~~~~~~~~~~~~~~~

Supported values:

- FWD0
- FWD1
- BWD0
- PREALIAS
- FINAL

These values may be combined using ":" as a separator.
The example below shows how to set ``POPART_DOT_CHECKS`` to export
dot graphs ``FWD0`` and ``FINAL``.

.. code-block:: console

  export POPART_DOT_CHECKS=FWD0:FINAL

The values of ``POPART_DOT_CHECKS`` will be combined with any values
that are found in the session options.

POPART_TENSOR_TILE_MAP
~~~~~~~~~~~~~~~~~~~~~~~

The mapping of tensors to tiles in the session can be saved to a file by setting this variable
to the name of a file. The tensor tile map will be written in JSON format.

The tensor tile map will be saved when you call ``Session::prepareDevice``.
The below show how to set the variable to save the tensor tile map to ``ttm.js``.

.. code-block:: console

  export POPART_TENSOR_TILE_MAP=ttm.js
