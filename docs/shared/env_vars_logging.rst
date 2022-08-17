Logging
-------

PopART can output information about its activity as described in :numref:`popart_logging`.
You can control the level of logging information using environment variables.

.. _POPART_LOG_LEVEL:

POPART_LOG_LEVEL
~~~~~~~~~~~~~~~~~

``POPART_LOG_LEVEL`` controls the amount of information written to the log output for all modules. Finer control
can be achieved with :ref:`POPART_LOG_CONFIG`.


POPART_LOG_DEST
~~~~~~~~~~~~~~~~

``POPART_LOG_DEST`` defines the output for the logging information. The value can be ``stdout``, ``stderr`` or a file name.

The default is ``stderr``.

.. _POPART_LOG_CONFIG:

POPART_LOG_CONFIG
~~~~~~~~~~~~~~~~~

If set, ``POPART_LOG_CONFIG`` defines the name of a configuration file which specifies the logging level for each module.
This is a JSON format file with pairs of module:level strings.
For example, you can specify a file called ``conf.py`` by setting the environment variable:

.. code-block:: console

  export POPART_LOG_CONFIG=conf.py

To set the logging level of the ``devicex`` and ``session`` modules, ``conf.py`` would contain:

.. code-block:: json

  {
    "devicex":"INFO",
    "session":"WARN"
  }

These values override the value specified in :ref:`POPART_LOG_LEVEL`.