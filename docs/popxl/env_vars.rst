.. _popart_env_vars:

Environment variables
=====================

There are several environment variables which you can use to control the
behaviour of PopXL. Most of these use the underlying PopART library.

.. include:: env_vars_logging.rst

Generating DOT files
---------------------

You can ask PopXL to export DOT files of your graphs at different points during IR construction. This is done using  :py:func:`~popart.Ir.dot_checkpoint`. The following example shows this:

.. code-block:: python

    with g:
        c = ops.add(a, b)
        ir.dot_checkpoint("ADD")
        d = ops.mul(a, c)
        ir.dot_checkpoint("MULTIPLY")
    popxl.transforms.autodiff(g)
    ir.dot_checkpoint("BWD")

By default, PopXL will produce DOT files for all your checkpoints. You can, however, dynamically control which ones are produced using the ``POPART_DOT_CHECKS`` environment variable. Multiple checkpoint names can be passed, separated  ```:``.  For example, we can select only ``ADD`` and ``BWD`` like so:

.. code-block:: console

  $ export POPART_DOT_CHECKS=ADD:BWD

The special value ``ALL`` will select all checkpoints. This is the default value.

The values in ``POPART_DOT_CHECKS`` will be combined with any values
that are defined in the session options.


Caching of compiled executables
-------------------------------

It can take a long time to compile a large graph into an executable for the IPU. You can enable caching of compiled executables to avoid re-compiling the same graph every time it is run.

To enable the cache, set the environment variable ``POPXL_CACHE_DIR`` to the path where the compiled executables for the PopXL graph will be stored. For example:

.. code-block:: console

  $ export POPXL_CACHE_DIR="/tmp/cachedir"

An executable binary file with the extension ``.popef`` will be saved for each Poplar graph required to execute the PopXL program.

The cache does not *manage* the files within the directory. It is your responsibility to delete out-of-date files. No index is kept of the files, so they can be deleted without risk.
