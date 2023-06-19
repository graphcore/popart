.. _setup_quick_start:

Setup quick start
=================

This section describes how to set up your system to start using PopART.

Ensure you have completed the steps described in the `getting started guide for your system <https://docs.graphcore.ai/en/latest/getting-started.html>`__ before completing the steps in this section.

.. _sec_quick_enable_sdk:

Enable Poplar SDK
-----------------

You need to enable the Poplar SDK before you can use PopXL.

.. code-block::

    $ source [path-to-sdk]/enable

where ``[path-to-sdk]`` is the path to the Poplar SDK.

You can verify that Poplar has been successfully set up by running:

.. code-block:: console

  $ popc --version

This will display the version of the installed software.

Create and enable a Python virtual environment
-----------------------------------------------

It is recommended that you work in a Python virtual environment. You can create and activate a virtual environment as follows:

.. code-block::

    $ python[pv_ver] -m venv ~/[base_dir]/[venv_name]
    $ source ~/[base_dir]/[venv_name]/bin/activate

where ``[base_dir]`` is a location of your choice and ``[venv_name]`` is the name of the directory that will be created for the virtual environment. ``[py_ver]`` is the version of Python you are using and it depends on your OS.

You can get more information about the versions of Python and other tools supported in the Poplar SDK for different operating systems in the :doc:`release-notes:index`.  You can check which OS you are running with ``lsb_release -a``.
