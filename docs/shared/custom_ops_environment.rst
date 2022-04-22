.. _sec_custom_op_environment:

Environment
-----------

To implement a custom operation you first need to configure your environment so
you can compile C++ custom operations and create Python bindings easily. To do
this, ensure you have the following packages in the Python environment that you
use to run your models:

.. code-block:: bash

   pip install cppimport==21.3.7
   pip install pybind11==2.6.2

The ``cppimport`` package automatically compiles and includes C++ code in
Python, avoiding the use of, for example, Makefiles.


The ``pybind11`` package is what we use to provide Python bindings for C++ code.
