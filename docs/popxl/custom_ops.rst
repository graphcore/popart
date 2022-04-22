.. _sec_custom_operations:

Custom operations
=================

.. note::
    PopXL custom operations are built on top of PopART custom operations
    (see the :doc:`popart-user-guide:index`). The main C++ code is shared
    between PopART and PopXL, but there are subtle differences. You do
    not require ONNX creators or ONNX-based shape inference for PopXL and,
    conversely you do need to add Python bindings and wrappers for PopXL, which
    are not needed for PopART custom operations. It is good practice to make
    your custom operations reusable in both frameworks.

PopXL ships with a large number of built-in operations as standard (see
:numref:`sec_supported_operations`). However, in addition to this, PopXL has a
mechanism to add *custom operations*. You can use this mechanism when you are
unable to express the semantics you need via built-in operations

This section explains how to add custom operations to your model via an example
operation: a *Leaky ReLU*. This operation is akin to a conventional ReLU
operation except that negative inputs produce small, negative outputs,
multiplying negative values by a small non-zero scalar, :math:`\alpha`. That is,
the Leaky ReLU operation applies the following arithmetic element-wise:


.. math::

   \text{LeakyReLU}(x) =
   \begin{cases}
     x        &\text{ if } x \geq 0 \\
     \alpha x &\text{ if } x < 0
   \end{cases}

Creating and using custom operations requires some environment setup and
requires implementing a number of C++ types and Python bindings.

.. include:: custom_ops_environment.rst
.. include:: custom_ops_param_struct.rst
.. include:: custom_ops_op_class.rst
.. include:: custom_ops_opx_class.rst
.. include:: custom_ops_grad_op_class.rst
.. include:: custom_ops_grad_opx_class.rst


.. _sec_custom_op_python_bindings:

Python bindings
---------------

.. warning::

   The Python binding function used in this section is experimental and may
   change in future API updates.

It is necessary to define Python bindings for your custom operation's C++
code, so that the custom operations can be used in Python. We use the
:ref:`Pybind11 <https://github.com/pybind/pybind11>` library to create these
bindings, using an experimental template function
``makeParameterizedOpBindings`` as follows:

.. literalinclude:: files/leaky_relu_op_impl.cpp
    :language: cpp
    :name: leaky_relu_grad_op_class
    :caption: Intermediate representation of Leaky ReLU's gradient operation
    :lines: 223-237

.. only:: html

    :download:`Download <files/leaky_relu_op_impl.cpp>`


The above binding gives us a Python module named ``leaky_relu_op_impl``, but
this module isn't very user-friendly. In the next section we therefore make an
easy-to-use Python wrapper that *is* user-friendly.

.. _sec_custom_op_python_wrapper:

Python wrapper
--------------

.. warning::

   This Python wrapper solution uses some internal PopXL definitions that
   likely will change in future API updates.

The last remaining step is to define a user-facing Python function which uses
the Python bindings to provide a nice clean Pythonic interface for adding a
Leaky ReLU to the IR:

.. literalinclude:: files/leaky_relu_op.py
    :language: python
    :name: leaky_relu_grad_op_class
    :caption: PopXL Python wrapper for Leaky ReLU
    :lines: 2-10, 16-58

.. only:: html

    :download:`Download <files/leaky_relu_op.py>`

Note that in :numref:`leaky_relu_grad_op_class`, the module name
``leaky_relu_op_impl`` is based on the name of the
:ref:`Pybind11 <https://github.com/pybind/pybind11>` module in
:numref:`sec_custom_op_python_bindings`.

.. include:: custom_ops_cppimport.rst

.. _sec_custom_op_using

Using your custom operation
---------------------------

Finally, to use your custom operation, as highlighted in
:numref:`leaky_relu_use`, just import the ``leaky_relu`` Python
wrapper function your defined in :numref:`sec_custom_op_python_wrapper`. Then,
you can use this function much like you can built-in PopXL operations:

.. literalinclude:: files/run_leaky_relu.py
    :language: cpp
    :name: leaky_relu_use
    :caption: Using a custom operation in PopXL
    :emphasize-lines: 8, 35
    :lines: 2-49

.. only:: html

    :download:`Download <files/run_leaky_relu.py>`
