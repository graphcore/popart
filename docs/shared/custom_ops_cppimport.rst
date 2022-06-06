.. _sec_custom_op_cppimport

Auto-compiling custom operations with cppimport
-----------------------------------------------

Although it is possible to manually compile custom operations, we recommend
using :ref:`cppimport <https://pypi.org/project/cppimport/>` to
automatically compile the C++ code of custom operations. This
makes for a much improved user experience of custom operations.
Conveniently, :ref:`cppimport <https://pypi.org/project/cppimport/>`
will detect when your C++ source has changed and only compile when needed;
it is no longer necessary to manually compile your custom operation every
time you make a change.

In the remainder of this section, we assume all of your custom operation's C++
code (not including the Python wrapper in
:numref:`sec_custom_op_python_wrapper`) lives in a single ``.cpp`` file called
``leaky_relu_op_impl.cpp``.

To use :ref:`cppimport <https://pypi.org/project/cppimport/>`, you first
need to add the following comment to the first line of
``leaky_relu_op_impl.cpp``:

.. literalinclude:: files/leaky_relu_op_impl.cpp
    :language: cpp
    :name: leaky_relu_grad_op_class
    :lines: 1
    :linenos:

This comment is used by
:ref:`cppimport <https://pypi.org/project/cppimport/>` as an opt-in
mechanism to specify the file is meant to be used with ``cppimport``, and *must*
appear on the first line.

Then, at the end of the file, include the following multi-line comment:

.. literalinclude:: files/leaky_relu_op_impl.cpp
    :language: cpp
    :name: leaky_relu_grad_op_class
    :start-after: cppimport-compilation begin
    :end-before: cppimport-compilation end

This contains all the information
:ref:`cppimport <https://pypi.org/project/cppimport/>` needs to compile
your operation.
