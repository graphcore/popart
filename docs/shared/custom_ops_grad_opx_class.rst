.. _sec_custom_op_grad_opx_class:

Gradient opx class
------------------

Similar to the forward operation we need a ``LeakyReluGradOpx`` class that
implements the semantics of the gradient operation of Leaky ReLU. Again, we do
this by implementing a C++ class that derives from
:cpp:class:`~popart-cpp-api:popart::popx::Opx`. Gradient operations are
implemented just like the forward operations - there are no additional functions
that need to be implemented.

.. literalinclude:: files/leaky_relu_op_impl.cpp
    :language: cpp
    :name: leaky_relu_grad_op_class
    :caption: Opx implementation of Leaky ReLU's gradient operation
    :start-after: GradOpx begin
    :end-before: GradOpx end

.. only:: html

    :download:`Download <files/leaky_relu_op_impl.cpp>`
