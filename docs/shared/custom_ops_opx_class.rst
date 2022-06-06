.. _sec_custom_op_opx_class:

Opx class
---------

In addition to the ``LeakyReluOp`` class, which is the class used to represent
Leaky ReLU operations in the IR, we also need a ``LeakyReluOpx`` class that
implements the semantics of the operation. You can do this by implementing a C++
class that derives from :cpp:class:`~popart-cpp-api:popart::popx::Opx`:

.. literalinclude:: files/leaky_relu_op_impl.cpp
    :language: cpp
    :name: leaky_relu_opx_class
    :caption: Opx implementation of Leaky ReLU
    :start-after: Opx begin
    :end-before: Opx end

.. only:: html

    :download:`Download <files/leaky_relu_op_impl.cpp>`

The :cpp:func:`~popart-cpp-api:popart::popx::Opx::grow` function is the main
:cpp:class:`~popart-cpp-api:popart::popx::Opx` function you need to implement.
In this function, you are expected to add the code required to
produce your operation's outputs to a Poplar
:cpp:class:`~poplar-api:poplar::program::Sequence` object. Then, once you have described how to compute
Poplar :cpp:class:`~poplar-api:poplar::Tensor` objects for said outputs, you
must use :cpp:func:`~popart-cpp-api:popart::popx::Opx::setOutTensor` to
associate, for each output, the output index with a specific Poplar
:cpp:class:`~poplar-api:poplar::Tensor` object.

Our Leaky ReLU example produces only one output tensor, and that tensor is
output at index 0, so it only calls
:cpp:func:`~popart-cpp-api:popart::popx::Opx::setOutTensor` once.

For further details on how to write Poplar programs, see the
:doc:`poplar-user-guide:index`.

.. note::

   Similar to the :cpp:class:`~popart-cpp-api:popart::Op` base class, the
   :cpp:func:`~popart-cpp-api:popart::popx::Opx::grow` class has a number
   of additional virtual methods that may be helpful to advanced users. In
   particular, the methods that control the unwinding algorithm, which
   determines tensor layouts on device.
