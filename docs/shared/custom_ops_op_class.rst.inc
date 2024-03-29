.. Copyright (c) 2022 Graphcore Ltd. All rights reserved.

.. _sec_custom_op_op_class:

Operation class
---------------

The next step is to implement a C++ class that derives from
:external+popart-user-guide:cpp:class:`popart::Op`. This derived class will be the type
used to represent your custom operation in the IR:

.. literalinclude:: files/leaky_relu_op_impl.cpp
    :language: cpp
    :name: leaky_relu_op_class
    :caption: Intermediate representation of Leaky ReLU
    :start-after: Op begin
    :end-before: Op end

.. only:: html

    :download:`Download <files/leaky_relu_op_impl.cpp>`

Instead of subclassing :external+popart-user-guide:cpp:class:`popart::Op` directly,
the ``LeakyReluOp`` class derives from the class template
:external+popart-user-guide:cpp:class:`popart::ParameterizedOp`. Using this class template
means a number of the :external+popart-user-guide:cpp:class:`popart::Op` virtual functions
are implemented automatically (for example:
:external+popart-user-guide:cpp:func:`popart::Op::clone`,
:external+popart-user-guide:cpp:func:`popart::Op::appendAttributes`).

All custom operations should implement a static ``defaultOperatorId`` function
returning an :external+popart-user-guide:cpp:struct:`popart::OperatorIdentifier` object
that uniquely identifies your operation. This is a function that is assumed to
exist when you generate Python bindings.

Additionally, a custom operation should implement the virtual function
:external+popart-user-guide:cpp:func:`popart::Op::setup`. This is a function that is used
to determine the shape and type of the tensors that your custom operation
produces. This function requires you to set, for each output that your custom
operation produces, the :external+popart-user-guide:cpp:class:`popart::TensorInfo` object
at :code:`outInfo(n)`, where :code:`n` is the index of the output. Note that a
:external+popart-user-guide:cpp:class:`popart::TensorInfo` object holds both type and shape
information. Typically, the :external+popart-user-guide:cpp:class:`popart::TensorInfo`
object for outputs is identical or in part derived from the
:external+popart-user-guide:cpp:class:`popart::TensorInfo` object for inputs. You can get
the :external+popart-user-guide:cpp:class:`popart::TensorInfo` object for an input at index
:code:`n` via :code:`inInfo(n)`.

In our example, the Leaky ReLU has one input at index 0 and one output, also
at index 0. The output shape and size matches exactly that of the input,
resulting in the following implementation:

.. literalinclude:: files/leaky_relu_op_impl.cpp
    :language: cpp
    :name: leaky_relu_op_class_setup
    :lines: 106-109

Next, to add support for the :ref:`autodiff <autodiff>` transform, it is
necessary to implement a :external+popart-user-guide:cpp:func:`popart::Op::getGradOps`
function. In our IR, gradients are generated by operations themselves, and these
operations are distinct from operation in the forward pass. Hence, if you need a
custom operations to support :ref:`autodiff <autodiff>`, you will probably want
to also define one or more custom gradient operations. A typical pattern is
that the gradients for a custom operation can be defined by a single custom
gradient operation. We will stick to this pattern here, but there are other ways
of achieving the same result. We will define the custom gradient operation for
Leaky ReLU in
:numref:`sec_custom_op_grad_op_class`.


.. note::

   In addition to the functions defined above, the
   :external+popart-user-guide:cpp:class:`popart::Op` base class has a number of additional
   virtual methods that may be helpful to advanced users. For Leaky ReLU, we
   have implemented the bare minimum and rely on the default implementations of
   these virtual functions. For advanced use cases, please read the
   :external+popart-user-guide:cpp:class:`Op documentation<popart::Op>`.
   Most of these methods are for
   enabling other transforms to work with the op, like how ``getGradOps`` is
   overridden to enable ``autodiff``.
