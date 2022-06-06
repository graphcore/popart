.. _sec_custom_op_grad_op_class:

Gradient operation class
------------------------

To fully support the :ref:`autodiff <autodiff>` transform we next define a
gradient operation for Leaky ReLU, ``LeakyReluGradOp``, which is similar to
the forward operation defined in :numref:`sec_custom_op_op_class`:

.. literalinclude:: files/leaky_relu_op_impl.cpp
    :language: cpp
    :name: leaky_relu_grad_op_class
    :caption: Intermediate representation of Leaky ReLU's gradient operation
    :start-after: GradOp begin
    :end-before: GradOp end

.. only:: html

    :download:`Download <files/leaky_relu_op_impl.cpp>`

We emphasise that gradient operations are operations themselves and hence
require the definition of a static ``defaultOperatorId`` function and
:cpp:func:`~popart-cpp-api:popart::Op::setup`, as explained in
:numref:`sec_custom_op_op_class`. Also, note that ``LeakyReluOp`` and
``LeakyReluGradOp`` share the parameter struct definition, ``LeakyReluParams``.

The job of gradient operations (meaning the set of operations obtained by
calling :cpp:func:`~popart-cpp-api:popart::Op::getGradOps` on a forward
operation) is to perform one step of the
:ref:`chain rule <https://en.wikipedia.org/wiki/Chain_rule>`.

For ``LeakyReluOp``, recall that
:cpp:func:`~popart-cpp-api:popart::Op::getGradOps` returns a single gradient
operation that is ``LeakyReluGradOp``, so we have one operation that has to
produce a partial derivative for a single output of the forward operation.

Performing one step of the chain rule for ``LeakyReluGradOp`` means computing
:math:`\frac{\partial F}{\partial x}` for *some function* :math:`F`
(the partial derivative of :math:`F` with respect to input tensor :math:`x`)
having been given as input
:math:`\frac{\partial F}{\partial \text{LeakyReLU}}`
(the partial derivative of :math:`F` with respect to the output of
:math:`\text{LeakyReLU}`) as well as any forward tensors that it needs,
using the chain rule:

.. math::

    \frac{\partial F}{\partial x} = \frac{\partial F}{\partial \text{LeakyReLU}}\cdot\frac{\partial \text{LeakyReLU}}{\partial x}

The right-hand side of this equation (the partial derivative of
:math:`\text{LeakyReLU}` with respect to its input, :math:`x`)
is mathematically defined as follows:

.. math::

    \frac{\partial \text{LeakyReLU}}{\partial x}(x) =
    \begin{cases}
      1        &\text{ if } x \geq 0 \\
      \alpha   &\text{ if } x < 0
    \end{cases}

Note that this partial derivative needs the forward tensor, :math:`x`, as input
because it is used in the condition.

Our operation, ``LeakyReluGradOp``, needs to calculate this right-hand side
of the chain rule equation and multiply it with the left-hand side,
:math:`\frac{\partial F}{\partial \text{LeakyReLU}}`,
to obtain :math:`\frac{\partial F}{\partial x}`. This left-hand side is given
as an input to  ``LeakyReluGradOp``. Putting this all together, using
:math:`y` to denote :math:`\text{LeakyReLU}` (``LeakyReluOp`` output) and
:math:`y'` to denote the partial derivative
:math:`\frac{\partial F}{\partial \text{LeakyReLU}}`, then
we can express the calculation that ``LeakyReluGradOp`` has to do as follows:

.. math::

   \text{LeakyReLUGrad}(y', x) =
   \begin{cases}
     y'        &\text{ if } x \geq 0 \\
     \alpha y'   &\text{ if } x < 0
   \end{cases}

This definition is what we will use when defining the semantics of
``LeakyReluGradOp`` in :numref:`sec_custom_op_grad_opx_class`. For now, all we
need to understand is that ``LeakyReluGradOp`` consumes two tensor inputs and
produces one tensor output. In terms of data type and tensor shape these inputs
and outputs are all identical to the forward tensor :math:`x`.

This information is what we need when implementing
:cpp:func:`~popart-cpp-api:popart::Op::gradInputInfo` and
:cpp:func:`~popart-cpp-api:popart::Op::gradOutToNonGradIn`, which is an
additional requirement we place on gradient operations.

The :cpp:func:`~popart-cpp-api:popart::Op::gradInputInfo` function tells the
:ref:`autodiff <autodiff>` transform what input tensors an operation requires.
Gradient operations can request to connect
to any inputs or outputs of the forward operation, or can request to connect
to a gradient of an output of the forward operation.

In this instance, the gradient operation ``LeakyReluGradOp`` asks for it's input
at index 0 to be connected with the partial derivative of ``LeakyReluOp``'s
output at index 0 (so
:math:`\frac{\partial F}{\partial \text{LeakyReLU}}`), and ``LeakyReluGradOp``'s
input at index 1 to be connected to the input of ``LeakyReluOp`` at index 0 (so
:math:`x`):

.. literalinclude:: files/leaky_relu_op_impl.cpp
    :language: cpp
    :name: leaky_relu_grad_op_in_connections
    :start-after: GradOp-gradInputInfo begin
    :end-before: GradOp-gradInputInfo end
    :dedent: 2

The :cpp:func:`~popart-cpp-api:popart::Op::gradOutToNonGradIn` function is what
the :ref:`autodiff <autodiff>` transform uses to determine what outputs a
gradient operation produces. Gradient operations produce gradients for some
inputs of the forward graph. To do this, this function must be implemented so it
returns a mapping from gradient operation output indices to forward operation
input indices.

The ``LeakyReluGradOp`` operation only produces one output,
:math:`y'`, at index 0, and it
is the gradient of the ``LeakyReluOp``'s input :math:`\partial x` at index 0.
The appropriate mapping is therefore as follows:

.. literalinclude:: files/leaky_relu_op_impl.cpp
    :language: cpp
    :name: leaky_relu_grad_op_out_connections
    :start-after: GradOp-gradOutToNonGradIn begin
    :end-before: GradOp-gradOutToNonGradIn end
    :dedent: 2
