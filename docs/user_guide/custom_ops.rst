Custom operators
================

This section explains how to implement a custom operator (op) in PopART. Code
from the `Leaky ReLU custom op example
<https://github.com/graphcore/examples/tree/master/code_examples/popart/custom_operators/leaky_relu_example>`_
in the Graphcore GitHub repository will be used to illustrate the concepts.

Overview
--------

You need to write some C++ classes to implement the op.

One is an implementations of the op as PopART's intermediate representation
(IR). This is used during PopART's compilation process to transform and optimise
the graph. There is also a Poplar implementation of the op, which provides the
code that is run when the graph is executed. If the op will be used for
training, then you also need gradient versions of these.

These classes are compiled to create a shared object library that can be
linked with a Python program when it is run.

You also need to define an "operator identifier". This consists of a unique
combination of domain, operator name and operator version strings. This
is used to register the custom op classes with PopART so that it can be used.

There are two ways of using the new custom op: from the builder
API or from an ONNX file.

* **Builder API**: You can include the new op with the builder API using the
  domain, op name and op version that match the custom op definition.

* **ONNX file:** You can reference the op from an ONNX file using a
  ``NodeProto`` definition that matches the custom op definition.

The custom op will then be instantiated from the shared object library and
treated like any other op in PopART.

You can also provide an "op definition" when you register the custom op. PopART
will use that to check that the correct inputs, outputs and attributes are
provided, and are of the expected types.

.. TODO: the LeakyRelu example of OpDefinition is just `{}` - do we have an
   example that shows inputs, outputs and attribute types?


Custom op classes
~~~~~~~~~~~~~~~~~

The two key base classes in PopART that define an op are:

- ``Op``: the intermediate representation (IR) of an op in PopART. This
  provides methods that are called during PopART's optimisation passes and
  transformations of the compute graph. This representation of the op is
  decoupled from the Poplar implementation.

- ``Opx``: a Poplar implementation of the op. This is the code that will
  actually be run on the IPU.

If the op is required for training, then a ``GradOp`` and ``GradOpx`` must also
be defined for the gradient operation (see :numref:`fig_custom_op_OpGradOpOp`
and :numref:`fig_custom_op_OpGradOpOpx`).

To make these classes visible to PopART, you must instantiate ``OpCreator`` and
``OpxCreator`` objects. These map from the string identifier of the new op
(for example, "LeakyRelu"; see :numref:`define_op_identifier`) to constructors for
your newly-defined ``Op`` and ``Opx`` C++ classes.

.. figure:: images/custom_op_OpGradOpOp.png
  :align: center
  :width: 50%
  :name: fig_custom_op_OpGradOpOp

  Op class diagram

.. figure:: images/custom_op_OpGradOpOpx.png
  :align: center
  :width: 50%
  :name: fig_custom_op_OpGradOpOpx

  Opx class diagram

These classes are compiled to create a shared object library that can be
dynamically linked into the Python program at runtime, as shown below:

.. code-block:: python

  import ctypes

  ctypes.cdll.LoadLibrary(so_path)

You can see how this is done in the `LeakyReLU example
<https://github.com/graphcore/examples/blob/master/code_examples/popart/custom_operators/leaky_relu_example/run_leaky_relu.py#L55>`_.


Implementing a custom op
------------------------

Some of the examples in the GitHub repository have a single C++ file that
defines all of the classes for a custom op. Although this can make it easier to
see everything in one place, it can be more difficult to follow. So, in this
section the main elements of the ``LeakyRelu`` example are extracted with some
more detailed descriptions of each method.


The op class
~~~~~~~~~~~~

The `Op
<https://github.com/graphcore/popart/tree/sdk-release-1.3/willow/include/popart/op.hpp#L59>`_
base class provides the methods necessary for the PopART IR passes and
transformations.

The main methods that you need to override or implement are:

* Attributes should be passed into the constructor and corresponding accessors
  defined.

* ``clone()``: returns a copy of the op. Usually, this means returning a
  ``std::make_unique`` copy of the op. This must be implemented.

* ``setup()``: sets the shape and type of the arguments to the op. This must set
  the type and shape information for all the output ``TensorInfo`` objects (see
  `tensorinfo.hpp
  <https://github.com/graphcore/popart/tree/sdk-release-1.3/willow/include/popart/tensorinfo.hpp#L163>`_).

* ``appendAttributes()``: appends attributes when serialising the op to a
  stream. This is used for some debugging purposes but also for generating the
  PopART IR hash. This hash is used to determine whether a Poplar cache can be
  reused so it is important that op attributes which may alter the Poplar
  compilation are appended to this stream. If this method is overridden, then it
  must also call the base class method.

* ``appendOutlineAttributes()``: determines which ops are functionally
  equivalent during outlining.

* ``getGradOps()``: returns a vector of ``GradOp`` object for each ``Op`` in
  the forward graph to automatically generate the backward pass. There can be a
  separate grad op for each input (this is usually cleaner to implement)
  or a single grad op that generates gradients for all inputs.

  The mapping from the index of each output tensor of the grad op to the index
  of each input tensor of the non-grad op is configured using the
  ``gradOutToNonGradIn()`` method that should be overridden in the ``GradOp``
  definitions (see below).

* ``getSubgraphValue()``: this is used by outlining algorithm to determine
  whether or not to outline ops. There are high and low bounding values
  retrieved by ``getHighSubgraphValue()`` (for expensive ops such as Conv) or
  ``getLowSubgraphValue()`` (for inexpensive ops such as Relu).

* ``requiresRandomSeed()``: this is set to false by default. This should be
  overridden and set to true if an IPU random seed tensor is required by the op.
  If so it will be connected to ``inTensor(getSeedInIndex())`` by the IR
  process.

* ``inplacePriorityDefault()``: if the op can be replaced by an in-place
  variant of itself, this method should be overridden to return a vector of
  ``<OperatorIdentifier, float>`` tuples in descending order of preference. For
  example, the LeakyReLU implementation for this is:

  .. code-block:: cpp

    return {{Onnx::CustomOperators::LeakyReluInplace, 10}};

* ``getInplaceVariant()``: this is called to instantiate a particular in-place
  variant of the Op with a specified ``OperatorIdentifier`` from the vector
  returned by ``inplacePriorityDefault()``.

The op class
~~~~~~~~~~~~

.. literalinclude:: files/custom_op/custom_op.cpp
  :start-after: Op begin
  :end-before: Op end

The grad op class
~~~~~~~~~~~~~~~~~

.. literalinclude:: files/custom_op/custom_op.cpp
  :start-after: GradOp begin
  :end-before: GradOp end


The opx class
~~~~~~~~~~~~~

The `Opx
<https://github.com/graphcore/popart/tree/sdk-release-1.3/willow/include/popart/popx/opx.hpp>`_
class provides a ``grow()`` function that implements the corresponding ``Op``
definition as Poplar or PopLibs calls using the provided ``program::Sequence``.
Since ``OpxCreator`` uses a generic constructor, you should also check that the
``Op`` passed in is of the expected type and matches the ``OperatorIdentifier``.

.. literalinclude:: files/custom_op/custom_op.cpp
  :start-after: Opx begin
  :end-before: Opx end

The grad opx class
~~~~~~~~~~~~~~~~~~

.. literalinclude:: files/custom_op/custom_op.cpp
  :start-after: GradOpx begin
  :end-before: GradOpx end


Making the op available to PopART
---------------------------------

After you have written the classes that implement the op, you will need to make
the op available to PopART. This means defining an op identifier and using the
op creator class to register the op with PopART.

.. _define_op_identifier:

Define the op identifier
~~~~~~~~~~~~~~~~~~~~~~~~

The first step is to define an ``OperatorIdentifier`` with the domain, op name
and op version so that the op can be be found by the ``builder.customOp()`` call
in PopART or by a reference to the op in an ONNX file.

The ``OperatorIdentifier`` is a structure with the components ``domain``,
``opName`` and ``opVersion``.

For example, from `leaky_relu_custom_op.cpp
<https://github.com/graphcore/examples/blob/master/code_examples/popart/custom_operators/leaky_relu_example/leaky_relu_custom_op.cpp#L13>`_:

.. literalinclude:: files/custom_op/custom_op.cpp
  :start-after: OpId begin
  :end-before: OpId end

Define the op creator
~~~~~~~~~~~~~~~~~~~~~

The op creator registers the the new Op with PopART.

The ``OperatorIdentifier`` and a factory function that generates the new Op
class are passed to the constructor of ``OpCreator`` to create a mapping. When
your program loads the shared object library, this ``OpCreator`` is instantiated
and registers the new Op.

You can also pass in an ``OpDefinition`` that allows the ``inputs``,
``outputs`` and ``attributes`` to be checked against those provided in the model
implementation.

The ``GradOp`` class will be implicitly created when the overridden method
``getGradOps()`` is called during the backwards pass.

.. literalinclude:: files/custom_op/custom_op.cpp
  :start-after: OpCreator begin
  :end-before: OpCreator end


Define the opx creator
~~~~~~~~~~~~~~~~~~~~~~

You add the ``Opx`` definitions in a similar to the ``Op``. In this case, a
generic constructor of the Opx is always used of the form ``Opx(Op *op, Devicex
*devicex)``. For example:

.. literalinclude:: files/custom_op/custom_op.cpp
  :start-after: OpxCreator begin
  :end-before: OpxCreator end

ONNX schema and shape inference
-------------------------------

To enable ONNX to use the op as part of an ONNX model, you must define a
schema for it. This includes inputs, outputs, domain, and versions.

To register
an ``OpSchema``, you can use the macro ``ONNX_OPERATOR_SCHEMA(name)`` and then
append the various functions in the class. See `schema.h
<https://github.com/onnx/onnx/blob/master/onnx/defs/schema.h>`_ for more
examples.

.. literalinclude:: files/custom_op/custom_op.cpp
  :start-after: Onnx begin
  :end-before: Onnx end


In the same namespace you can define the shape inference for the op. This allows
ONNX to infer from the shape of the inputs the shape of the outputs. With simple
operations, such as this example, the output shape is the same as the first
input, so you can use the ONNX function ``propagateShapeAndTypeFromFirstInput``
from `shape_inference.h
<https://github.com/onnx/onnx/blob/master/onnx/defs/shape_inference.h>`_.

There
are other methods to use for shape inference in ONNX contained in that
header. For example, numpy-style broadcasting, shape from attributes, and so on.
Defining shape inference is optional, however you may encounter issues with
operations later in your model if ONNX is not able to infer the input shape of
an operation from earlier inputs.

Using the op in a program
-------------------------

The op can be referenced, using the values in the op identifer, in a Python
program using the ``builder``. For example, from `run_leaky_relu.py
<https://github.com/graphcore/examples/blob/master/code_examples/popart/custom_operators/leaky_relu_example/run_leaky_relu.py>`_:

.. code-block:: python

  output_tensor = builder.customOp(opName="LeakyRelu",
                                   opVersion=6,
                                   domain="ai.onnx",
                                   inputs=[input_tensor],
                                   attributes={"alpha": alpha})[0]



Or the op can be referenced from an ONNX file using a `NodeProto
<https://github.com/onnx/onnx/blob/master/onnx/onnx.proto#L191>`_
definition that matches the domain, name and version of the op.

.. TODO: Do we have an example of this?
