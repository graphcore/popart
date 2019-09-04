Creating custom operations
--------------------------

This shows how to create a custom operator and include it from an ONNX graph in C++:

.. literalinclude:: ../../examples/cplusplus/custom_op.cpp

If you want to use your custom op in Python, an example is
provided in the file ``examples/custom_op/custom_op.cpp``. This wraps the code
in an ``extern "C"`` wrapper to avoid mangled names when using with Python. Other than replacing the C++ example code,
no more changes to the C++ file are required.

To compile into a shared object file, make sure you have an up to date version of g++ installed, and
your build tree is activated by running the ``activate.sh`` file.
then cd into the ``examples/custom_op/`` folder and run:

::

  g++  -fPIC custom_op.cpp -shared -lpopart -o custom_op.so

Then, run:

::

  python custom_op.py

See `here <https://linux.die.net/man/1/g++>`_ for an explanation of the g++ flags used.

Ensure you still have your build environment activated, as well as any Python virtual environments, then run ``custom_op.py``
which invokes the ``CubeOp`` via python.

To create your own custom op, For an op in PopART, you need to implement 4 classes.

 - CubeOp;

 - CubeGradOp;

 - CubeOpx;

 - CubeGradOpx;

The ``Op`` is a Poplar and hardware agnostic description of the computation. The ``OpX`` is the Poplar implementation of the ``Op``.
Gradients are used in the backwards pass. So, for inference only, you can disregard the gradient ``Op`` & ``OpX``. For an op to be
"visible" for PopART to use, you must register it and provide an OpSet version and domain:

::

  namespace Onnx {
  namespace CustomOperators {
  const popart::OperatorIdentifier Cube = {"ai.acme", "Cube", 1};
  } // namespace CustomOperators
  namespace CustomGradOperators {
  const popart::OperatorIdentifier CubeGrad = {"ai.acme", "CubeGrad", 1};
  } // namespace CustomGradOperators
  } // namespace Onnx

ONNX is defined such that the IR can evolve independently from the set of operators. So we version operators to ensure that any new,
breaking changes would not affect previous model implementations of that operator. Domains are just groupings of operators to allow
different implementations across organisations. At Graphcore we have ai.graphcore, for the included example we have used com.acme.