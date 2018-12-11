Introduction
------------

Poponnx is part of the Poplar set of tools for designing and running algorithms
on networks of Graphcore IPU processors.

It has three main features:

1) It can import ONNX graphs into a runtime environment.
2) It runs imported graphs in inference, evaluation or training modes, by
   building a Poplar engine, connecting data feeds and schduling the execution
   of the Engine.
3) It provdes a simple interface for constructing ONNX graphs without needing
   a third party framework.

APIs are available for C++ and python.


