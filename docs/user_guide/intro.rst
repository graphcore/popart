Introduction
------------

The Poplar Advanced Run Time (PopART) is part of the Poplar SDK for designing and running algorithms on
networks of Graphcore IPU processors. It enables you to import models from frameworks such as the 
Open Neural Network Exchange (ONNX) and run them using the Poplar tools.

This document describes the feature of PopART. It assumes you are familiar with machine learning and the 
ONNX framework.

.. TODO: add link to docs
An overview of the IPU architecture and programming model can be found in the *IPU Programmer's Manual*. 
For more information on the Poplar framework refer to the *Poplar and Poplibs User Guide*.

PopART has three main features:

1) It can import ONNX graphs into a runtime environment.
2) It runs imported graphs in inference, evaluation or training modes, by
   building a Poplar engine, connecting data feeds and scheduling the execution
   of the Engine.
3) It provides a simple interface for constructing ONNX graphs without needing
   a third party framework.

IPU-specific annotations on ONNX operations allow the provider of the graph to
control IPU-specific features, such as mapping an algorithm across multiple
IPUs.

APIs are available for C++ and Python.
