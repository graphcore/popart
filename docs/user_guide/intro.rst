Introduction
------------

The Poplar Advanced Run Time (PopART) is part of the Poplar SDK for implementing and running algorithms on
networks of Graphcore IPU processors. It enables you to import models using the
Open Neural Network Exchange (ONNX) and run them using the Poplar tools. ONNX is a serialisation format
for neural network systems that can be created and read by several frameworks including Caffe2, PyTorch and MXNet.

This document describes the features of PopART. It assumes that you are familiar with machine learning and the
ONNX framework.

An overview of the IPU architecture and programming model can be found in the
`IPU Programmer's Guide <https://www.graphcore.ai/docs/ipu-programmers-guide>`_.
For more information on the Poplar framework refer to the
`Poplar and PopLibs User Guide
<https://www.graphcore.ai/docs/poplar-and-poplibs-user-guide>`_.

PopART has three main features:

1) It can import ONNX graphs into a runtime environment (see :any:`popart_importing`).

2) It provides a simple interface for constructing ONNX graphs without needing
   a third party framework (described in :any:`popart_building`).

3) It runs imported graphs in inference, evaluation or training modes, by
   building a Poplar engine, connecting data feeds and scheduling the execution
   of the Engine (see :any:`popart_executing`).

IPU-specific annotations on ONNX operations allow the provider of the graph to
control IPU-specific features, such as mapping an algorithm across multiple
IPUs.

APIs are available for C++ and Python. Most of the examples in this document use the Python API.
