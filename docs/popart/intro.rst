Introduction
------------

The Poplar Advanced Run Time (PopART) is part of the Poplar SDK for implementing
and running algorithms on networks of Graphcore IPUs. It enables you to import
models using the Open Neural Network Exchange (ONNX) and run them using the
Poplar tools. ONNX is a serialisation format for neural network systems that can
be created and read by several frameworks including Caffe2, PyTorch and MXNet.

This document describes the features of PopART. It assumes that you are familiar
with machine learning and the ONNX framework.

An overview of the IPU architecture and programming model can be found in the
:doc:`ipu-programmers-guide:index`. For more information on the Poplar graph
programming framework, refer to the :doc:`poplar-user-guide:index`.

PopART has three main features:

1) It can import ONNX graphs into a runtime environment
   (:numref:`popart_importing`).

2) It provides a simple interface for constructing ONNX graphs without need for
   a third party framework (described in :numref:`popart_building`).

3) It runs imported graphs in inference, evaluation or training modes, by
   building a Poplar Engine, connecting data feeds and scheduling the execution
   of the Engine (:numref:`popart_executing`).

IPU-specific annotations on ONNX operations allow the provider of the graph to
control IPU-specific features, such as mapping an algorithm across multiple
IPUs.

PopART has both a :ref:`C++ API <popart_cpp_api_reference>` and a :ref:`Python API <popart_python_api_reference>`. Most of the examples in this document use the Python API.
