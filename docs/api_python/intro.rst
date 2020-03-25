Introduction
------------

The Poplar Advanced Run Time (PopART) is part of the Poplar SDK for implementing
and running algorithms on networks of Graphcore IPU processors.

This document describes the PopART Python API. Many classes are wrappers around
the equivalent C++ class, for example ``popart.builder.Builder`` wraps the C++
``Builder`` class (renamed ``BuilderCore`` in Python).
There are more detailed descriptions of some functions in the
`PopART C++ API <https://documents.graphcore.ai/documents/UG11/latest>`_.

For more information about PopART, please refer to the
`PopART User Guide <https://documents.graphcore.ai/documents/UG8/latest>`_.
