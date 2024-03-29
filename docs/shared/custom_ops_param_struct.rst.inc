.. Copyright (c) 2022 Graphcore Ltd. All rights reserved.

.. _sec_custom_op_param_struct:

Parameter struct
----------------

The first step is to define a C++ struct that encapsulate the parameters that
the custom operation needs. In our case, the LeakyReLU operation has one
parameter, ``alpha``, resulting in a struct defined as follows:

.. literalinclude:: files/leaky_relu_op_impl.cpp
    :language: cpp
    :name: leaky_relu_op_params
    :caption: Struct that encapsulates Leaky ReLU parameters
    :start-after: ParamStruct begin
    :end-before: ParamStruct end

.. only:: html

    :download:`Download <files/leaky_relu_op_impl.cpp>`


Note that this struct must implement two methods: ``appendAttributes`` and
``makeFromAttributes``.

The method ``appendAttributes`` appends the parameters from an instance of
PopART's :external+popart-user-guide:cpp:class:`popart::OpSerialiserBase` class. This is so
that two operations with different parameter values can be distinguished from
each other in the IR.

The static method, ``makeFromAttributes``, creates an instance of the parameter
struct from PopART's :external+popart-user-guide:cpp:class:`popart::Attributes` class.
