Introduction
============

.. warning::
     The ``popxl`` Python package is currently experimental and may be subject to change
     in future releases in ways that are backwards incompatible without
     deprecation warnings.

.. warning::
     Due to the experimental nature of `PopXL` the user and API reference documentation provided is incomplete.

PopART models are represented using an intermediate representation (IR). As an
alternative to using the ONNX builder to create PopART models, ``popxl`` is an
experimental PopART Python package which you can use to directly create (and, to
a limited degree, manipulate) PopART IRs. This provides greater flexibility than
is possible using the standard PopART API.
