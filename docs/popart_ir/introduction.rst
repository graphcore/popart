Introduction
============

.. warning::
     The ``popart.ir`` Python module is currently experimental and may be subject to change
     in future releases in ways that are backwards incompatible without
     deprecation warnings.

.. warning::
     Due to the experimental nature of ``popart.ir`` the documentation provided in
     this section is incomplete.

As an alternative to using the ONNX builder to create models, ``popart.ir`` is
an experimental PopART Python module which you can use to create
(and, to a limited degree, manipulate) PopART models directly.
This provides greater flexibility than is possible using the standard PopART API.

PopART models are represented using an intermediate representation (IR).
The ``popart.ir`` package allows you to directly manipulate these IRs.
