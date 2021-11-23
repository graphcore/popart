``popart.ir`` User Guide (experimental)
---------------------------------------
.. warning::
     The ``popart.ir`` Python module is currently experimental and may be subject to change
     in future releases in ways that are backwards incompatible without
     deprecation warnings.

.. warning::
     Due to the experimental nature of ``popart.ir`` the documentation provided in
     this section is incomplete.

..
  NOTE: Comments in .rst are '..' followed by a new line and an indentation.
  As you write content for a section heading that is commented out, please
  un-comment the heading also.

As an alternative to using the ONNX builder to create models, ``popart.ir`` is
an experimental PopART Python module which you can use to create
(and, to a limited degree, manipulate) PopART models directly.

PopART models are represented using an intermediate representation (IR).
The ``popart.ir`` package allows you to manipulate these IRs.

.. include:: popartir_concepts.rst
.. include:: builderpopartir.rst
