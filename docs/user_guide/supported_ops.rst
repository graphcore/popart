.. _popart_supported_ops:

Supported operators
===================

PopART is compatible with ONNX versions up to and including 1.6.
(see `ONNX Versioning <https://github.com/onnx/onnx/blob/master/docs/Versioning.md>`_).
This section lists the supported operators.

The Graphcore (:ref:`ai.graphcore <sec_domain_ai.graphcore>`) and ONNX (:ref:`ai.onnx <sec_domain_ai.onnx>`) operators, and versions supported,
are listed below.
See `ONNX Operators <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_
for more information.

.. note:: The limitations of these operators are listed in :numref:`sec_supported_ops_limitations`.

.. include:: ../popart_supported_ops_gen.rst

.. _sec_supported_ops_limitations:

Limitations
-----------
.. warning::
     The information provided in this section is incomplete.

Limitations of ai.onnx operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Clip-11: Does not support variable min/max input parameters. The parameters
  must contain a value at model initialisation and any run-time changes to
  these parameters will not be read by the model.

Limitations of ai.graphcore operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There are no known limitations.
