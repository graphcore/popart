.. _popart_supported_ops:

Supported operators
===================

PopART is compatible with ONNX versions up to and including 1.6.
(see `ONNX Versioning <https://github.com/onnx/onnx/blob/master/docs/Versioning.md>`_).
This section lists the supported operators. :numref:`sec_supported_ops_unsupported_opset` describes how to convert an ONNX model that uses an opset that is not supported by PopART.

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


.. _sec_supported_ops_unsupported_opset:

Converting ONNX models with opset versions not supported by PopART
------------------------------------------------------------------

If you have an ONNX model that uses an opset version that is not supported by PopART then you can convert the model using the `ONNX Version Converter <https://github.com/onnx/onnx/blob/main/docs/VersionConverter.md>`__. There are both `Python <https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#converting-version-of-an-onnx-model-within-default-domain-aionnx>`__ and `C++ <https://github.com/onnx/onnx/blob/main/docs/VersionConverter.md#invoking-the-version-converter>`__ APIs.

You will use the converter when your ONNX model uses an opset different to those listed in :numref:`sec_domain_ai.onnx` and the target opset will be a version supported by PopART.

.. note:: Currently, the highest opset PopART supports is 11. If your model uses opsets higher than 11, then you *will* have to use the ONNX Version Converter.