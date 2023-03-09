.. _sec_data_types:

Data types
==========

Currently, PopXL supports the data types listed in :numref:`datatypes_table`.
These data types are defined in ``popxl`` directly and
will be converted to their IPU-compatible data type. Note that if the session option :py:attr:`popart.SessionOptions.enableSupportedDataTypeCasting` is set to ``True``, then ``int64``
and ``uint64`` will be downcast to ``int32`` and ``uint32``, respectively.

.. list-table:: Data types in PopXL
   :header-rows: 1
   :name: datatypes_table

   * - PopXL dtype
     - int
     - floating point
     - signed
     - NumPy dtype
     - Python dtype
     - alias
   * - ``bool``
     - False
     - False
     - False
     - bool
     - builtins.bool
     - N/A
   * - ``int8``
     - True
     - False
     - True
     - int8
     - None
     - N/A
   * - ``int32``
     - True
     - False
     - True
     - int32
     - None
     - N/A
   * - ``uint8``
     - True
     - False
     - False
     - uint8
     - None
     - N/A
   * - ``uint32``
     - True
     - False
     - False
     - uint32
     - None
     - N/A
   * - ``float16``
     - False
     - True
     - True
     - float16
     - None
     - ``half``
   * - ``float32``
     - False
     - True
     - True
     - float32
     - builtins.float
     - ``float``
   * - ``float64``
     - False
     - True
     - True
     - float64
     - None
     - ``double``
   * - ``float8_143``
     - False
     - True
     - True
     - uint8
     - None
     - N/A
   * - ``float8_152``
     - False
     - True
     - True
     - uint8
     - None
     - N/A

8-bit floating point datatypes
------------------------------

.. _sec_float8_datatypes:

There are two 8-bit float datatypes in PopXL, namely `popxl.float8_143` and
`popxl.float8_152`. The numbers in the names of these types refer to the format:
the number of bits used to represent the sign, exponent and mantissa. As with
other floating point representations, the exponent is subject to a bias. This
bias is different for each of the two formats:

.. list-table:: Float8 formats in PopXL
   :header-rows: 1
   :name: datatypes_float8_table

   * - PopXL dtype
     - Number of sign bits
     - Number of exponent bits
     - Number of mantissa bits
     - Exponent bias
     - Smallest positive value
     - Largest positive value
   * - ``float8_143``
     - 1
     - 4
     - 3
     - -8
     - :math:`2^-10`
     - :math:`240.0`
   * - ``float8_152``
     - 1
     - 5
     - 2
     - -16
     - :math:`2^-17`
     - :math:`57344.0`

More details about the numerical properties of these two 8-bit floating point
data types can be found in arXiv paper `8-Bit Numerical Formats for Deep Neural
Networks <https://arxiv.org/pdf/2206.02915.pdf>`_.

Because of the limited numerical range of 8-bit floating point numbers,
operations that consume or produce tensors of these types are usually fused with
a `pow2` scaling operation. These operations have a `log2_scale` parameter.
Internally, these operations multiply your 8-bit floating point data with a
factor of `pow2(log2_scale)`. Note that you can use a positive `log2_scale` to
accommodate large numerical ranges or you can use negative values for smaller
numerical ranges. Currently, we support `log2_scale` parameter values in the
range :math:`[-32,32)`.

:numref:`popxl_ops_available_float8_api` lists a number of utility
functions and operations for 8-bit floats. 

.. list-table:: 8-Bit floating point API
   :header-rows: 1
   :width: 100%
   :widths: 45, 55
   :name: popxl_ops_available_float8_api
   :class: longtable

   * - API function
     - Description

   * - :py:func:`~popxl.utils.host_pow2scale_cast_to_fp8`
     - Host-based conversion from 16/32/64-bit floating point data to a 8-bit floating point representation.

   * - :py:func:`~popxl.utils.host_pow2scale_cast_from_fp8`
     - Host-based conversion from a 8-bit floating point representation back to 16/32/64-bit floating point data.

   * - :py:func:`~popxl.utils.pow2scale_cast_to_fp8`
     - Operation to convert from 16-bit floating point to 8-bit floating point.

   * - :py:func:`~popxl.utils.pow2scale_cast_from_fp8`
     - Operation to convert from 8-bit floating point to 16-bit floating point.

   * - :py:func:`~popxl.utils.matmul_pow2scaled`
     - Operation to perform a matmul on 8-bit floating point data resulting in 16-bit floating point output.

Note that for device-based operations that support 8-bit float operands the
`log2_scale` operand is also a tensor parameter in its own right. This means you
can change this scaling at runtime if you so desire.

8-bit floating point inference model example
--------------------------------------------

An example of using float8 tensors in an inference graph is shown in the example 
:download:`float8_inference.py <files/float8_inference.py>`. 
The float16 input data is loaded onto the device as-is, then cast to float8 on the
device with a :py:func:`~popxl.ops.pow2scale_cast_to_fp8` operator. 
After this we do the cast on the host of the trained weight data (in this example 
the weights are randomly generated), then creating the :py:func:`popxl.variable` for the float8 weights.

Note that in both cases we do not scale the values, as this is done within the :py:func:`~popxl.ops.conv_pow2scaled` operator.

  .. literalinclude:: files/float8_inference.py
    :language: python
    :start-after: Cast begin
    :end-before: Cast end
    :name: cast-float8-example
    :caption: Example of host-based casting to float8
    :linenos:
    :lineno-match:

  .. only:: html

      :download:`Download float8_inference.py <files/float8_inference.py>`

In the PopXL :py:class:`~popxl.Module` you can see the :py:func:`~popxl.ops.conv_pow2scaled` operator which takes a 
`log2_scale` tensor, in addition to our float8 input and weight tensors, as well as all 
of the usual parameters used in a :py:func:`~popxl.ops.conv` operator.

  .. literalinclude:: files/float8_inference.py
    :language: python
    :start-after: ConvFloat8 begin
    :end-before: ConvFloat8 end
    :name: float8-module-example
    :caption: Example of using float8 tensors
    :linenos:
    :lineno-match:

  .. only:: html

      :download:`Download float8_inference.py <files/float8_inference.py>`


See :py:func:`~popxl.ops.conv_pow2scaled` for more details on this operator.