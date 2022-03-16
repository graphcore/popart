.. _sec_data_types:

Data types
==========

Currently, PopXL supports the data types listed in :numref:`datatypes_table`.
These data types are defined in ``popxl`` directly and
will be converted to their IPU-compatible data type. Note that if the session option :py:func:`popart_core.SessionOptions.enableSupportedDataTypeCasting` is set to ``True``, then ``int64``
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
