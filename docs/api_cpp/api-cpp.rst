PopART C++ API
==============

.. TODO: Complete API documentation. Currently only includes objects which have (some) Doxygen comments

Sessions
--------

.. doxygenclass:: popart::InferenceSession
  :members:

.. doxygenclass:: popart::TrainingSession
  :members:

Session option flags
....................

.. doxygenstruct:: popart::SessionOptions
  :members:

Training operations
-------------------

Optimisers
..........

.. doxygenclass:: popart::ConstSGD
  :members:

.. doxygenclass:: popart::SGD
  :members:

Builder
-------

.. doxygenclass:: popart::Builder
   :members:

.. doxygenclass:: popart::AiGraphcoreOpset1
   :members:


.. doxygenclass:: popart::BuilderImpl
   :members:

Data flow
---------

.. doxygenenum:: popart::AnchorReturnTypeId

.. doxygenclass:: popart::AnchorReturnType
   :members:

.. doxygenclass:: popart::DataFlow
   :members:

Device manager
--------------

.. doxygenenum:: popart::DeviceType

.. doxygenclass:: popart::DeviceInfo
   :members:

.. doxygenclass:: popart::DeviceManager
   :members:

.. doxygenclass:: popart::DeviceProvider
   :members:

Error handling
--------------

.. doxygenenum:: popart::ErrorSource

.. doxygenclass:: popart::error
   :members:

.. doxygenclass:: popart::memory_allocation_err
   :members:
