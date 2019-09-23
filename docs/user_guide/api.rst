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

Losses
......

.. doxygenclass:: popart::L1Loss
  :members:

.. doxygenclass:: popart::NllLoss
  :members:

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

.. doxygenclass:: popart::DataFlow
   :members:

.. doxygenclass:: popart::DeviceInfo
   :members:

.. doxygenclass:: popart::DeviceManager
   :members:

.. doxygenclass:: popart::DeviceProvider
   :members:

.. doxygenfunction:: popart::operator<<(std::ostream&, VirtualGraphMode)
.. doxygenfunction:: popart::operator<<(std::ostream&, const ConvPartialsType&)
.. doxygenfunction:: popart::operator<<(std::ostream&, VirtualGraphMode)
.. doxygenfunction:: popart::operator<<(std::ostream&, const DeviceInfo&)
.. doxygenfunction:: popart::operator<<(std::ostream&, const DeviceType&)
.. doxygenfunction:: popart::operator<<(std::ostream&, const GradInOutMapper&)
.. doxygenfunction:: popart::operator<<(std::ostream&, const GradOpInType&)
.. doxygenfunction:: popart::operator<<(std::ostream&, const GraphId&)
.. doxygenfunction:: popart::operator<<(std::ostream&, const Half&)
.. doxygenfunction:: popart::operator<<(std::ostream&, const OperatorIdentifier&)
.. doxygenfunction:: popart::operator<<(std::ostream&, const Patterns&)
.. doxygenfunction:: popart::operator<<(std::ostream&, const Scope&)
.. doxygenfunction:: popart::operator<<(std::ostream&, const TensorInfo&)
.. doxygenfunction:: popart::operator<<(std::ostream&, const TensorType&)
.. doxygenfunction:: popart::operator<<(std::ostream&, const TopoCons&)
.. doxygenfunction:: popart::operator<<(std::ostream&, const std::vector<std::size_t>&)
.. doxygenfunction:: popart::operator<<(std::ostream&, const NDArrayWrapper<T>&)
Error handling
--------------

.. doxygenenum:: popart::ErrorSource
   :members:

.. doxygenclass:: popart::error
   :members:

.. doxygenclass:: popart::memory_allocation_err
   :members:
