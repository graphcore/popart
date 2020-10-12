PopART C++ API
==============

.. TODO: Complete API documentation. Currently only includes objects which have (some) Doxygen comments

Sessions
--------

.. doxygenclass:: popart::InferenceSession
  :members:

.. doxygenclass:: popart::TrainingSession
  :members:

Session options
..............

.. doxygenenum:: popart::DotCheck

.. doxygenenum:: popart::RecomputationType

.. doxygenenum:: popart::MergeVarUpdateType

.. doxygenenum:: popart::VirtualGraphMode

.. doxygenenum:: popart::IrSerializationFormat

.. doxygenenum:: popart::SyntheticDataMode

.. doxygenenum:: popart::Instrumentation

.. doxygenenum:: popart::BatchSerializationBatchSchedule

.. doxygenenum:: popart::ExecutionPhaseIOSchedule

.. doxygenenum:: popart::ExecutionPhaseSchedule

.. doxygenenum:: popart::AccumulateOuterFragmentSchedule

.. doxygenstruct:: popart::TensorLocationSettings
  :members:

.. doxygenstruct:: popart::BatchSerializationSettings
  :members:

.. doxygenstruct:: popart::ExecutionPhaseSettings
  :members:

.. doxygenstruct:: popart::AccumulateOuterFragmentSettings
  :members:

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

.. doxygenenum:: popart::DeviceConnectionType

.. doxygenenum:: popart::SyncPattern

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
