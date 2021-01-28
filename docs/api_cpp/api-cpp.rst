PopART C++ API
==============

.. TODO: Complete API documentation. Currently only includes objects which have (some) Doxygen comments

Sessions
--------

.. doxygenclass:: popart::IStepIO
  :members:

.. doxygenclass:: popart::InferenceSession
  :members:

.. doxygenclass:: popart::TrainingSession
  :members:

Session options
..............

.. doxygenenum:: popart::AccumulateOuterFragmentSchedule

.. doxygenenum:: popart::BatchSerializationBatchSchedule

.. doxygenenum:: popart::DotCheck

.. doxygenenum:: popart::ExecutionPhaseIOSchedule

.. doxygenenum:: popart::ExecutionPhaseSchedule

.. doxygenenum:: popart::Instrumentation

.. doxygenenum:: popart::IrSerializationFormat

.. doxygenenum:: popart::MergeVarUpdateType

.. doxygenenum:: popart::SyntheticDataMode

.. doxygenenum:: popart::RecomputationType

.. doxygenenum:: popart::ReplicatedTensorSharding

.. doxygenenum:: popart::TensorStorage

.. doxygenenum:: popart::TileSet

.. doxygenenum:: popart::VirtualGraphMode

.. doxygenenum:: popart::SubgraphCopyingStrategy

.. doxygenstruct:: popart::AccumulateOuterFragmentSettings
  :members:

.. doxygenstruct:: popart::BatchSerializationSettings
  :members:

.. doxygenstruct:: popart::ExecutionPhaseSettings
  :members:

.. doxygenstruct:: popart::TensorLocationSettings
  :members:

.. doxygenclass:: popart::TensorLocation
  :members:

.. doxygenstruct:: popart::SessionOptions
  :members:

Optimizers
----------

.. doxygenclass:: popart::OptimizerValue
  :members:

.. doxygenenum:: popart::WeightDecayMode

Stochastic Gradient Descent (SGD)
.................................

.. doxygenstruct:: popart::ClipNormSettings
  :members:

.. doxygenclass:: popart::SGD
  :members:

.. doxygenclass:: popart::ConstSGD
  :members:

Adam, AdaMax & Lamb
...................

.. doxygenenum:: popart::AdamMode

.. doxygenclass:: popart::Adam
  :members:

AdaDelta, RMSProp & AdaGrad
.........................

.. doxygenenum:: popart::AdaptiveMode

.. doxygenclass:: popart::Adaptive
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
