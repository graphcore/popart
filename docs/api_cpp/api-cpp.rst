PopART C++ API
==============

Sessions
--------

.. code-block:: cpp

  #include <popart/session.hpp>

.. doxygenclass:: popart::Session
  :members:


Training session
................

.. code-block:: cpp

  #include <popart/session.hpp>

.. doxygenclass:: popart::TrainingSession
  :members:


Inference session
..................

.. code-block:: cpp

  #include <popart/session.hpp>

.. doxygenclass:: popart::InferenceSession
  :members:


Data input and output (IStepIO)
...............................

.. code-block:: cpp

  #include <popart/istepio.hpp>

.. doxygenclass:: popart::IStepIO
  :members:


.. code-block:: cpp

  #include <popart/stepio_generic.hpp>

.. doxygenclass:: popart::StepIOGeneric
  :members:

..doxygentypedef:: popart::StepIOCallback::InputCallback
..doxygentypedef:: popart::StepIOCallback::InputCallbackComplete
..doxygentypedef:: popart::StepIOCallback::OutputCallback
..doxygentypedef:: popart::StepIOCallback::OutputCallbackComplete

.. doxygenclass:: popart::StepIOCallback
  :members:


Session options
...............

.. code-block:: cpp

  #include <popart/sessionoptions.hpp>

.. doxygenstruct:: popart::SessionOptions
  :members:

.. doxygenenum:: popart::AccumulateOuterFragmentSchedule

.. doxygenstruct:: popart::AccumulateOuterFragmentSettings
  :members:

.. doxygenstruct:: popart::AutomaticLossScalingSettings
  :members:

.. doxygenenum:: popart::BatchSerializationBatchSchedule

.. doxygenenum:: popart::BatchSerializationMethod

.. doxygenstruct:: popart::BatchSerializationSettings
  :members:

.. doxygenenum:: popart::BatchSerializationTransformContext

.. doxygenenum:: popart::DotCheck

.. doxygenenum:: popart::ExecutionPhaseIOSchedule

.. doxygenstruct:: popart::ExecutionPhaseSettings
  :members:

.. doxygenenum:: popart::ExecutionPhaseSchedule

.. doxygenenum:: popart::Instrumentation

.. doxygenenum:: popart::IrSerializationFormat

.. doxygenenum:: popart::MergeVarUpdateType

.. doxygenenum:: popart::RecomputationType

.. doxygenenum:: popart::SubgraphCopyingStrategy

.. doxygenenum:: popart::SyntheticDataMode

.. doxygenstruct:: popart::TensorLocationSettings
  :members:

.. doxygenenum:: popart::VirtualGraphMode

Optimizers
----------

.. code-block:: cpp

  #include <popart/optimizer.hpp>

.. doxygenclass:: popart::Optimizer
  :members:

.. doxygenenum:: popart::OptimizerType

.. doxygenenum:: popart::OptimizerReductionType

.. doxygenenum:: popart::WeightDecayMode

.. code-block:: cpp

  #include <popart/optimizervalue.hpp>

.. doxygenclass:: popart::OptimizerValue
  :members:


Stochastic Gradient Descent (SGD)
.................................

.. code-block:: cpp

  #include <popart/clipnormsettings.hpp>

.. doxygenclass:: popart::ClipNormSettings
  :members:

.. code-block:: cpp

  #include <popart/sgd.hpp>

.. doxygenclass:: popart::SGD
  :members:

.. doxygenclass:: popart::ConstSGD
  :members:


Adam, AdaMax & Lamb
...................

.. code-block:: cpp

  #include <popart/adam.hpp>

.. doxygenenum:: popart::AdamMode

.. doxygenclass:: popart::Adam
  :members:


AdaDelta, RMSProp & AdaGrad
...........................

.. code-block:: cpp

  #include <popart/adaptive.hpp>

.. doxygenenum:: popart::AdaptiveMode

.. doxygenclass:: popart::Adaptive
  :members:


Builder
-------

.. code-block:: cpp

  #include <popart/builder.hpp>

.. doxygenclass:: popart::Builder
   :members:

.. doxygenclass:: popart::AiGraphcoreOpset1
   :members:


Data flow
---------

.. code-block:: cpp

  #include <popart/dataflow.hpp>

.. doxygenenum:: popart::AnchorReturnTypeId

.. doxygenclass:: popart::AnchorReturnType
   :members:

.. doxygenclass:: popart::DataFlow
   :members:


Device manager
--------------

.. code-block:: cpp

  #include <popart/devicemanager.hpp>

.. doxygenenum:: popart::DeviceType

.. doxygenenum:: popart::DeviceConnectionType

.. doxygenenum:: popart::SyncPattern

.. doxygenclass:: popart::DeviceInfo
   :members:

.. doxygenclass:: popart::DeviceManager
   :members:

.. doxygenclass:: popart::DeviceProvider
   :members:

.. doxygenclass:: popart::popx::Devicex
   :members:



Op creation
-----------

Op definition for PopART IR
...........................

.. code-block:: cpp

  #include <popart/op.hpp>

.. doxygenclass:: popart::Op
   :members:

.. doxygenstruct:: popart::POpCmp
   :members:

.. code-block:: cpp

  #include <popart/opmanager.hpp>

.. doxygenclass:: popart::OpDefinition
   :members:

.. doxygenclass:: popart::OpCreatorInfo
   :members:

.. doxygenclass:: popart::OpManager
   :members:


.. code-block:: cpp

  #include <popart/op/varupdate.hpp>

.. doxygenclass:: popart::VarUpdateOp
   :members:


Op definition for Poplar implementation
.......................................

.. code-block:: cpp

  #include <popart/popx/opx.hpp>

.. doxygenclass:: popart::popx::Opx
   :members:


Utility classes
---------------

Tensor information
..................

.. code-block:: cpp

  #include <popart/tensorinfo.hpp>

.. doxygenenum:: popart::DataType

.. doxygenclass:: popart::DataTypeInfo
  :members:

.. doxygenclass:: popart::TensorInfo
  :members:


Tensor location
...............

.. code-block:: cpp

  #include <popart/tensorlocation.hpp>

.. doxygenenum:: popart::ReplicatedTensorSharding

.. doxygenclass:: popart::TensorLocation
  :members:

.. doxygenenum:: popart::TensorStorage

.. doxygenenum:: popart::TileSet


Region
......

.. code-block:: cpp

  #include <popart/region.hpp>

.. doxygenclass:: popart::view::Region
  :members:


Error handling
..............

.. code-block:: cpp

  #include <popart/error.hpp>

.. doxygenenum:: popart::ErrorSource

.. doxygenclass:: popart::error
   :members:

.. doxygenclass:: popart::memory_allocation_err
   :members:


Debug context
.............

.. code-block:: cpp

  #include <popart/debugcontext.hpp>

.. doxygenclass:: popart::DebugContext
   :members:


Attributes
..........

.. code-block:: cpp

  #include <popart/attributes.hpp>

.. doxygenclass:: popart::Attributes
   :members:


Void data
.........

.. code-block:: cpp

  #include <popart/voiddata.hpp>

.. doxygenclass:: popart::ConstVoidData
   :members:

.. doxygenclass:: popart::MutableVoidData
   :members:


Input shape information
.......................

.. code-block:: cpp

  #include <popart/inputshapeinfo.hpp>

.. doxygenclass:: popart::InputShapeInfo
   :members:


Patterns
........

.. code-block:: cpp

  #include <popart/patterns.hpp>

.. doxygenclass:: popart::Patterns
   :members:


Type definitions
................

.. doxygenfile:: names.hpp
  :sections: innernamespace typedef
