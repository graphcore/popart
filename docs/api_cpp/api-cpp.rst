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


Session options
...............

.. code-block:: cpp

  #include <popart/sessionoptions.hpp>

.. doxygenstruct:: popart::SessionOptions
  :members:

.. doxygenenum:: popart::AccumulateOuterFragmentSchedule

.. doxygenstruct:: popart::AccumulateOuterFragmentSettings
  :members:

.. doxygenenum:: popart::AutodiffStitchStrategy

.. doxygenstruct:: popart::AutodiffSettings
  :members:

.. doxygenstruct:: popart::AutomaticLossScalingSettings
  :members:

.. doxygenenum:: popart::GradientTensorTrackingMethod

.. doxygenenum:: popart::BatchSerializationBatchSchedule

.. doxygenenum:: popart::BatchSerializationMethod

.. doxygenstruct:: popart::BatchSerializationSettings
  :members:

.. doxygenenum:: popart::BatchSerializationTransformContext

.. doxygenenum:: popart::ExecutionPhaseIOSchedule

.. doxygenstruct:: popart::ExecutionPhaseSettings
  :members:

.. doxygenenum:: popart::ExecutionPhaseSchedule

.. doxygenenum:: popart::Instrumentation

.. doxygenenum:: popart::IrSerializationFormat

.. doxygenenum:: popart::MeanReductionStrategy

.. doxygenenum:: popart::MergeVarUpdateType

.. doxygenenum:: popart::RecomputationType

.. doxygenenum:: popart::ReductionType

.. doxygenenum:: popart::SubgraphCopyingStrategy

.. doxygenenum:: popart::SyntheticDataMode

.. doxygenstruct:: popart::TensorLocationSettings
  :members:

.. doxygenenum:: popart::VirtualGraphMode

.. doxygenstruct:: popart::DeveloperSettings

.. doxygenclass:: popart::VariableSettings

.. doxygenclass:: popart::CommGroup

Data input and output (IStepIO)
-------------------------------

.. code-block:: cpp

  #include <popart/istepio.hpp>

.. doxygenclass:: popart::IStepIO
  :members:


.. code-block:: cpp

  #include <popart/stepio.hpp>

.. doxygenclass:: popart::StepIO

.. doxygenclass:: popart::StepIOCallback
  :members:

.. doxygenclass:: popart::IWeightsIO

.. doxygenclass:: popart::WeightsIO

.. doxygenstruct:: popart::IArrayAccessor

.. code-block:: cpp

  #include <popart/stepio_generic.hpp>

.. doxygenclass:: popart::StepIOGeneric
  :members:

.. doxygenstruct:: popart::ArrayInfo

.. code-block:: cpp

  #include <popart/iarray.hpp>

.. doxygenclass:: popart::IArray
  :members:

Tensors
-------

.. code-block:: cpp

  #include <popart/tensor.hpp>

.. doxygenclass:: popart::Tensor

.. doxygenenum:: popart::TensorType

.. doxygenenum:: popart::VariableUpdateType

.. code-block:: cpp

  #include <popart/tensorinfo.hpp>

.. doxygenenum:: popart::DataType

.. doxygenclass:: popart::DataTypeInfo
  :members:

.. doxygenclass:: popart::TensorInfo
  :members:

.. code-block:: cpp

  #include <popart/tensorindex.hpp>

.. doxygenclass:: popart::TensorIndexMap
  :members:

.. code-block:: cpp

  #include <popart/tensorlocation.hpp>

.. doxygenenum:: popart::ReplicatedTensorSharding

.. doxygenclass:: popart::TensorLocation
  :members:

.. doxygenenum:: popart::TensorStorage

.. doxygenenum:: popart::TileSet


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

.. code-block:: cpp

  #include <popart/optimizervaluemap.hpp>

.. doxygenclass:: popart::OptimizerValueMap

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

.. doxygenenum:: popart::SGDAccumulatorAndMomentum


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

.. doxygenclass:: popart::Ir
   :members:

.. doxygentypedef:: popart::HashesMap

.. doxygenenum:: popart::RequireOptimalSchedule

.. doxygenclass:: popart::Graph
   :members:

.. doxygenclass:: popart::AiOnnxMlOpset1
   :members:

.. doxygenclass:: popart::AiGraphcoreOpset1
   :members:

.. code-block:: cpp

  #include <popart/scope.hpp>

.. doxygenclass:: popart::Scope
   :members:

Data flow
---------

.. code-block:: cpp

  #include <popart/dataflow.hpp>

.. doxygenenum:: popart::AnchorReturnTypeId
.. doxygenenum:: popart::ExchangeStrategy

.. doxygenclass:: popart::AnchorReturnType
   :members: AnchorReturnType, str, tileSet, exchangeStrategy


.. doxygenclass:: popart::DataFlow
   :members: DataFlow, setBatchesPerStep

.. doxygenclass:: popart::InputSettings
   :members:

.. doxygentypedef:: popart::AnchorReturnTypeMap

.. code-block:: cpp

  #include <popart/replicatedstreammode.hpp>

.. doxygenenum:: popart::ReplicatedStreamMode


Device manager
--------------

.. code-block:: cpp

  #include <popart/devicemanager.hpp>

.. doxygenenum:: popart::DeviceType

.. doxygenenum:: popart::DeviceConnectionType

.. doxygenenum:: popart::SyncPattern

.. doxygenclass:: popart::DeviceInfo
   :members:

.. doxygenclass:: popart::popx::DevicexInfo

.. doxygenclass:: popart::popx::DevicexCpuInfo
.. doxygenclass:: popart::popx::DevicexIpuInfo
.. doxygenclass:: popart::popx::DevicexIpuModelInfo
.. doxygenclass:: popart::popx::DevicexSimInfo

.. doxygenclass:: popart::popx::DevicexOfflineIpuInfo

.. doxygenclass:: popart::DeviceManager
   :members:

.. doxygenclass:: popart::DeviceProvider
   :members:

.. doxygenclass:: popart::popx::DevicexManager

.. code-block:: cpp

  #include <popart/popx/devicex.hpp>

.. doxygenclass:: popart::popx::Devicex
   :members:

.. doxygentypedef:: popart::popx::PopStreamId

.. doxygenclass:: popart::popx::Executablex

.. code-block:: cpp

  #include <popart/popx/irlowering.hpp>

.. doxygenclass:: popart::popx::IrLowering
   :members:

.. code-block:: cpp

  #include <popart/popx/poptensors.hpp>

.. doxygenclass:: popart::popx::PopTensors
   :members:

.. code-block:: cpp

  #include <popart/popx/popprograms.hpp>

.. doxygenclass:: popart::popx::PopPrograms
   :members:

.. code-block:: cpp

  #include <popart/popx/popopx.hpp>

.. doxygenclass:: popart::popx::PopOpx
   :members:

.. doxygentypedef:: popart::popx::ICreatorCandidatePtr

.. doxygenstruct:: popart::POpCmp

.. doxygenenum:: popart::popx::InputCreatorType


.. code-block:: cpp

  #include <popart/popx/inittensor.hpp>

.. doxygenclass:: popart::popx::ICreatorCandidate
   :members:

.. code-block:: cpp

  #include <popart/popx/replicatedtensorshardingbundle.hpp>

.. doxygenclass:: popart::popx::ReplicatedTensorShardingBundle
   :members:

.. code-block:: cpp

  #include <popart/popx/linearmapper.hpp>

.. doxygenclass:: popart::popx::LinearMapper
   :members:

Ops
---

Op definition for PopART IR
...........................

.. code-block:: cpp

  #include <popart/op.hpp>

.. doxygenclass:: popart::Op
   :members:

.. doxygenclass:: popart::GradInOutMapper
   :members:

.. code-block:: cpp

  #include <popart/operatoridentifier.hpp>

.. doxygenstruct:: popart::OperatorIdentifier

.. doxygenstruct:: popart::NumInputs

.. code-block:: cpp

  #include <popart/tensorlocation.hpp>

.. doxygentypedef:: popart::VGraphIdAndTileSet

.. code-block:: cpp

  #include <popart/basicoptionals.hpp>

.. doxygentypedef:: popart::OptionalTensorLocation

.. doxygentypedef:: popart::OptionalVGraphId

.. doxygentypedef:: popart::OptionalPipelineStage

.. doxygentypedef:: popart::OptionalExecutionPhase

.. doxygentypedef:: popart::OptionalBatchSerializedPhase

.. doxygentypedef:: popart::OptionalStochasticRoundingMethod

.. doxygentypedef:: popart::OptionalDataType

.. code-block:: cpp

  #include <popart/opmanager.hpp>

.. doxygenclass:: popart::OpDefinition
   :members:

.. doxygenclass:: popart::OpCreatorInfo
   :members:

.. doxygenclass:: popart::OpManager
   :members:

.. doxygenenum:: popart::RecomputeType

.. doxygenenum:: popart::ExecutionContext

.. doxygenenum:: popart::GradOpInType


.. code-block:: cpp

  #include <popart/op/varupdate.hpp>

.. doxygenclass:: popart::VarUpdateOp
   :members:

.. doxygenclass:: popart::AccumulatorScaleOp

.. doxygenclass:: popart::AccumulatorZeroOp

.. doxygenclass:: popart::VarUpdateWithUpdaterOp

.. doxygenclass:: popart::AccumulateBaseOp
.. doxygenclass:: popart::AccumulateOp
.. doxygenclass:: popart::RescaleAccumulateOp
.. doxygenclass:: popart::SparseAccumulateOp
.. doxygenclass:: popart::AdamComboOp
.. doxygenclass:: popart::AdamVarUpdateOp
.. doxygenclass:: popart::AdaptiveComboOp
.. doxygenclass:: popart::CopyVarUpdateOp
.. doxygenclass:: popart::SGD0ComboOp
.. doxygenclass:: popart::SGD0VarUpdateOpBase
.. doxygenclass:: popart::SGD0VarUpdateOp
.. doxygenclass:: popart::SGD1AcclUpdateOp
.. doxygenclass:: popart::SGD2PartialAcclUpdateOp
.. doxygenclass:: popart::SGD1VarUpdateOp
.. doxygenclass:: popart::SGD2VarUpdateOp
.. doxygenclass:: popart::SGDMComboBaseOp
.. doxygenclass:: popart::SGD1ComboOp
.. doxygenclass:: popart::SGD2ComboOp
.. doxygenclass:: popart::ScaledVarUpdateOp

.. code-block:: cpp

  #include <popart/alias/aliasmodel.hpp>

.. doxygenclass:: popart::AliasModel
   :members:

.. code-block:: cpp

  #include <popart/op/ipucopy.hpp>

.. doxygenclass:: popart::IpuCopyOp
   :members:

.. doxygentypedef:: popart::SourceIpuMap
.. doxygentypedef:: popart::SourceTensorMap

Op definition for Poplar implementation
.......................................

.. code-block:: cpp

  #include <popart/popx/opx.hpp>

.. doxygenclass:: popart::popx::Opx

.. doxygenclass:: popart::popx::RoiAlignGradOpx
.. doxygenclass:: popart::popx::RoiAlignOpx


Available Ops (Op class)
........................

.. doxygenstruct:: popart::AiGraphcoreOpIdV1
.. doxygenclass:: popart::AbortOp
.. doxygenclass:: popart::AbsGradOp
.. doxygenclass:: popart::AbsOp
.. doxygenclass:: popart::AdaDeltaUpdaterOp
.. doxygenclass:: popart::AdamUpdaterOp
.. doxygenclass:: popart::AddArg0GradOp
.. doxygenclass:: popart::AddArg1GradOp
.. doxygenclass:: popart::AddBiasBiasGradOp
.. doxygenclass:: popart::AddBiasDataGradOp
.. doxygenclass:: popart::AddBiasInplaceOp
.. doxygenclass:: popart::AddBiasOp
.. doxygenclass:: popart::AddLhsInplaceOp
.. doxygenclass:: popart::AddRhsInplaceOp
.. doxygenclass:: popart::AllReduceGradOp
.. doxygenclass:: popart::AllReduceOp
.. doxygenclass:: popart::AndOp
.. doxygenclass:: popart::ArgExtremaOp
.. doxygenclass:: popart::ArgMaxOp
.. doxygenclass:: popart::ArgMinOp
.. doxygenclass:: popart::AsinGradOp
.. doxygenclass:: popart::AsinInplaceOp
.. doxygenclass:: popart::AsinOp
.. doxygenclass:: popart::Atan2Arg0GradOp
.. doxygenclass:: popart::Atan2Arg1GradOp
.. doxygenclass:: popart::Atan2LhsInplaceOp
.. doxygenclass:: popart::AtanGradOp
.. doxygenclass:: popart::AtanInplaceOp
.. doxygenclass:: popart::AtanOp
.. doxygenclass:: popart::AutoLossScaleProxyGradOp
.. doxygenclass:: popart::AutoLossScaleProxyOp
.. doxygenclass:: popart::AveragePoolGradOp
.. doxygenclass:: popart::AveragePoolOp
.. doxygenclass:: popart::BaseOnnxRNNGradOp
.. doxygenclass:: popart::BaseOnnxRNNOp
.. doxygenclass:: popart::BasePadOp
.. doxygenclass:: popart::BasePadOutplaceOp
.. doxygenclass:: popart::BaseSliceOp
.. doxygenclass:: popart::BaseSortOp
.. doxygenclass:: popart::BatchNormGradOp
.. doxygenclass:: popart::BatchNormOp
.. doxygenclass:: popart::BinaryComparisonOp
.. doxygenclass:: popart::BinaryConstScalarOp
.. doxygenclass:: popart::BitwiseBinaryOp
.. doxygenclass:: popart::BitwiseNotOp
.. doxygenclass:: popart::BoundaryOp
.. doxygenclass:: popart::CallGradOp
.. doxygenclass:: popart::CallOp
.. doxygenclass:: popart::CastGradOp
.. doxygenclass:: popart::CastOp
.. doxygenclass:: popart::CeilInplaceOp
.. doxygenclass:: popart::CeilOp
.. doxygenclass:: popart::ClipGradOp
.. doxygenclass:: popart::ClipInplaceOp
.. doxygenclass:: popart::ClipOp
.. doxygenclass:: popart::CollectivesBaseOp
.. doxygenclass:: popart::ConcatGradOp
.. doxygenclass:: popart::ConcatInplaceOp
.. doxygenclass:: popart::ConcatOp
.. doxygenclass:: popart::ConvDataGradOp
.. doxygenclass:: popart::ConvFlipWeightsGradOp
.. doxygenclass:: popart::ConvFlipWeightsOp
.. doxygenclass:: popart::ConvOp
.. doxygenclass:: popart::ConvTransposeOp
.. doxygenclass:: popart::ConvWeightsGradOp
.. doxygenclass:: popart::CosGradOp
.. doxygenclass:: popart::CosOp
.. doxygenclass:: popart::CoshOp
.. doxygenclass:: popart::CtcBeamSearchDecoderOp
.. doxygenclass:: popart::CtcGradOp
.. doxygenclass:: popart::CtcOp
.. doxygenclass:: popart::CumSumGradOp
.. doxygenclass:: popart::CumSumOp
.. doxygenclass:: popart::DetachInplaceOp
.. doxygenclass:: popart::DetachOp
.. doxygenclass:: popart::DivArg0GradOp
.. doxygenclass:: popart::DivArg1GradOp
.. doxygenclass:: popart::DropoutBaseOp
.. doxygenclass:: popart::DropoutOp
.. doxygenclass:: popart::DropoutGradOp
.. doxygenclass:: popart::DynamicAddInplaceOp
.. doxygenclass:: popart::DynamicAddOp
.. doxygenclass:: popart::DynamicBaseOp
.. doxygenclass:: popart::DynamicBinaryBaseInplaceOp
.. doxygenclass:: popart::DynamicBinaryBaseOp
.. doxygenclass:: popart::DynamicSliceBaseOp
.. doxygenclass:: popart::DynamicSliceInplaceOp
.. doxygenclass:: popart::DynamicSliceOp
.. doxygenclass:: popart::DynamicSlicePadGradOp
.. doxygenclass:: popart::DynamicTernaryBaseInplaceOp
.. doxygenclass:: popart::DynamicTernaryBaseOp
.. doxygenclass:: popart::DynamicUpdateInplaceOp
.. doxygenclass:: popart::DynamicUpdateOp
.. doxygenclass:: popart::DynamicUpdateToUpdateGradOp
.. doxygenclass:: popart::DynamicUpdateUpdaterGradOp
.. doxygenclass:: popart::DynamicZeroGradOp
.. doxygenclass:: popart::DynamicZeroInplaceOp
.. doxygenclass:: popart::DynamicZeroOp
.. doxygenclass:: popart::ElementWiseBinaryArg0GradOp
.. doxygenclass:: popart::ElementWiseBinaryArg1GradOp
.. doxygenclass:: popart::ElementWiseBinaryBaseOp
.. doxygenclass:: popart::ElementWiseBinaryGradOp
.. doxygenclass:: popart::ElementWiseBinaryInplaceLhsOp
.. doxygenclass:: popart::ElementWiseBinaryInplaceRhsOp
.. doxygenclass:: popart::ElementWiseBinaryOp
.. doxygenclass:: popart::ElementWiseInplaceUnaryOp
.. doxygenclass:: popart::ElementWiseNonLinearUnaryGradOp
.. doxygenclass:: popart::ElementWiseNpBroadcastableBinaryWithGradOp
.. doxygenclass:: popart::ElementWiseUnaryBooleanOp
.. doxygenclass:: popart::ElementWiseUnaryOp
.. doxygenclass:: popart::EluGradOp
.. doxygenclass:: popart::EluInplaceOp
.. doxygenclass:: popart::EluOp
.. doxygenclass:: popart::EqualOp
.. doxygenclass:: popart::ErfGradOp
.. doxygenclass:: popart::ErfOp
.. doxygenclass:: popart::ExchangeBaseOp
.. doxygenclass:: popart::ExpGradOp
.. doxygenclass:: popart::ExpInplaceOp
.. doxygenclass:: popart::ExpOp
.. doxygenclass:: popart::ExpandGradOp
.. doxygenclass:: popart::ExpandInplaceOp
.. doxygenclass:: popart::ExpandOp
.. doxygenclass:: popart::Expm1GradOp
.. doxygenclass:: popart::Expm1InplaceOp
.. doxygenclass:: popart::Expm1Op
.. doxygenclass:: popart::FloorInplaceOp
.. doxygenclass:: popart::FloorOp
.. doxygenclass:: popart::FmodArg0GradOp
.. doxygenclass:: popart::FmodOp
.. doxygenclass:: popart::GRUGradOp
.. doxygenclass:: popart::GRUOp
.. doxygenclass:: popart::GatherGradOp
.. doxygenclass:: popart::GatherOp
.. doxygenclass:: popart::GeluGradOp
.. doxygenclass:: popart::GeluInplaceOp
.. doxygenclass:: popart::GeluOp
.. doxygenclass:: popart::GetRandomSeedOp
.. doxygenclass:: popart::GlobalAveragePoolGradOp
.. doxygenclass:: popart::GlobalAveragePoolOp
.. doxygenclass:: popart::GlobalMaxPoolGradOp
.. doxygenclass:: popart::GlobalMaxPoolOp
.. doxygenclass:: popart::GreaterOp
.. doxygenclass:: popart::GroupNormGradOp
.. doxygenclass:: popart::GroupNormOp
.. doxygenclass:: popart::HardSigmoidGradOp
.. doxygenclass:: popart::HardSigmoidInplaceOp
.. doxygenclass:: popart::HardSigmoidOp
.. doxygenclass:: popart::HasReceptiveFieldOp
.. doxygenclass:: popart::HistogramOp
.. doxygenclass:: popart::HostBaseOp
.. doxygenclass:: popart::HostLoadInplaceOp
.. doxygenclass:: popart::HostLoadOp
.. doxygenclass:: popart::HostStoreOp
.. doxygenclass:: popart::IdentityGradOp
.. doxygenclass:: popart::IdentityInplaceOp
.. doxygenclass:: popart::IdentityLossGradOp
.. doxygenclass:: popart::IdentityLossOp
.. doxygenclass:: popart::IdentityOp
.. doxygenclass:: popart::IfConditionGradOp
.. doxygenclass:: popart::IfGradOp
.. doxygenclass:: popart::IfOp
.. doxygenclass:: popart::IncrementModInplaceOp
.. doxygenclass:: popart::IncrementModOp
.. doxygenclass:: popart::InitOp
.. doxygenclass:: popart::InstanceNormGradOp
.. doxygenclass:: popart::InstanceNormOp
.. doxygenclass:: popart::IoTileCopyOp
.. doxygenclass:: popart::IsInf
.. doxygenclass:: popart::IsNaN
.. doxygenclass:: popart::L1GradOp
.. doxygenclass:: popart::L1Op
.. doxygenclass:: popart::LRNGradOp
.. doxygenclass:: popart::LRNOp
.. doxygenclass:: popart::LSTMGradOp
.. doxygenclass:: popart::LSTMOp
.. doxygenclass:: popart::LambSquareOp
.. doxygenclass:: popart::LeakyReluGradOp
.. doxygenclass:: popart::LeakyReluInplaceOp
.. doxygenclass:: popart::LeakyReluOp
.. doxygenclass:: popart::LessOp
.. doxygenclass:: popart::LinearVariadicGradOp
.. doxygenclass:: popart::Log1pGradOp
.. doxygenclass:: popart::Log1pInplaceOp
.. doxygenclass:: popart::Log1pOp
.. doxygenclass:: popart::LogGradOp
.. doxygenclass:: popart::LogOp
.. doxygenclass:: popart::LogSoftmaxGradOp
.. doxygenclass:: popart::LogSoftmaxInplaceOp
.. doxygenclass:: popart::LogSoftmaxOp
.. doxygenclass:: popart::LoopOp
.. doxygenclass:: popart::LossOp
.. doxygenclass:: popart::LossScaleUpdateOp
.. doxygenclass:: popart::MatMulBaseGradOp
.. doxygenclass:: popart::MatMulBaseOp
.. doxygenclass:: popart::MatMulLhsGradOp
.. doxygenclass:: popart::MatMulOp
.. doxygenclass:: popart::MatMulRhsGradOp
.. doxygenclass:: popart::MaxArgGradOp
.. doxygenclass:: popart::MaxOp
.. doxygenclass:: popart::MaxPoolGradOp
.. doxygenclass:: popart::MaxPoolOp
.. doxygenclass:: popart::MeanArgGradOp
.. doxygenclass:: popart::MeanOp
.. doxygenclass:: popart::MinArgGradOp
.. doxygenclass:: popart::MinOp
.. doxygenclass:: popart::ModifyRandomSeedOp
.. doxygenclass:: popart::MulArg0GradOp
.. doxygenclass:: popart::MulArg1GradOp
.. doxygenclass:: popart::MulLhsInplaceOp
.. doxygenclass:: popart::MulRhsInplaceOp
.. doxygenclass:: popart::MultiCollectiveBaseOp
.. doxygenclass:: popart::MultiConvBaseOp
.. doxygenclass:: popart::MultiConvDataGradBaseOp
.. doxygenclass:: popart::MultiConvDataGradOp
.. doxygenclass:: popart::MultiConvOp
.. doxygenclass:: popart::MultiConvWeightsGradBaseOp
.. doxygenclass:: popart::MultiConvWeightsGradOp
.. doxygenclass:: popart::MultiExchangeOp
.. doxygenclass:: popart::MultiReplicatedAllReduceOp
.. doxygenclass:: popart::NegateGradOp
.. doxygenclass:: popart::NegateOp
.. doxygenclass:: popart::NllGradOp
.. doxygenclass:: popart::NllOp
.. doxygenclass:: popart::NlllWithSoftmaxGradDirectOp
.. doxygenclass:: popart::NonLinearVariadicGradOp
.. doxygenclass:: popart::NopOp
.. doxygenclass:: popart::NotOp
.. doxygenclass:: popart::OneWayUnaryInPlaceOp
.. doxygenclass:: popart::OneWayUnaryOp
.. doxygenclass:: popart::OnehotGradOp
.. doxygenclass:: popart::OnehotOp
.. doxygenclass:: popart::OrOp
.. doxygenclass:: popart::PReluOp
.. doxygenclass:: popart::PackedDataBlockOp
.. doxygenclass:: popart::PadGradOp
.. doxygenclass:: popart::PadInplaceOp
.. doxygenclass:: popart::PadOp
.. doxygenclass:: popart::ParameterizedOp
.. doxygenclass:: popart::PlaceholderOp
.. doxygenclass:: popart::PopartLSTMGradOp
.. doxygenclass:: popart::PopartLSTMOp
.. doxygenclass:: popart::PowArg0GradOp
.. doxygenclass:: popart::PowArg1GradOp
.. doxygenclass:: popart::PowLhsInplaceOp
.. doxygenclass:: popart::PrintTensorOp
.. doxygenclass:: popart::RMSPropUpdaterOp
.. doxygenclass:: popart::RNNGradOp
.. doxygenclass:: popart::RNNOp
.. doxygenclass:: popart::RandomBaseOp
.. doxygenclass:: popart::RandomNormalBaseOp
.. doxygenclass:: popart::RandomNormalLikeOp
.. doxygenclass:: popart::RandomNormalOp
.. doxygenclass:: popart::RandomUniformBaseOp
.. doxygenclass:: popart::RandomUniformLikeOp
.. doxygenclass:: popart::RandomUniformOp
.. doxygenclass:: popart::ReciprocalGradOp
.. doxygenclass:: popart::ReciprocalOp
.. doxygenclass:: popart::ReduceGradOp
.. doxygenclass:: popart::ReduceL1GradOp
.. doxygenclass:: popart::ReduceL1Op
.. doxygenclass:: popart::ReduceL2GradOp
.. doxygenclass:: popart::ReduceL2Op
.. doxygenclass:: popart::ReduceLogSumExpGradOp
.. doxygenclass:: popart::ReduceLogSumExpOp
.. doxygenclass:: popart::ReduceLogSumGradOp
.. doxygenclass:: popart::ReduceLogSumOp
.. doxygenclass:: popart::ReduceMaxGradOp
.. doxygenclass:: popart::ReduceMaxOp
.. doxygenclass:: popart::ReduceMeanGradOp
.. doxygenclass:: popart::ReduceMeanOp
.. doxygenclass:: popart::ReduceMedianGradOp
.. doxygenclass:: popart::ReduceMedianOp
.. doxygenclass:: popart::ReduceMinGradOp
.. doxygenclass:: popart::ReduceMinOp
.. doxygenclass:: popart::ReduceOp
.. doxygenclass:: popart::ReduceProdGradOp
.. doxygenclass:: popart::ReduceProdOp
.. doxygenclass:: popart::ReduceSumGradOp
.. doxygenclass:: popart::ReduceSumOp
.. doxygenclass:: popart::ReduceSumSquareGradOp
.. doxygenclass:: popart::ReduceSumSquareOp
.. doxygenclass:: popart::ReluGradOp
.. doxygenclass:: popart::ReluInplaceOp
.. doxygenclass:: popart::ReluOp
.. doxygenclass:: popart::RemoteBaseOp
.. doxygenclass:: popart::RemoteLoadInplaceOp
.. doxygenclass:: popart::RemoteLoadOp
.. doxygenclass:: popart::RemoteStoreOp
.. doxygenclass:: popart::ReplicatedAllGatherOp
.. doxygenclass:: popart::ReplicatedAllReduceInplaceOp
.. doxygenclass:: popart::ReplicatedAllReduceOp
.. doxygenclass:: popart::ReplicatedReduceScatterOp
.. doxygenclass:: popart::ReshapeBaseOp
.. doxygenclass:: popart::ReshapeGradOp
.. doxygenclass:: popart::ReshapeInplaceOp
.. doxygenclass:: popart::ReshapeOp
.. doxygenclass:: popart::ResizeGradOp
.. doxygenclass:: popart::ResizeOp
.. doxygenclass:: popart::RestoreInplaceOp
.. doxygenclass:: popart::RestoreOp
.. doxygenclass:: popart::ReverseBaseOp
.. doxygenclass:: popart::ReverseGradOp
.. doxygenclass:: popart::ReverseInplaceOp
.. doxygenclass:: popart::ReverseOp
.. doxygenclass:: popart::RoiAlignGradOp
.. doxygenclass:: popart::RoiAlignOp
.. doxygenclass:: popart::RoundInplaceOp
.. doxygenclass:: popart::RoundOp
.. doxygenclass:: popart::ScaleGradOp
.. doxygenclass:: popart::ScaleInplaceOp
.. doxygenclass:: popart::ScaleOp
.. doxygenclass:: popart::ScaledAddLhsInplaceOp
.. doxygenclass:: popart::ScaledAddOp
.. doxygenclass:: popart::ScaledAddRhsInplaceOp
.. doxygenclass:: popart::ScanOp
.. doxygenclass:: popart::ScatterDataGradOp
.. doxygenclass:: popart::ScatterOp
.. doxygenclass:: popart::ScatterReduceGradOp
.. doxygenclass:: popart::ScatterReduceOp
.. doxygenclass:: popart::ScatterUpdateGradOp
.. doxygenclass:: popart::SeluGradOp
.. doxygenclass:: popart::SeluInplaceOp
.. doxygenclass:: popart::SeluOp
.. doxygenclass:: popart::SequenceSliceInplaceOp
.. doxygenclass:: popart::SequenceSliceOp
.. doxygenclass:: popart::ShapeOrLikeOp
.. doxygenclass:: popart::ShapedDropoutOp
.. doxygenclass:: popart::ShapedDropoutGradOp
.. doxygenclass:: popart::ShrinkGradOp
.. doxygenclass:: popart::ShrinkInplaceOp
.. doxygenclass:: popart::ShrinkOp
.. doxygenclass:: popart::SigmoidGradOp
.. doxygenclass:: popart::SigmoidInplaceOp
.. doxygenclass:: popart::SigmoidOp
.. doxygenclass:: popart::SignInplaceOp
.. doxygenclass:: popart::SignOp
.. doxygenclass:: popart::SinGradOp
.. doxygenclass:: popart::SinOp
.. doxygenclass:: popart::SinhGradOp
.. doxygenclass:: popart::SinhInplaceOp
.. doxygenclass:: popart::SinhOp
.. doxygenclass:: popart::SliceGradOp
.. doxygenclass:: popart::SliceInplaceOp
.. doxygenclass:: popart::SliceOp
.. doxygenclass:: popart::SoftPlusGradOp
.. doxygenclass:: popart::SoftPlusInplaceOp
.. doxygenclass:: popart::SoftPlusOp
.. doxygenclass:: popart::SoftSignGradOp
.. doxygenclass:: popart::SoftSignInplaceOp
.. doxygenclass:: popart::SoftSignOp
.. doxygenclass:: popart::SoftmaxGradDirectOp
.. doxygenclass:: popart::SoftmaxGradOp
.. doxygenclass:: popart::SoftmaxInplaceOp
.. doxygenclass:: popart::SoftmaxOp
.. doxygenclass:: popart::SplitGradOp
.. doxygenclass:: popart::SplitOp
.. doxygenclass:: popart::SqrtGradOp
.. doxygenclass:: popart::SqrtOp
.. doxygenclass:: popart::SquareOp
.. doxygenclass:: popart::StashOp
.. doxygenclass:: popart::SubgraphOp
.. doxygenclass:: popart::SubsampleBaseOp
.. doxygenclass:: popart::SubsampleGradOp
.. doxygenclass:: popart::SubsampleInplaceOp
.. doxygenclass:: popart::SubsampleOp
.. doxygenclass:: popart::SubtractArg0GradOp
.. doxygenclass:: popart::SubtractArg1GradOp
.. doxygenclass:: popart::SumArgGradOp
.. doxygenclass:: popart::SumOp
.. doxygenclass:: popart::SwishGradOp
.. doxygenclass:: popart::SwishInplaceOp
.. doxygenclass:: popart::SwishOp
.. doxygenclass:: popart::SyncOp
.. doxygenclass:: popart::TanhGradOp
.. doxygenclass:: popart::TanhOp
.. doxygenclass:: popart::TensorRemapOp
.. doxygenclass:: popart::ThresholdedReluGradOp
.. doxygenclass:: popart::ThresholdedReluInplaceOp
.. doxygenclass:: popart::ThresholdedReluOp
.. doxygenclass:: popart::TiedGatherGradOp
.. doxygenclass:: popart::TiedGatherOp
.. doxygenclass:: popart::TileGradOp
.. doxygenclass:: popart::TileOp
.. doxygenclass:: popart::TopKGradOp
.. doxygenclass:: popart::TopKOp
.. doxygenclass:: popart::TransposeBaseOp
.. doxygenclass:: popart::TransposeGradOp
.. doxygenclass:: popart::TransposeInplaceOp
.. doxygenclass:: popart::TransposeOp
.. doxygenclass:: popart::UnaryZeroGradOp
.. doxygenclass:: popart::UpsampleOp
.. doxygenclass:: popart::VariadicGradOp
.. doxygenclass:: popart::VariadicOp
.. doxygenclass:: popart::WhereLhsInplaceOp
.. doxygenclass:: popart::WhereOp
.. doxygenclass:: popart::WhereRhsInplaceOp
.. doxygenclass:: popart::WhereXGradOp
.. doxygenclass:: popart::WhereYGradOp
.. doxygenclass:: popart::ZerosBaseOp
.. doxygenclass:: popart::ZerosLikeOp
.. doxygenclass:: popart::ZerosOp


Available Ops (Opx class)
.........................

.. doxygenclass:: popart::popx::AbortOpx
.. doxygenclass:: popart::popx::AbsOpx
.. doxygenclass:: popart::popx::AccumulateBaseOpx
.. doxygenclass:: popart::popx::AccumulateOpx
.. doxygenclass:: popart::popx::AccumulatorScaleOpx
.. doxygenclass:: popart::popx::AdaDeltaUpdaterOpx
.. doxygenclass:: popart::popx::AdamUpdaterOpx
.. doxygenclass:: popart::popx::AdamVarUpdateOpx
.. doxygenclass:: popart::popx::AddArg0GradOpx
.. doxygenclass:: popart::popx::AddArg1GradOpx
.. doxygenclass:: popart::popx::AddBiasBiasGradOpx
.. doxygenclass:: popart::popx::AddBiasDataGradOpx
.. doxygenclass:: popart::popx::AddBiasInplaceOpx
.. doxygenclass:: popart::popx::AddBiasOpx
.. doxygenclass:: popart::popx::AddLhsInplaceOpx
.. doxygenclass:: popart::popx::AddOpx
.. doxygenclass:: popart::popx::AddRhsInplaceOpx
.. doxygenclass:: popart::popx::AllReduceOpx
.. doxygenclass:: popart::popx::AndOpx
.. doxygenclass:: popart::popx::ArgExtremaOpx
.. doxygenclass:: popart::popx::ArgMaxOpx
.. doxygenclass:: popart::popx::ArgMinOpx
.. doxygenclass:: popart::popx::AsinGradOpx
.. doxygenclass:: popart::popx::AsinInplaceOpx
.. doxygenclass:: popart::popx::AsinOpx
.. doxygenclass:: popart::popx::Atan2LhsInplaceOpx
.. doxygenclass:: popart::popx::Atan2Opx
.. doxygenclass:: popart::popx::AtanGradOpx
.. doxygenclass:: popart::popx::AtanInplaceOpx
.. doxygenclass:: popart::popx::AtanOpx
.. doxygenclass:: popart::popx::BaseConcatOpx
.. doxygenclass:: popart::popx::BaseExpandOpx
.. doxygenclass:: popart::popx::BasePadOpx
.. doxygenclass:: popart::popx::BaseSliceOpx
.. doxygenclass:: popart::popx::BaseSortOpx
.. doxygenclass:: popart::popx::BaseWhereOpx
.. doxygenclass:: popart::popx::BatchNormGradOpx
.. doxygenclass:: popart::popx::BatchNormOpx
.. doxygenclass:: popart::popx::BinaryComparisonOpx
.. doxygenclass:: popart::popx::BitwiseBinaryOpx
.. doxygenclass:: popart::popx::BitwiseNotOpx
.. doxygenclass:: popart::popx::CallGradOpx
.. doxygenclass:: popart::popx::CallOpx
.. doxygenclass:: popart::popx::CastGradOpx
.. doxygenclass:: popart::popx::CastOpx
.. doxygenclass:: popart::popx::CeilInplaceOpx
.. doxygenclass:: popart::popx::CeilOpx
.. doxygenclass:: popart::popx::ClipGradOpx
.. doxygenclass:: popart::popx::ClipInplaceOpx
.. doxygenclass:: popart::popx::ClipOpx
.. doxygenclass:: popart::popx::CollectivesBaseOpx
.. doxygenclass:: popart::popx::ConcatGradOpx
.. doxygenclass:: popart::popx::ConcatInplaceOpx
.. doxygenclass:: popart::popx::ConcatOpx
.. doxygenclass:: popart::popx::ConvFlipWeightsGradOpx
.. doxygenclass:: popart::popx::ConvOpx
.. doxygenclass:: popart::popx::ConvWeightsGradOpx
.. doxygenclass:: popart::popx::CopyVarUpdateOpx
.. doxygenclass:: popart::popx::CosOpx
.. doxygenclass:: popart::popx::CtcBeamSearchDecoderOpx
.. doxygenclass:: popart::popx::CtcGradOpx
.. doxygenclass:: popart::popx::CtcOpx
.. doxygenclass:: popart::popx::CumSumGradOpx
.. doxygenclass:: popart::popx::CumSumOpx
.. doxygenclass:: popart::popx::DetachInplaceOpx
.. doxygenclass:: popart::popx::DetachOpx
.. doxygenclass:: popart::popx::DivOpx
.. doxygenclass:: popart::popx::DropoutOpx
.. doxygenclass:: popart::popx::DynamicAddInplaceOpx
.. doxygenclass:: popart::popx::DynamicAddOpx
.. doxygenclass:: popart::popx::DynamicSliceInplaceOpx
.. doxygenclass:: popart::popx::DynamicSliceOpx
.. doxygenclass:: popart::popx::DynamicUpdateInplaceOpx
.. doxygenclass:: popart::popx::DynamicUpdateOpx
.. doxygenclass:: popart::popx::DynamicZeroInplaceOpx
.. doxygenclass:: popart::popx::DynamicZeroOpx
.. doxygenclass:: popart::popx::ElementWiseBinaryInplaceOpx
.. doxygenclass:: popart::popx::ElementWiseBinaryOpx
.. doxygenclass:: popart::popx::ElementWiseBinaryOutplaceOpx
.. doxygenclass:: popart::popx::ElementWiseUnaryInplaceOpx
.. doxygenclass:: popart::popx::ElementWiseUnaryOpx
.. doxygenclass:: popart::popx::ElementWiseUnaryOutplaceOpx
.. doxygenclass:: popart::popx::EluGradOpx
.. doxygenclass:: popart::popx::EluInplaceOpx
.. doxygenclass:: popart::popx::EluOpx
.. doxygenclass:: popart::popx::EqualOpx
.. doxygenclass:: popart::popx::ErfxGradOpx
.. doxygenclass:: popart::popx::ErfxOpx
.. doxygenclass:: popart::popx::ExchangeBaseOpx
.. doxygenclass:: popart::popx::ExpInplaceOpx
.. doxygenclass:: popart::popx::ExpOpx
.. doxygenclass:: popart::popx::ExpandGradOpx
.. doxygenclass:: popart::popx::ExpandInplaceOpx
.. doxygenclass:: popart::popx::ExpandOpx
.. doxygenclass:: popart::popx::Expm1InplaceOpx
.. doxygenclass:: popart::popx::Expm1Opx
.. doxygenclass:: popart::popx::FloorInplaceOpx
.. doxygenclass:: popart::popx::FloorOpx
.. doxygenclass:: popart::popx::FmodOpx
.. doxygenclass:: popart::popx::GRUGradOpx
.. doxygenclass:: popart::popx::GRUOpx
.. doxygenclass:: popart::popx::GatherBaseOpx
.. doxygenclass:: popart::popx::GatherGradOpx
.. doxygenclass:: popart::popx::GatherOpx
.. doxygenclass:: popart::popx::GeluGradOpx
.. doxygenclass:: popart::popx::GeluInplaceOpx
.. doxygenclass:: popart::popx::GeluOpx
.. doxygenclass:: popart::popx::GetRandomSeedOpx
.. doxygenclass:: popart::popx::GreaterOpx
.. doxygenclass:: popart::popx::GroupNormGradOpx
.. doxygenclass:: popart::popx::GroupNormOpx
.. doxygenclass:: popart::popx::HardSigmoidGradOpx
.. doxygenclass:: popart::popx::HardSigmoidInplaceOpx
.. doxygenclass:: popart::popx::HardSigmoidOpx
.. doxygenclass:: popart::popx::HistogramOpx
.. doxygenclass:: popart::popx::HostBaseOpx
.. doxygenclass:: popart::popx::HostLoadInplaceOpx
.. doxygenclass:: popart::popx::HostLoadOpx
.. doxygenclass:: popart::popx::HostStoreOpx
.. doxygenclass:: popart::popx::IdentityGradOpx
.. doxygenclass:: popart::popx::IdentityInplaceOpx
.. doxygenclass:: popart::popx::IdentityLossGradOpx
.. doxygenclass:: popart::popx::IdentityLossOpx
.. doxygenclass:: popart::popx::IdentityOpx
.. doxygenclass:: popart::popx::IfGradOpx
.. doxygenclass:: popart::popx::IfOpx
.. doxygenclass:: popart::popx::IncrementModInplaceOpx
.. doxygenclass:: popart::popx::IncrementModOpx
.. doxygenclass:: popart::popx::InitOpx
.. doxygenclass:: popart::popx::InstanceNormGradOpx
.. doxygenclass:: popart::popx::InstanceNormOpx
.. doxygenclass:: popart::popx::IoTileCopyOpx
.. doxygenclass:: popart::popx::IpuCopyOpx
.. doxygenclass:: popart::popx::L1GradOpx
.. doxygenclass:: popart::popx::L1Opx
.. doxygenclass:: popart::popx::LRNGradOpx
.. doxygenclass:: popart::popx::LRNOpx
.. doxygenclass:: popart::popx::LSTMGradOpx
.. doxygenclass:: popart::popx::LSTMOpx
.. doxygenclass:: popart::popx::LambSquareOpx
.. doxygenclass:: popart::popx::LeakyReluGradOpx
.. doxygenclass:: popart::popx::LeakyReluInplaceOpx
.. doxygenclass:: popart::popx::LeakyReluOpx
.. doxygenclass:: popart::popx::LessOpx
.. doxygenclass:: popart::popx::Log1pInplaceOpx
.. doxygenclass:: popart::popx::Log1pOpx
.. doxygenclass:: popart::popx::LogOpx
.. doxygenclass:: popart::popx::LogSoftmaxGradOpx
.. doxygenclass:: popart::popx::LogSoftmaxInplaceOpx
.. doxygenclass:: popart::popx::LogSoftmaxOpx
.. doxygenclass:: popart::popx::LoopOpx
.. doxygenclass:: popart::popx::LossScaleUpdateOpx
.. doxygenclass:: popart::popx::MatMulOpx
.. doxygenclass:: popart::popx::MaxArgGradOpx
.. doxygenclass:: popart::popx::MaxOpx
.. doxygenclass:: popart::popx::MeanArgGradOpx
.. doxygenclass:: popart::popx::MeanOpx
.. doxygenclass:: popart::popx::MinArgGradOpx
.. doxygenclass:: popart::popx::MinOpx
.. doxygenclass:: popart::popx::ModifyRandomSeedOpx
.. doxygenclass:: popart::popx::MulLhsInplaceOpx
.. doxygenclass:: popart::popx::MulOpx
.. doxygenclass:: popart::popx::MulRhsInplaceOpx
.. doxygenclass:: popart::popx::MultiCollectiveBaseOpx
.. doxygenclass:: popart::popx::MultiConvBaseOpx
.. doxygenclass:: popart::popx::MultiConvOpx
.. doxygenclass:: popart::popx::MultiConvWeightsGradBaseOpx
.. doxygenclass:: popart::popx::MultiConvWeightsGradOpx
.. doxygenclass:: popart::popx::MultiExchangeOpx
.. doxygenclass:: popart::popx::MultiReplicatedAllReduceOpx
.. doxygenclass:: popart::popx::NegateGradOpx
.. doxygenclass:: popart::popx::NegateOpx
.. doxygenclass:: popart::popx::NllGradOpx
.. doxygenclass:: popart::popx::NllOpx
.. doxygenclass:: popart::popx::NlllWithSoftmaxGradDirectOpx
.. doxygenclass:: popart::popx::NopOpx
.. doxygenclass:: popart::popx::NormOpx
.. doxygenclass:: popart::popx::NotOpx
.. doxygenclass:: popart::popx::OnehotGradOpx
.. doxygenclass:: popart::popx::OnehotOpx
.. doxygenclass:: popart::popx::OrOpx
.. doxygenclass:: popart::popx::PReluOpx
.. doxygenclass:: popart::popx::PadGradOpx
.. doxygenclass:: popart::popx::PadInplaceOpx
.. doxygenclass:: popart::popx::PadOpx
.. doxygenclass:: popart::popx::PopartLSTMOpxBase
.. doxygenclass:: popart::popx::PowLhsInplaceOpx
.. doxygenclass:: popart::popx::PowOpx
.. doxygenclass:: popart::popx::PrintTensorOpx
.. doxygenclass:: popart::popx::RMSPropUpdaterOpx
.. doxygenclass:: popart::popx::RNNGradOpx
.. doxygenclass:: popart::popx::RNNOpx
.. doxygenclass:: popart::popx::RandomNormalOpx
.. doxygenclass:: popart::popx::RandomUniformOpx
.. doxygenclass:: popart::popx::ReciprocalOpx
.. doxygenclass:: popart::popx::ReduceL1GradOpx
.. doxygenclass:: popart::popx::ReduceL1Opx
.. doxygenclass:: popart::popx::ReduceL2GradOpx
.. doxygenclass:: popart::popx::ReduceL2Opx
.. doxygenclass:: popart::popx::ReduceLogSumExpGradOpx
.. doxygenclass:: popart::popx::ReduceLogSumExpOpx
.. doxygenclass:: popart::popx::ReduceLogSumGradOpx
.. doxygenclass:: popart::popx::ReduceLogSumOpx
.. doxygenclass:: popart::popx::ReduceMaxGradOpx
.. doxygenclass:: popart::popx::ReduceMaxOpx
.. doxygenclass:: popart::popx::ReduceMeanGradOpx
.. doxygenclass:: popart::popx::ReduceMeanOpx
.. doxygenclass:: popart::popx::ReduceMedianGradOpx
.. doxygenclass:: popart::popx::ReduceMedianOpx
.. doxygenclass:: popart::popx::ReduceMinGradOpx
.. doxygenclass:: popart::popx::ReduceMinOpx
.. doxygenclass:: popart::popx::ReduceProdGradOpx
.. doxygenclass:: popart::popx::ReduceProdOpx
.. doxygenclass:: popart::popx::ReduceSumGradOpx
.. doxygenclass:: popart::popx::ReduceSumOpx
.. doxygenclass:: popart::popx::ReduceSumSquareGradOpx
.. doxygenclass:: popart::popx::ReduceSumSquareOpx
.. doxygenclass:: popart::popx::ReluGradOpx
.. doxygenclass:: popart::popx::ReluInplaceOpx
.. doxygenclass:: popart::popx::ReluOpx
.. doxygenclass:: popart::popx::RemoteBaseOpx
.. doxygenclass:: popart::popx::RemoteLoadInplaceOpx
.. doxygenclass:: popart::popx::RemoteLoadOpx
.. doxygenclass:: popart::popx::RemoteStoreOpx
.. doxygenclass:: popart::popx::ReplicatedAllGatherOpx
.. doxygenclass:: popart::popx::ReplicatedAllReduceInplaceOpx
.. doxygenclass:: popart::popx::ReplicatedAllReduceOpx
.. doxygenclass:: popart::popx::ReplicatedReduceScatterOpx
.. doxygenclass:: popart::popx::RescaleAccumulateOpx
.. doxygenclass:: popart::popx::ReshapeBaseOpx
.. doxygenclass:: popart::popx::ReshapeGradOpx
.. doxygenclass:: popart::popx::ReshapeInplaceOpx
.. doxygenclass:: popart::popx::ReshapeOpx
.. doxygenclass:: popart::popx::ResizeGradOpx
.. doxygenclass:: popart::popx::ResizeOpx
.. doxygenclass:: popart::popx::RestoreBaseOpx
.. doxygenclass:: popart::popx::ReverseBaseOpx
.. doxygenclass:: popart::popx::ReverseGradOpx
.. doxygenclass:: popart::popx::ReverseInplaceOpx
.. doxygenclass:: popart::popx::ReverseOpx
.. doxygenclass:: popart::popx::RoundInplaceOpx
.. doxygenclass:: popart::popx::RoundOpx
.. doxygenclass:: popart::popx::SGD0VarUpdateOpx
.. doxygenclass:: popart::popx::SGD1AcclUpdateOpx
.. doxygenclass:: popart::popx::SGD1VarUpdateOpx
.. doxygenclass:: popart::popx::ScaleInplaceOpx
.. doxygenclass:: popart::popx::ScaleGradOpx
.. doxygenclass:: popart::popx::ScaleOpx
.. doxygenclass:: popart::popx::ScaledAddLhsInplaceOpx
.. doxygenclass:: popart::popx::ScaledAddOpx
.. doxygenclass:: popart::popx::ScaledAddRhsInplaceOpx
.. doxygenclass:: popart::popx::ScaledVarUpdateOpx
.. doxygenclass:: popart::popx::ScatterDataGradOpx
.. doxygenclass:: popart::popx::ScatterOpx
.. doxygenclass:: popart::popx::ScatterReduceGradOpx
.. doxygenclass:: popart::popx::ScatterReduceOpx
.. doxygenclass:: popart::popx::ScatterUpdateGradOpx
.. doxygenclass:: popart::popx::SeluGradOpx
.. doxygenclass:: popart::popx::SeluInplaceOpx
.. doxygenclass:: popart::popx::SeluOpx
.. doxygenclass:: popart::popx::SequenceSliceInplaceOpx
.. doxygenclass:: popart::popx::SequenceSliceOpx
.. doxygenclass:: popart::popx::ShapedDropoutOpx
.. doxygenclass:: popart::popx::ShrinkGradOpx
.. doxygenclass:: popart::popx::ShrinkInplaceOpx
.. doxygenclass:: popart::popx::ShrinkOpx
.. doxygenclass:: popart::popx::SigmoidGradOpx
.. doxygenclass:: popart::popx::SigmoidInplaceOpx
.. doxygenclass:: popart::popx::SigmoidOpx
.. doxygenclass:: popart::popx::SignInplaceOpx
.. doxygenclass:: popart::popx::SignOpx
.. doxygenclass:: popart::popx::SinOpx
.. doxygenclass:: popart::popx::SinhGradOpx
.. doxygenclass:: popart::popx::SinhInplaceOpx
.. doxygenclass:: popart::popx::SinhOpx
.. doxygenclass:: popart::popx::SliceInplaceOpx
.. doxygenclass:: popart::popx::SliceOpx
.. doxygenclass:: popart::popx::SoftPlusGradOpx
.. doxygenclass:: popart::popx::SoftPlusInplaceOpx
.. doxygenclass:: popart::popx::SoftPlusOpx
.. doxygenclass:: popart::popx::SoftSignGradOpx
.. doxygenclass:: popart::popx::SoftSignInplaceOpx
.. doxygenclass:: popart::popx::SoftSignOpx
.. doxygenclass:: popart::popx::SoftmaxGradDirectOpx
.. doxygenclass:: popart::popx::SoftmaxGradOpx
.. doxygenclass:: popart::popx::SoftmaxInplaceOpx
.. doxygenclass:: popart::popx::SoftmaxOpx
.. doxygenclass:: popart::popx::SparseAccumulateOpx
.. doxygenclass:: popart::popx::SplitOpx
.. doxygenclass:: popart::popx::SqrtOpx
.. doxygenclass:: popart::popx::SquareOpx
.. doxygenclass:: popart::popx::StashOpx
.. doxygenclass:: popart::popx::SubgraphOpx
.. doxygenclass:: popart::popx::SubsampleGradOpx
.. doxygenclass:: popart::popx::SubsampleInplaceOpx
.. doxygenclass:: popart::popx::SubsampleOpx
.. doxygenclass:: popart::popx::SubtractArg0GradOpx
.. doxygenclass:: popart::popx::SubtractOpx
.. doxygenclass:: popart::popx::SumArgGradOpx
.. doxygenclass:: popart::popx::SumOpx
.. doxygenclass:: popart::popx::SwishGradOpx
.. doxygenclass:: popart::popx::SwishInplaceOpx
.. doxygenclass:: popart::popx::SwishOpx
.. doxygenclass:: popart::popx::SyncOpx
.. doxygenclass:: popart::popx::TanhGradOpx
.. doxygenclass:: popart::popx::TanhOpx
.. doxygenclass:: popart::popx::TensorRemapOpx
.. doxygenclass:: popart::popx::ThresholdedReluGradOpx
.. doxygenclass:: popart::popx::ThresholdedReluInplaceOpx
.. doxygenclass:: popart::popx::ThresholdedReluOpx
.. doxygenclass:: popart::popx::TiedGatherOpx
.. doxygenclass:: popart::popx::TileGradOpx
.. doxygenclass:: popart::popx::TileOpx
.. doxygenclass:: popart::popx::TopKGradOpx
.. doxygenclass:: popart::popx::TopKOpx
.. doxygenclass:: popart::popx::TransposeGradOpx
.. doxygenclass:: popart::popx::TransposeInplaceOpx
.. doxygenclass:: popart::popx::TransposeOpx
.. doxygenclass:: popart::popx::VarUpdateOpx
.. doxygenclass:: popart::popx::WhereLhsInplaceOpx
.. doxygenclass:: popart::popx::WhereOpx
.. doxygenclass:: popart::popx::WhereRhsInplaceOpx
.. doxygenclass:: popart::popx::WhereXGradOpx
.. doxygenclass:: popart::popx::WhereYGradOpx
.. doxygenclass:: popart::popx::ZerosOpx

Patterns
--------

.. code-block:: cpp

  #include <popart/patterns/patterns.hpp>

.. doxygenclass:: popart::Patterns
   :members:

.. doxygenclass:: popart::PreAliasPattern
   :members:

Available patterns
..................

.. doxygenclass:: popart::AllReduceToIdentityPattern
.. doxygenclass:: popart::BinaryGradOpPattern
.. doxygenclass:: popart::ContiguateIpuCopyIndicesPattern
.. doxygenclass:: popart::ConvDataGradPattern
.. doxygenclass:: popart::ConvFlipWeightsDoubleFlipPattern
.. doxygenclass:: popart::ConvFlipWeightsGradOpPattern
.. doxygenclass:: popart::ConvTransposePattern
.. doxygenclass:: popart::CosGradOpPattern
.. doxygenclass:: popart::CoshOpPattern
.. doxygenclass:: popart::DecomposeBinaryConstScalar
.. doxygenclass:: popart::ElementWiseGradOpPattern
.. doxygenclass:: popart::ExpGradOpPattern
.. doxygenclass:: popart::ExpandCastPattern
.. doxygenclass:: popart::Expm1GradOpPattern
.. doxygenclass:: popart::Fuser
.. doxygenclass:: popart::InitAccumulatePattern
.. doxygenclass:: popart::LSTMPattern
.. doxygenclass:: popart::LambSerialisedWeightPattern
.. doxygenclass:: popart::LikeOpsPattern
.. doxygenclass:: popart::Log1pGradOpPattern
.. doxygenclass:: popart::LogGradOpPattern
.. doxygenclass:: popart::LoopScanOutPattern
.. doxygenclass:: popart::MatMulGradPattern
.. doxygenclass:: popart::MatMulPattern
.. doxygenclass:: popart::MulArgGradOpPattern
.. doxygenclass:: popart::NlllWithSoftmaxGradDirect
.. doxygenclass:: popart::OptimizerDecompose
.. doxygenclass:: popart::PackedDataBlockPattern
.. doxygenclass:: popart::PadSumPattern
.. doxygenclass:: popart::PostNRepl
.. doxygenclass:: popart::PreUniRepl
.. doxygenclass:: popart::ReciprocalGradOpPattern
.. doxygenclass:: popart::RemoveUnnecessaryLossGradCast
.. doxygenclass:: popart::ScanToLoopPattern
.. doxygenclass:: popart::SequenceExpander
.. doxygenclass:: popart::SplitGatherPattern
.. doxygenclass:: popart::SplitOpPattern
.. doxygenclass:: popart::SqrtGradOpPattern
.. doxygenclass:: popart::SumToAddPattern
.. doxygenclass:: popart::TiedGatherAccumulatePattern
.. doxygenclass:: popart::TiedGatherPattern
.. doxygenclass:: popart::TransposeToIdentityOrReshapePattern
.. doxygenclass:: popart::UpsampleToResizePattern
.. doxygenclass:: popart::ViewSimplifyPattern
.. doxygenclass:: popart::AdamDecompose
.. doxygenclass:: popart::AdaptiveDecompose
.. doxygenclass:: popart::Atan2Arg0GradOpPattern
.. doxygenclass:: popart::Atan2Arg1GradOpPattern
.. doxygenclass:: popart::DivArg0GradOpPattern
.. doxygenclass:: popart::DivArg1GradOpPattern
.. doxygenclass:: popart::FmodArg0GradOpPattern
.. doxygenclass:: popart::MatMulLhsGradPattern
.. doxygenclass:: popart::MatMulRhsGradPattern
.. doxygenclass:: popart::NegativeOneScalePattern
.. doxygenclass:: popart::OpToIdentityPattern
.. doxygenclass:: popart::PowArg0GradOpPattern
.. doxygenclass:: popart::PowArg1GradOpPattern
.. doxygenclass:: popart::SGD0Decompose
.. doxygenclass:: popart::SGD1Decompose
.. doxygenclass:: popart::SGD2Decompose
.. doxygenclass:: popart::SoftmaxGradDirect
.. doxygenclass:: popart::SplitGradOpToConcatPattern
.. doxygenclass:: popart::SubtractArg1GradOpPattern

Transforms
----------

.. code-block:: cpp

  #include <popart/transforms/transform.hpp>

.. doxygenclass:: popart::Transform

.. code-block:: cpp

Available transforms
....................

.. doxygenclass:: popart::AccumulateOuterFragmentParallelizer
.. doxygenclass:: popart::AutoVirtualGraph
.. doxygenclass:: popart::Autodiff
.. doxygenclass:: popart::AutomaticLossScale
.. doxygenclass:: popart::BatchSerialize
.. doxygenclass:: popart::ClipWeightGradientsByNorm
.. doxygenclass:: popart::ContiguateCollectivesTransform
.. doxygenclass:: popart::DecomposeGradSum
.. doxygenclass:: popart::DecomposeLoops
.. doxygenclass:: popart::DynamicOpTransform
.. doxygenclass:: popart::EnsureFp32LossScale
.. doxygenclass:: popart::ExplicitRecompute
.. doxygenclass:: popart::HostIOSetup
.. doxygenclass:: popart::InferPipelineStages
.. doxygenclass:: popart::InplaceAccumulateGradPartialsIntoOptimizerAccumTensor
.. doxygenclass:: popart::InterIpuCopy
.. doxygenclass:: popart::IoComputeTileCopy
.. doxygenclass:: popart::MainLoops
.. doxygenclass:: popart::MergeAllVarUpdates
.. doxygenclass:: popart::MergeAuto
.. doxygenclass:: popart::MergeLooseThreshold
.. doxygenclass:: popart::MergeTightThreshold
.. doxygenclass:: popart::MergeCollectivesTransform
.. doxygenclass:: popart::MergeCopies
.. doxygenclass:: popart::MergeDuplicateOps
.. doxygenclass:: popart::MergeExchange
.. doxygenclass:: popart::MergeLoops
.. doxygenclass:: popart::MergeVarUpdates
.. doxygenclass:: popart::OverlapIO
.. doxygenclass:: popart::Pipeline
.. doxygenclass:: popart::PreAutomaticLossScale
.. doxygenclass:: popart::Prune
.. doxygenclass:: popart::RandomSetup
.. doxygenclass:: popart::RemoteSetup
.. doxygenclass:: popart::SerializeMatMuls
.. doxygenclass:: popart::StochasticRounding
.. doxygenclass:: popart::StreamingMemory
.. doxygenclass:: popart::SubgraphOutline

.. code-block:: cpp

  #include <popart/bwdgraphinfo.hpp>

.. doxygenstruct:: popart::BwdGraphInfo
  :members:

.. doxygenenum:: popart::ExpectedConnectionType

.. doxygenstruct:: popart::ExpectedConnection
  :members:


Utility classes
---------------

Graph
.....

.. code-block:: cpp

  #include <popart/graphutils.hpp>

.. doxygentypedef:: popart::graphutils::CallStack
.. doxygentypedef:: popart::graphutils::TensorAndCallStack


Region
......

.. code-block:: cpp

  #include <popart/region.hpp>

.. doxygenfile:: region.hpp
   :sections: func enum innerclass

Error handling
..............

.. code-block:: cpp

  #include <popart/error.hpp>

.. doxygenenum:: popart::ErrorSource

.. doxygenclass:: popart::error
   :members:

.. doxygenclass:: popart::internal_error
.. doxygenclass:: popart::memory_allocation_err
.. doxygenclass:: popart::runtime_error
.. doxygenclass:: popart::popx::devicex_memory_allocation_err


Debug context
.............

.. code-block:: cpp

  #include <popart/debugcontext.hpp>

.. doxygenclass:: popart::DebugContext
   :members:

.. doxygenclass:: popart::DebugInfo
.. doxygenclass:: popart::OnnxOpDebugInfo
.. doxygenclass:: popart::OnnxVariableDebugInfo
.. doxygenclass:: popart::OpDebugInfo
.. doxygenclass:: popart::TensorDebugInfo


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


Profiling
.........

.. code-block:: cpp

  #include <popart/liveness.hpp>

.. doxygenclass:: popart::liveness::LivenessAnalyzer

.. code-block:: cpp

  #include <popart/subgraphpartitioner.hpp>

.. doxygenclass:: popart::liveness::SubgraphPartitioner

.. code-block:: cpp

  #include <popart/aliaszerocopy.hpp>

.. doxygenclass:: popart::liveness::AliasZeroCopy

.. doxygenclass:: popart::liveness::Intervals

.. doxygenenum:: popart::liveness::ProducerInterval


Task information
................

.. code-block:: cpp

  #include <popart/taskid.hpp>

.. doxygenclass:: popart::TaskId


Type definitions
................

.. doxygenfile:: names.hpp
  :sections: innernamespace typedef enum

.. doxygentypedef:: popart::FwdGraphToBwdGraphInfo
.. doxygentypedef:: popart::popx::PreparedCopyTensors
.. doxygentypedef:: popart::popx::PreparedTensorInfos


Enums
.....

.. doxygenenum:: popart::AccumulationType
.. doxygenenum:: popart::ActivationFunction
.. doxygenenum:: popart::AutoPad
.. doxygenenum:: popart::CollectiveOperator
.. doxygenenum:: popart::CommGroupType
.. doxygenenum:: popart::DeviceSelectionCriterion
.. doxygenenum:: popart::InitType
.. doxygenenum:: popart::MatMulPartialsType
.. doxygenenum:: popart::ResizeCoordinateTransformationMode
.. doxygenenum:: popart::ResizeMode
.. doxygenenum:: popart::ResizeNearestMode
.. doxygenenum:: popart::ScatterReduction
.. doxygenenum:: popart::TensorRemapType

Structs
.......

.. doxygenstruct:: popart::BranchInfo
.. doxygenstruct:: popart::ClonedGraphMaps
.. doxygenstruct:: popart::ConvParameters
.. doxygenstruct:: popart::popx::OpxInAndOutIndex
.. doxygenstruct:: popart::PTensorCmp
.. doxygenstruct:: popart::ReplicatedTensorShardingOpInfo


Other classes
.............

.. doxygenclass:: popart::BasicOptional
.. doxygenclass:: popart::ExchangeDescriptor
.. doxygenclass:: popart::GraphId
.. doxygenclass:: popart::LeakyReluOpBaseAttributes
.. doxygenclass:: popart::MultiConvOptions
.. doxygenclass:: popart::OpEquivIdCreator
.. doxygenclass:: popart::OpJsonSerialiser
.. doxygenclass:: popart::OpSerialiser
.. doxygenclass:: popart::OpSerialiserBase
.. doxygenclass:: popart::PriTaskDependency
.. doxygenclass:: popart::ReplicaEqualAnalysisProxy
.. doxygenclass:: popart::ReplicatedTensorShardingTracer
.. doxygenclass:: popart::TensorLocationInfo
.. doxygenclass:: popart::popx::InputCreatorCandidate
.. doxygenclass:: popart::popx::InputMultiCreatorCandidate
.. doxygenclass:: popart::popx::IsInfx
.. doxygenclass:: popart::popx::IsNaNx
.. doxygenclass:: popart::popx::ViewChanger
.. doxygenclass:: popart::popx::ViewChangers
.. doxygenclass:: popart::popx::ReplicatedGatherInScatterOutViewChanger
.. doxygenclass:: popart::popx::ReplicatedGatherOutScatterInViewChanger
.. doxygenclass:: popart::popx::serialization::Reader
