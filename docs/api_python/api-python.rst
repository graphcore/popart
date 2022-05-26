PopART Python API
=================

Sessions
--------

Training session
^^^^^^^^^^^^^^^^

.. autoclass:: popart.TrainingSession

Inference session
^^^^^^^^^^^^^^^^^

.. autoclass:: popart.InferenceSession

Session Options
^^^^^^^^^^^^^^^

.. autoclass:: popart.SessionOptions
.. autoclass:: popart.AccumulateOuterFragmentSchedule
.. autoclass:: popart.AccumulateOuterFragmentSettings
.. autoclass:: popart.AutodiffSettings
.. autoclass:: popart.AutodiffStitchStrategy
.. autoclass:: popart.AutomaticLossScalingSettings
.. autoclass:: popart.BatchSerializationBatchSchedule
.. autoclass:: popart.BatchSerializationMethod
.. autoclass:: popart.BatchSerializationSettings
.. autoclass:: popart.BatchSerializationTransformContext
.. autoclass:: popart.CommGroup
.. autoclass:: popart.CommGroupType
.. autoclass:: popart.DeveloperSettings
.. autoclass:: popart.ExecutionPhaseIOSchedule
.. autoclass:: popart.ExecutionPhaseSchedule
.. autoclass:: popart.ExecutionPhaseSettings
.. autoclass:: popart.GradientTensorTrackingMethod
.. autoclass:: popart.Instrumentation
.. autoclass:: popart.IrSerializationFormat
.. autoclass:: popart.MeanReductionStrategy
.. autoclass:: popart.MergeVarUpdateType
.. autoclass:: popart.RecomputationType
.. autoclass:: popart.ReductionType
.. autoclass:: popart.ReplicatedTensorSharding
.. autoclass:: popart.SubgraphCopyingStrategy
.. autoclass:: popart.SyntheticDataMode
.. autoclass:: popart.TensorLocationSettings
.. autoclass:: popart.TileSet
.. autoclass:: popart.VariableRetrievalMode
.. autoclass:: popart.VariableSettings
.. autoclass:: popart.VirtualGraphMode


Data input and output
---------------------

.. note:: The base class for data input and output in PopART is
    :cpp:class:`popart::IStepIO`. The way in which this class is used is
    detailed in the :doc:`popart-cpp-api:index` in the
    :ref:`popart-cpp-api:data input and output (istepio)` section.

.. autoclass:: popart.PyStepIO
    :special-members: __init__

.. autoclass:: popart.PyStepIOCallback
    :special-members: __init__

.. autoclass:: popart.InputShapeInfo
    :special-members: __init__

.. autoclass:: popart.DataFlow
    :special-members: __init__

Tensors
-------

.. autoclass:: popart.DataType
.. autoclass:: popart.DataTypeInfo
.. autoclass:: popart.ReplicatedTensorSharding
.. autoclass:: popart.Tensor
.. autoclass:: popart.TensorIndexMap
.. autoclass:: popart.TensorInfo
.. autoclass:: popart.TensorLocation
.. autoclass:: popart.TensorStorage
.. autoclass:: popart.TensorType
.. autoclass:: popart.TileSet
.. autoclass:: popart.VariableUpdateType
.. automodule:: popart.tensorinfo

Optimizers
----------

.. autoclass:: popart.Optimizer
.. autoclass:: popart.OptimizerType
.. autoclass:: popart.OptimizerReductionType
.. autoclass:: popart.WeightDecayMode
.. autoclass:: popart.OptimizerValue
.. autoclass:: popart.OptimizerValueMap

SGD
^^^

.. autoclass:: popart.ClipNormSettings
.. autoclass:: popart.SGD
.. autoclass:: popart.ConstSGD
.. autoclass:: popart.SGDAccumulatorAndMomentum

ConstSGD
^^^^^^^^

.. autoclass:: popart.ConstSGD

Adam
^^^^

.. autoclass:: popart.AdamMode
.. autoclass:: popart.Adam

AdaDelta, RMSProp & AdaGrad
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: popart.AdaptiveMode
.. autoclass:: popart.Adaptive

Builder
--------

.. automodule:: popart.builder
.. autoclass:: popart.Builder
.. autoclass:: popart.Ir
.. autoclass:: popart.HashesMap
.. autoclass:: popart.RequireOptimalSchedule
.. autoclass:: popart.Graph
.. autoclass:: popart.AiOnnxMlOpset1
.. autoclass:: popart.AiGraphcoreOpset1
.. autoclass:: popart.Scope


AiGraphcoreOpset1
^^^^^^^^^^^^^^^^^

.. autoclass:: popart.AiGraphcoreOpset1

Data flow
---------

.. autoclass:: popart.AnchorReturnTypeId
.. autoclass:: popart.ExchangeStrategy
.. autoclass:: popart.AnchorReturnType
.. autoclass:: popart.DataFlow
.. autoclass:: popart.InputSettings
.. autoclass:: popart.AnchorReturnTypeMap
.. autoclass:: popart.ReplicatedStreamMode

Device manager
--------------

.. autoclass:: popart.DeviceType
.. autoclass:: popart.DeviceConnectionType
.. autoclass:: popart.SyncPattern
.. autoclass:: popart.DeviceInfo
.. autoclass:: popart.popx.DevicexInfo
.. autoclass:: popart.popx.DevicexCpuInfo
.. autoclass:: popart.popx.DevicexIpuInfo
.. autoclass:: popart.popx.DevicexIpuModelInfo
.. autoclass:: popart.popx.DevicexSimInfo
.. autoclass:: popart.popx.DevicexOfflineIpuInfo
.. autoclass:: popart.DeviceManager
.. autoclass:: popart.DeviceProvider
.. autoclass:: popart.popx.DevicexManager
.. autoclass:: popart.popx.Devicex
.. autoclass:: popart.popx.PopStreamId
.. autoclass:: popart.popx.Executablex
.. autoclass:: popart.popx.IrLowering
.. autoclass:: popart.popx.PopTensors
.. autoclass:: popart.popx.PopPrograms
.. autoclass:: popart.popx.PopOpx
.. autoclass:: popart.popx.ICreatorCandidatePtr
.. autoclass:: popart.POpCmp
.. autoclass:: popart.popx.InputCreatorType
.. autoclass:: popart.popx.ICreatorCandidate
.. autoclass:: popart.popx.ReplicatedTensorShardingBundle
.. autoclass:: popart.popx.LinearMapper

Ops
---

Op definition for PopART IR
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: popart.Op
.. autoclass:: popart.GradInOutMapper
.. autoclass:: popart.RecomputeType
.. autoclass:: popart.ExecutionContext
.. autoclass:: popart.GradOpInType
.. autoclass:: popart.OperatorIdentifier
.. autoclass:: popart.NumInputs
.. autoclass:: popart.VGraphIdAndTileSet
.. autoclass:: popart.OptionalTensorLocation
.. autoclass:: popart.OptionalVGraphId
.. autoclass:: popart.OptionalPipelineStage
.. autoclass:: popart.OptionalExecutionPhase
.. autoclass:: popart.OptionalBatchSerializedPhase
.. autoclass:: popart.OptionalStochasticRoundingMethod
.. autoclass:: popart.OptionalDataType
.. autoclass:: popart.OpDefinition
.. autoclass:: popart.OpCreatorInfo
.. autoclass:: popart.OpManager
.. autoclass:: popart.VarUpdateOp
.. autoclass:: popart.AccumulatorScaleOp
.. autoclass:: popart.AccumulatorZeroOp
.. autoclass:: popart.VarUpdateWithUpdaterOp
.. autoclass:: popart.AccumulateBaseOp
.. autoclass:: popart.AccumulateOp
.. autoclass:: popart.RescaleAccumulateOp
.. autoclass:: popart.SparseAccumulateOp
.. autoclass:: popart.AdamComboOp
.. autoclass:: popart.AdamVarUpdateOp
.. autoclass:: popart.AdaptiveComboOp
.. autoclass:: popart.CopyVarUpdateOp
.. autoclass:: popart.SGD0ComboOp
.. autoclass:: popart.SGD0VarUpdateOpBase
.. autoclass:: popart.SGD0VarUpdateOp
.. autoclass:: popart.SGD1AcclUpdateOp
.. autoclass:: popart.SGD2PartialAcclUpdateOp
.. autoclass:: popart.SGD1VarUpdateOp
.. autoclass:: popart.SGD2VarUpdateOp
.. autoclass:: popart.SGDMComboBaseOp
.. autoclass:: popart.SGD1ComboOp
.. autoclass:: popart.SGD2ComboOp
.. autoclass:: popart.ScaledVarUpdateOp
.. autoclass:: popart.AliasModel
.. autoclass:: popart.IpuCopyOp
.. autoclass:: popart.SourceIpuMap
.. autoclass:: popart.SourceTensorMap

Op definition for Poplar implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: popart.popx.Opx
.. autoclass:: popart.popx.RoiAlignGradOpx
.. autoclass:: popart.popx.RoiAlignOpx

Available Ops (Op class)
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: popart.AiGraphcoreOpIdV1
.. autoclass:: popart.AbortOp
.. autoclass:: popart.AbsGradOp
.. autoclass:: popart.AbsOp
.. autoclass:: popart.AdaDeltaUpdaterOp
.. autoclass:: popart.AdamUpdaterOp
.. autoclass:: popart.AddArg0GradOp
.. autoclass:: popart.AddArg1GradOp
.. autoclass:: popart.AddBiasBiasGradOp
.. autoclass:: popart.AddBiasDataGradOp
.. autoclass:: popart.AddBiasInplaceOp
.. autoclass:: popart.AddBiasOp
.. autoclass:: popart.AddLhsInplaceOp
.. autoclass:: popart.AddRhsInplaceOp
.. autoclass:: popart.AllReduceGradOp
.. autoclass:: popart.AllReduceOp
.. autoclass:: popart.AndOp
.. autoclass:: popart.ArgExtremaOp
.. autoclass:: popart.ArgMaxOp
.. autoclass:: popart.ArgMinOp
.. autoclass:: popart.AsinGradOp
.. autoclass:: popart.AsinInplaceOp
.. autoclass:: popart.AsinOp
.. autoclass:: popart.Atan2Arg0GradOp
.. autoclass:: popart.Atan2Arg1GradOp
.. autoclass:: popart.Atan2LhsInplaceOp
.. autoclass:: popart.AtanGradOp
.. autoclass:: popart.AtanInplaceOp
.. autoclass:: popart.AtanOp
.. autoclass:: popart.AutoLossScaleProxyGradOp
.. autoclass:: popart.AutoLossScaleProxyOp
.. autoclass:: popart.AveragePoolGradOp
.. autoclass:: popart.AveragePoolOp
.. autoclass:: popart.BaseOnnxRNNGradOp
.. autoclass:: popart.BaseOnnxRNNOp
.. autoclass:: popart.BasePadOp
.. autoclass:: popart.BasePadOutplaceOp
.. autoclass:: popart.BaseSliceOp
.. autoclass:: popart.BaseSortOp
.. autoclass:: popart.BatchNormGradOp
.. autoclass:: popart.BatchNormOp
.. autoclass:: popart.BinaryComparisonOp
.. autoclass:: popart.BinaryConstScalarOp
.. autoclass:: popart.BitwiseBinaryOp
.. autoclass:: popart.BitwiseNotOp
.. autoclass:: popart.BoundaryOp
.. autoclass:: popart.CallGradOp
.. autoclass:: popart.CallOp
.. autoclass:: popart.CastGradOp
.. autoclass:: popart.CastOp
.. autoclass:: popart.CeilInplaceOp
.. autoclass:: popart.CeilOp
.. autoclass:: popart.ClipGradOp
.. autoclass:: popart.ClipInplaceOp
.. autoclass:: popart.ClipOp
.. autoclass:: popart.CollectivesBaseOp
.. autoclass:: popart.ConcatGradOp
.. autoclass:: popart.ConcatInplaceOp
.. autoclass:: popart.ConcatOp
.. autoclass:: popart.ConvDataGradOp
.. autoclass:: popart.ConvFlipWeightsGradOp
.. autoclass:: popart.ConvFlipWeightsOp
.. autoclass:: popart.ConvOp
.. autoclass:: popart.ConvTransposeOp
.. autoclass:: popart.ConvWeightsGradOp
.. autoclass:: popart.CosGradOp
.. autoclass:: popart.CosOp
.. autoclass:: popart.CoshOp
.. autoclass:: popart.CtcBeamSearchDecoderOp
.. autoclass:: popart.CtcGradOp
.. autoclass:: popart.CtcOp
.. autoclass:: popart.CumSumGradOp
.. autoclass:: popart.CumSumOp
.. autoclass:: popart.DetachInplaceOp
.. autoclass:: popart.DetachOp
.. autoclass:: popart.DivArg0GradOp
.. autoclass:: popart.DivArg1GradOp
.. autoclass:: popart.DropoutBaseOp
.. autoclass:: popart.DropoutOp
.. autoclass:: popart.DropoutGradOp
.. autoclass:: popart.DynamicAddInplaceOp
.. autoclass:: popart.DynamicAddOp
.. autoclass:: popart.DynamicBaseOp
.. autoclass:: popart.DynamicBinaryBaseInplaceOp
.. autoclass:: popart.DynamicBinaryBaseOp
.. autoclass:: popart.DynamicSliceBaseOp
.. autoclass:: popart.DynamicSliceInplaceOp
.. autoclass:: popart.DynamicSliceOp
.. autoclass:: popart.DynamicSlicePadGradOp
.. autoclass:: popart.DynamicTernaryBaseInplaceOp
.. autoclass:: popart.DynamicTernaryBaseOp
.. autoclass:: popart.DynamicUpdateInplaceOp
.. autoclass:: popart.DynamicUpdateOp
.. autoclass:: popart.DynamicUpdateToUpdateGradOp
.. autoclass:: popart.DynamicUpdateUpdaterGradOp
.. autoclass:: popart.DynamicZeroGradOp
.. autoclass:: popart.DynamicZeroInplaceOp
.. autoclass:: popart.DynamicZeroOp
.. autoclass:: popart.ElementWiseBinaryArg0GradOp
.. autoclass:: popart.ElementWiseBinaryArg1GradOp
.. autoclass:: popart.ElementWiseBinaryBaseOp
.. autoclass:: popart.ElementWiseBinaryGradOp
.. autoclass:: popart.ElementWiseBinaryInplaceLhsOp
.. autoclass:: popart.ElementWiseBinaryInplaceRhsOp
.. autoclass:: popart.ElementWiseBinaryOp
.. autoclass:: popart.ElementWiseInplaceUnaryOp
.. autoclass:: popart.ElementWiseNonLinearUnaryGradOp
.. autoclass:: popart.ElementWiseNpBroadcastableBinaryWithGradOp
.. autoclass:: popart.ElementWiseUnaryBooleanOp
.. autoclass:: popart.ElementWiseUnaryOp
.. autoclass:: popart.EluGradOp
.. autoclass:: popart.EluInplaceOp
.. autoclass:: popart.EluOp
.. autoclass:: popart.EqualOp
.. autoclass:: popart.ErfGradOp
.. autoclass:: popart.ErfOp
.. autoclass:: popart.ExchangeBaseOp
.. autoclass:: popart.ExpGradOp
.. autoclass:: popart.ExpInplaceOp
.. autoclass:: popart.ExpOp
.. autoclass:: popart.ExpandGradOp
.. autoclass:: popart.ExpandInplaceOp
.. autoclass:: popart.ExpandOp
.. autoclass:: popart.Expm1GradOp
.. autoclass:: popart.Expm1InplaceOp
.. autoclass:: popart.Expm1Op
.. autoclass:: popart.FloorInplaceOp
.. autoclass:: popart.FloorOp
.. autoclass:: popart.FmodArg0GradOp
.. autoclass:: popart.FmodOp
.. autoclass:: popart.GRUGradOp
.. autoclass:: popart.GRUOp
.. autoclass:: popart.GatherGradOp
.. autoclass:: popart.GatherOp
.. autoclass:: popart.GeluGradOp
.. autoclass:: popart.GeluInplaceOp
.. autoclass:: popart.GeluOp
.. autoclass:: popart.GetRandomSeedOp
.. autoclass:: popart.GlobalAveragePoolGradOp
.. autoclass:: popart.GlobalAveragePoolOp
.. autoclass:: popart.GlobalMaxPoolGradOp
.. autoclass:: popart.GlobalMaxPoolOp
.. autoclass:: popart.GreaterOp
.. autoclass:: popart.GroupNormGradOp
.. autoclass:: popart.GroupNormOp
.. autoclass:: popart.HardSigmoidGradOp
.. autoclass:: popart.HardSigmoidInplaceOp
.. autoclass:: popart.HardSigmoidOp
.. autoclass:: popart.HasReceptiveFieldOp
.. autoclass:: popart.HistogramOp
.. autoclass:: popart.HostBaseOp
.. autoclass:: popart.HostLoadInplaceOp
.. autoclass:: popart.HostLoadOp
.. autoclass:: popart.HostStoreOp
.. autoclass:: popart.IdentityGradOp
.. autoclass:: popart.IdentityInplaceOp
.. autoclass:: popart.IdentityLossGradOp
.. autoclass:: popart.IdentityLossOp
.. autoclass:: popart.IdentityOp
.. autoclass:: popart.IfConditionGradOp
.. autoclass:: popart.IfGradOp
.. autoclass:: popart.IfOp
.. autoclass:: popart.IncrementModInplaceOp
.. autoclass:: popart.IncrementModOp
.. autoclass:: popart.InitOp
.. autoclass:: popart.InstanceNormGradOp
.. autoclass:: popart.InstanceNormOp
.. autoclass:: popart.IoTileCopyOp
.. autoclass:: popart.IsInf
.. autoclass:: popart.IsNaN
.. autoclass:: popart.L1GradOp
.. autoclass:: popart.L1Op
.. autoclass:: popart.LRNGradOp
.. autoclass:: popart.LRNOp
.. autoclass:: popart.LSTMGradOp
.. autoclass:: popart.LSTMOp
.. autoclass:: popart.LambSquareOp
.. autoclass:: popart.LeakyReluGradOp
.. autoclass:: popart.LeakyReluInplaceOp
.. autoclass:: popart.LeakyReluOp
.. autoclass:: popart.LessOp
.. autoclass:: popart.LinearVariadicGradOp
.. autoclass:: popart.Log1pGradOp
.. autoclass:: popart.Log1pInplaceOp
.. autoclass:: popart.Log1pOp
.. autoclass:: popart.LogGradOp
.. autoclass:: popart.LogOp
.. autoclass:: popart.LogSoftmaxGradOp
.. autoclass:: popart.LogSoftmaxInplaceOp
.. autoclass:: popart.LogSoftmaxOp
.. autoclass:: popart.LoopOp
.. autoclass:: popart.LossOp
.. autoclass:: popart.LossScaleUpdateOp
.. autoclass:: popart.MatMulBaseGradOp
.. autoclass:: popart.MatMulBaseOp
.. autoclass:: popart.MatMulLhsGradOp
.. autoclass:: popart.MatMulOp
.. autoclass:: popart.MatMulRhsGradOp
.. autoclass:: popart.MaxArgGradOp
.. autoclass:: popart.MaxOp
.. autoclass:: popart.MaxPoolGradOp
.. autoclass:: popart.MaxPoolOp
.. autoclass:: popart.MeanArgGradOp
.. autoclass:: popart.MeanOp
.. autoclass:: popart.MinArgGradOp
.. autoclass:: popart.MinOp
.. autoclass:: popart.ModifyRandomSeedOp
.. autoclass:: popart.MulArg0GradOp
.. autoclass:: popart.MulArg1GradOp
.. autoclass:: popart.MulLhsInplaceOp
.. autoclass:: popart.MulRhsInplaceOp
.. autoclass:: popart.MultiCollectiveBaseOp
.. autoclass:: popart.MultiConvBaseOp
.. autoclass:: popart.MultiConvDataGradBaseOp
.. autoclass:: popart.MultiConvDataGradOp
.. autoclass:: popart.MultiConvOp
.. autoclass:: popart.MultiConvWeightsGradBaseOp
.. autoclass:: popart.MultiConvWeightsGradOp
.. autoclass:: popart.MultiExchangeOp
.. autoclass:: popart.MultiReplicatedAllReduceOp
.. autoclass:: popart.NegateGradOp
.. autoclass:: popart.NegateOp
.. autoclass:: popart.NllGradOp
.. autoclass:: popart.NllOp
.. autoclass:: popart.NlllWithSoftmaxGradDirectOp
.. autoclass:: popart.NonLinearVariadicGradOp
.. autoclass:: popart.NopOp
.. autoclass:: popart.NotOp
.. autoclass:: popart.OneWayUnaryInPlaceOp
.. autoclass:: popart.OneWayUnaryOp
.. autoclass:: popart.OnehotGradOp
.. autoclass:: popart.OnehotOp
.. autoclass:: popart.OrOp
.. autoclass:: popart.PReluOp
.. autoclass:: popart.PackedDataBlockOp
.. autoclass:: popart.PadGradOp
.. autoclass:: popart.PadInplaceOp
.. autoclass:: popart.PadOp
.. autoclass:: popart.ParameterizedOp
.. autoclass:: popart.PlaceholderOp
.. autoclass:: popart.PopartLSTMGradOp
.. autoclass:: popart.PopartLSTMOp
.. autoclass:: popart.PowArg0GradOp
.. autoclass:: popart.PowArg1GradOp
.. autoclass:: popart.PowLhsInplaceOp
.. autoclass:: popart.PrintTensorOp
.. autoclass:: popart.RMSPropUpdaterOp
.. autoclass:: popart.RNNGradOp
.. autoclass:: popart.RNNOp
.. autoclass:: popart.RandomBaseOp
.. autoclass:: popart.RandomNormalBaseOp
.. autoclass:: popart.RandomNormalLikeOp
.. autoclass:: popart.RandomNormalOp
.. autoclass:: popart.RandomUniformBaseOp
.. autoclass:: popart.RandomUniformLikeOp
.. autoclass:: popart.RandomUniformOp
.. autoclass:: popart.ReciprocalGradOp
.. autoclass:: popart.ReciprocalOp
.. autoclass:: popart.ReduceGradOp
.. autoclass:: popart.ReduceL1GradOp
.. autoclass:: popart.ReduceL1Op
.. autoclass:: popart.ReduceL2GradOp
.. autoclass:: popart.ReduceL2Op
.. autoclass:: popart.ReduceLogSumExpGradOp
.. autoclass:: popart.ReduceLogSumExpOp
.. autoclass:: popart.ReduceLogSumGradOp
.. autoclass:: popart.ReduceLogSumOp
.. autoclass:: popart.ReduceMaxGradOp
.. autoclass:: popart.ReduceMaxOp
.. autoclass:: popart.ReduceMeanGradOp
.. autoclass:: popart.ReduceMeanOp
.. autoclass:: popart.ReduceMedianGradOp
.. autoclass:: popart.ReduceMedianOp
.. autoclass:: popart.ReduceMinGradOp
.. autoclass:: popart.ReduceMinOp
.. autoclass:: popart.ReduceOp
.. autoclass:: popart.ReduceProdGradOp
.. autoclass:: popart.ReduceProdOp
.. autoclass:: popart.ReduceSumGradOp
.. autoclass:: popart.ReduceSumOp
.. autoclass:: popart.ReduceSumSquareGradOp
.. autoclass:: popart.ReduceSumSquareOp
.. autoclass:: popart.ReluGradOp
.. autoclass:: popart.ReluInplaceOp
.. autoclass:: popart.ReluOp
.. autoclass:: popart.RemoteBaseOp
.. autoclass:: popart.RemoteLoadInplaceOp
.. autoclass:: popart.RemoteLoadOp
.. autoclass:: popart.RemoteStoreOp
.. autoclass:: popart.ReplicatedAllGatherOp
.. autoclass:: popart.ReplicatedAllReduceInplaceOp
.. autoclass:: popart.ReplicatedAllReduceOp
.. autoclass:: popart.ReplicatedReduceScatterOp
.. autoclass:: popart.ReshapeBaseOp
.. autoclass:: popart.ReshapeGradOp
.. autoclass:: popart.ReshapeInplaceOp
.. autoclass:: popart.ReshapeOp
.. autoclass:: popart.ResizeGradOp
.. autoclass:: popart.ResizeOp
.. autoclass:: popart.RestoreInplaceOp
.. autoclass:: popart.RestoreOp
.. autoclass:: popart.ReverseBaseOp
.. autoclass:: popart.ReverseGradOp
.. autoclass:: popart.ReverseInplaceOp
.. autoclass:: popart.ReverseOp
.. autoclass:: popart.RoiAlignGradOp
.. autoclass:: popart.RoiAlignOp
.. autoclass:: popart.RoundInplaceOp
.. autoclass:: popart.RoundOp
.. autoclass:: popart.ScaleGradOp
.. autoclass:: popart.ScaleInplaceOp
.. autoclass:: popart.ScaleOp
.. autoclass:: popart.ScaledAddLhsInplaceOp
.. autoclass:: popart.ScaledAddOp
.. autoclass:: popart.ScaledAddRhsInplaceOp
.. autoclass:: popart.ScanOp
.. autoclass:: popart.ScatterDataGradOp
.. autoclass:: popart.ScatterOp
.. autoclass:: popart.ScatterReduceGradOp
.. autoclass:: popart.ScatterReduceOp
.. autoclass:: popart.ScatterUpdateGradOp
.. autoclass:: popart.SeluGradOp
.. autoclass:: popart.SeluInplaceOp
.. autoclass:: popart.SeluOp
.. autoclass:: popart.SequenceSliceInplaceOp
.. autoclass:: popart.SequenceSliceOp
.. autoclass:: popart.ShapeOrLikeOp
.. autoclass:: popart.ShapedDropoutOp
.. autoclass:: popart.ShapedDropoutGradOp
.. autoclass:: popart.ShrinkGradOp
.. autoclass:: popart.ShrinkInplaceOp
.. autoclass:: popart.ShrinkOp
.. autoclass:: popart.SigmoidGradOp
.. autoclass:: popart.SigmoidInplaceOp
.. autoclass:: popart.SigmoidOp
.. autoclass:: popart.SignInplaceOp
.. autoclass:: popart.SignOp
.. autoclass:: popart.SinGradOp
.. autoclass:: popart.SinOp
.. autoclass:: popart.SinhGradOp
.. autoclass:: popart.SinhInplaceOp
.. autoclass:: popart.SinhOp
.. autoclass:: popart.SliceGradOp
.. autoclass:: popart.SliceInplaceOp
.. autoclass:: popart.SliceOp
.. autoclass:: popart.SoftPlusGradOp
.. autoclass:: popart.SoftPlusInplaceOp
.. autoclass:: popart.SoftPlusOp
.. autoclass:: popart.SoftSignGradOp
.. autoclass:: popart.SoftSignInplaceOp
.. autoclass:: popart.SoftSignOp
.. autoclass:: popart.SoftmaxGradDirectOp
.. autoclass:: popart.SoftmaxGradOp
.. autoclass:: popart.SoftmaxInplaceOp
.. autoclass:: popart.SoftmaxOp
.. autoclass:: popart.SplitGradOp
.. autoclass:: popart.SplitOp
.. autoclass:: popart.SqrtGradOp
.. autoclass:: popart.SqrtOp
.. autoclass:: popart.SquareOp
.. autoclass:: popart.StashOp
.. autoclass:: popart.SubgraphOp
.. autoclass:: popart.SubsampleBaseOp
.. autoclass:: popart.SubsampleGradOp
.. autoclass:: popart.SubsampleInplaceOp
.. autoclass:: popart.SubsampleOp
.. autoclass:: popart.SubtractArg0GradOp
.. autoclass:: popart.SubtractArg1GradOp
.. autoclass:: popart.SumArgGradOp
.. autoclass:: popart.SumOp
.. autoclass:: popart.SwishGradOp
.. autoclass:: popart.SwishInplaceOp
.. autoclass:: popart.SwishOp
.. autoclass:: popart.SyncOp
.. autoclass:: popart.TanhGradOp
.. autoclass:: popart.TanhOp
.. autoclass:: popart.TensorRemapOp
.. autoclass:: popart.ThresholdedReluGradOp
.. autoclass:: popart.ThresholdedReluInplaceOp
.. autoclass:: popart.ThresholdedReluOp
.. autoclass:: popart.TiedGatherGradOp
.. autoclass:: popart.TiedGatherOp
.. autoclass:: popart.TileGradOp
.. autoclass:: popart.TileOp
.. autoclass:: popart.TopKGradOp
.. autoclass:: popart.TopKOp
.. autoclass:: popart.TransposeBaseOp
.. autoclass:: popart.TransposeGradOp
.. autoclass:: popart.TransposeInplaceOp
.. autoclass:: popart.TransposeOp
.. autoclass:: popart.UnaryZeroGradOp
.. autoclass:: popart.UpsampleOp
.. autoclass:: popart.VariadicGradOp
.. autoclass:: popart.VariadicOp
.. autoclass:: popart.WhereLhsInplaceOp
.. autoclass:: popart.WhereOp
.. autoclass:: popart.WhereRhsInplaceOp
.. autoclass:: popart.WhereXGradOp
.. autoclass:: popart.WhereYGradOp
.. autoclass:: popart.ZerosBaseOp
.. autoclass:: popart.ZerosLikeOp
.. autoclass:: popart.ZerosOp

Available Ops (Opx class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: popart.popx.AbortOpx
.. autoclass:: popart.popx.AbsOpx
.. autoclass:: popart.popx.AccumulateBaseOpx
.. autoclass:: popart.popx.AccumulateOpx
.. autoclass:: popart.popx.AccumulatorScaleOpx
.. autoclass:: popart.popx.AdaDeltaUpdaterOpx
.. autoclass:: popart.popx.AdamUpdaterOpx
.. autoclass:: popart.popx.AdamVarUpdateOpx
.. autoclass:: popart.popx.AddArg0GradOpx
.. autoclass:: popart.popx.AddArg1GradOpx
.. autoclass:: popart.popx.AddBiasBiasGradOpx
.. autoclass:: popart.popx.AddBiasDataGradOpx
.. autoclass:: popart.popx.AddBiasInplaceOpx
.. autoclass:: popart.popx.AddBiasOpx
.. autoclass:: popart.popx.AddLhsInplaceOpx
.. autoclass:: popart.popx.AddOpx
.. autoclass:: popart.popx.AddRhsInplaceOpx
.. autoclass:: popart.popx.AllReduceOpx
.. autoclass:: popart.popx.AndOpx
.. autoclass:: popart.popx.ArgExtremaOpx
.. autoclass:: popart.popx.ArgMaxOpx
.. autoclass:: popart.popx.ArgMinOpx
.. autoclass:: popart.popx.AsinGradOpx
.. autoclass:: popart.popx.AsinInplaceOpx
.. autoclass:: popart.popx.AsinOpx
.. autoclass:: popart.popx.Atan2LhsInplaceOpx
.. autoclass:: popart.popx.Atan2Opx
.. autoclass:: popart.popx.AtanGradOpx
.. autoclass:: popart.popx.AtanInplaceOpx
.. autoclass:: popart.popx.AtanOpx
.. autoclass:: popart.popx.BaseConcatOpx
.. autoclass:: popart.popx.BaseExpandOpx
.. autoclass:: popart.popx.BasePadOpx
.. autoclass:: popart.popx.BaseSliceOpx
.. autoclass:: popart.popx.BaseSortOpx
.. autoclass:: popart.popx.BaseWhereOpx
.. autoclass:: popart.popx.BatchNormGradOpx
.. autoclass:: popart.popx.BatchNormOpx
.. autoclass:: popart.popx.BinaryComparisonOpx
.. autoclass:: popart.popx.BitwiseBinaryOpx
.. autoclass:: popart.popx.BitwiseNotOpx
.. autoclass:: popart.popx.CallGradOpx
.. autoclass:: popart.popx.CallOpx
.. autoclass:: popart.popx.CastGradOpx
.. autoclass:: popart.popx.CastOpx
.. autoclass:: popart.popx.CeilInplaceOpx
.. autoclass:: popart.popx.CeilOpx
.. autoclass:: popart.popx.ClipGradOpx
.. autoclass:: popart.popx.ClipInplaceOpx
.. autoclass:: popart.popx.ClipOpx
.. autoclass:: popart.popx.CollectivesBaseOpx
.. autoclass:: popart.popx.ConcatGradOpx
.. autoclass:: popart.popx.ConcatInplaceOpx
.. autoclass:: popart.popx.ConcatOpx
.. autoclass:: popart.popx.ConvFlipWeightsGradOpx
.. autoclass:: popart.popx.ConvOpx
.. autoclass:: popart.popx.ConvWeightsGradOpx
.. autoclass:: popart.popx.CopyVarUpdateOpx
.. autoclass:: popart.popx.CosOpx
.. autoclass:: popart.popx.CtcBeamSearchDecoderOpx
.. autoclass:: popart.popx.CtcGradOpx
.. autoclass:: popart.popx.CtcOpx
.. autoclass:: popart.popx.CumSumGradOpx
.. autoclass:: popart.popx.CumSumOpx
.. autoclass:: popart.popx.DetachInplaceOpx
.. autoclass:: popart.popx.DetachOpx
.. autoclass:: popart.popx.DivOpx
.. autoclass:: popart.popx.DropoutOpx
.. autoclass:: popart.popx.DynamicAddInplaceOpx
.. autoclass:: popart.popx.DynamicAddOpx
.. autoclass:: popart.popx.DynamicSliceInplaceOpx
.. autoclass:: popart.popx.DynamicSliceOpx
.. autoclass:: popart.popx.DynamicUpdateInplaceOpx
.. autoclass:: popart.popx.DynamicUpdateOpx
.. autoclass:: popart.popx.DynamicZeroInplaceOpx
.. autoclass:: popart.popx.DynamicZeroOpx
.. autoclass:: popart.popx.ElementWiseBinaryInplaceOpx
.. autoclass:: popart.popx.ElementWiseBinaryOpx
.. autoclass:: popart.popx.ElementWiseBinaryOutplaceOpx
.. autoclass:: popart.popx.ElementWiseUnaryInplaceOpx
.. autoclass:: popart.popx.ElementWiseUnaryOpx
.. autoclass:: popart.popx.ElementWiseUnaryOutplaceOpx
.. autoclass:: popart.popx.EluGradOpx
.. autoclass:: popart.popx.EluInplaceOpx
.. autoclass:: popart.popx.EluOpx
.. autoclass:: popart.popx.EqualOpx
.. autoclass:: popart.popx.ErfxGradOpx
.. autoclass:: popart.popx.ErfxOpx
.. autoclass:: popart.popx.ExchangeBaseOpx
.. autoclass:: popart.popx.ExpInplaceOpx
.. autoclass:: popart.popx.ExpOpx
.. autoclass:: popart.popx.ExpandGradOpx
.. autoclass:: popart.popx.ExpandInplaceOpx
.. autoclass:: popart.popx.ExpandOpx
.. autoclass:: popart.popx.Expm1InplaceOpx
.. autoclass:: popart.popx.Expm1Opx
.. autoclass:: popart.popx.FloorInplaceOpx
.. autoclass:: popart.popx.FloorOpx
.. autoclass:: popart.popx.FmodOpx
.. autoclass:: popart.popx.GRUGradOpx
.. autoclass:: popart.popx.GRUOpx
.. autoclass:: popart.popx.GatherBaseOpx
.. autoclass:: popart.popx.GatherGradOpx
.. autoclass:: popart.popx.GatherOpx
.. autoclass:: popart.popx.GeluGradOpx
.. autoclass:: popart.popx.GeluInplaceOpx
.. autoclass:: popart.popx.GeluOpx
.. autoclass:: popart.popx.GetRandomSeedOpx
.. autoclass:: popart.popx.GreaterOpx
.. autoclass:: popart.popx.GroupNormGradOpx
.. autoclass:: popart.popx.GroupNormOpx
.. autoclass:: popart.popx.HardSigmoidGradOpx
.. autoclass:: popart.popx.HardSigmoidInplaceOpx
.. autoclass:: popart.popx.HardSigmoidOpx
.. autoclass:: popart.popx.HistogramOpx
.. autoclass:: popart.popx.HostBaseOpx
.. autoclass:: popart.popx.HostLoadInplaceOpx
.. autoclass:: popart.popx.HostLoadOpx
.. autoclass:: popart.popx.HostStoreOpx
.. autoclass:: popart.popx.IdentityGradOpx
.. autoclass:: popart.popx.IdentityInplaceOpx
.. autoclass:: popart.popx.IdentityLossGradOpx
.. autoclass:: popart.popx.IdentityLossOpx
.. autoclass:: popart.popx.IdentityOpx
.. autoclass:: popart.popx.IfGradOpx
.. autoclass:: popart.popx.IfOpx
.. autoclass:: popart.popx.IncrementModInplaceOpx
.. autoclass:: popart.popx.IncrementModOpx
.. autoclass:: popart.popx.InitOpx
.. autoclass:: popart.popx.InstanceNormGradOpx
.. autoclass:: popart.popx.InstanceNormOpx
.. autoclass:: popart.popx.IoTileCopyOpx
.. autoclass:: popart.popx.IpuCopyOpx
.. autoclass:: popart.popx.L1GradOpx
.. autoclass:: popart.popx.L1Opx
.. autoclass:: popart.popx.LRNGradOpx
.. autoclass:: popart.popx.LRNOpx
.. autoclass:: popart.popx.LSTMGradOpx
.. autoclass:: popart.popx.LSTMOpx
.. autoclass:: popart.popx.LambSquareOpx
.. autoclass:: popart.popx.LeakyReluGradOpx
.. autoclass:: popart.popx.LeakyReluInplaceOpx
.. autoclass:: popart.popx.LeakyReluOpx
.. autoclass:: popart.popx.LessOpx
.. autoclass:: popart.popx.Log1pInplaceOpx
.. autoclass:: popart.popx.Log1pOpx
.. autoclass:: popart.popx.LogOpx
.. autoclass:: popart.popx.LogSoftmaxGradOpx
.. autoclass:: popart.popx.LogSoftmaxInplaceOpx
.. autoclass:: popart.popx.LogSoftmaxOpx
.. autoclass:: popart.popx.LoopOpx
.. autoclass:: popart.popx.LossScaleUpdateOpx
.. autoclass:: popart.popx.MatMulOpx
.. autoclass:: popart.popx.MaxArgGradOpx
.. autoclass:: popart.popx.MaxOpx
.. autoclass:: popart.popx.MeanArgGradOpx
.. autoclass:: popart.popx.MeanOpx
.. autoclass:: popart.popx.MinArgGradOpx
.. autoclass:: popart.popx.MinOpx
.. autoclass:: popart.popx.ModifyRandomSeedOpx
.. autoclass:: popart.popx.MulLhsInplaceOpx
.. autoclass:: popart.popx.MulOpx
.. autoclass:: popart.popx.MulRhsInplaceOpx
.. autoclass:: popart.popx.MultiCollectiveBaseOpx
.. autoclass:: popart.popx.MultiConvBaseOpx
.. autoclass:: popart.popx.MultiConvOpx
.. autoclass:: popart.popx.MultiConvWeightsGradBaseOpx
.. autoclass:: popart.popx.MultiConvWeightsGradOpx
.. autoclass:: popart.popx.MultiExchangeOpx
.. autoclass:: popart.popx.MultiReplicatedAllReduceOpx
.. autoclass:: popart.popx.NegateGradOpx
.. autoclass:: popart.popx.NegateOpx
.. autoclass:: popart.popx.NllGradOpx
.. autoclass:: popart.popx.NllOpx
.. autoclass:: popart.popx.NlllWithSoftmaxGradDirectOpx
.. autoclass:: popart.popx.NopOpx
.. autoclass:: popart.popx.NormOpx
.. autoclass:: popart.popx.NotOpx
.. autoclass:: popart.popx.OnehotGradOpx
.. autoclass:: popart.popx.OnehotOpx
.. autoclass:: popart.popx.OrOpx
.. autoclass:: popart.popx.PReluOpx
.. autoclass:: popart.popx.PadGradOpx
.. autoclass:: popart.popx.PadInplaceOpx
.. autoclass:: popart.popx.PadOpx
.. autoclass:: popart.popx.PopartLSTMOpxBase
.. autoclass:: popart.popx.PowLhsInplaceOpx
.. autoclass:: popart.popx.PowOpx
.. autoclass:: popart.popx.PrintTensorOpx
.. autoclass:: popart.popx.RMSPropUpdaterOpx
.. autoclass:: popart.popx.RNNGradOpx
.. autoclass:: popart.popx.RNNOpx
.. autoclass:: popart.popx.RandomNormalOpx
.. autoclass:: popart.popx.RandomUniformOpx
.. autoclass:: popart.popx.ReciprocalOpx
.. autoclass:: popart.popx.ReduceL1GradOpx
.. autoclass:: popart.popx.ReduceL1Opx
.. autoclass:: popart.popx.ReduceL2GradOpx
.. autoclass:: popart.popx.ReduceL2Opx
.. autoclass:: popart.popx.ReduceLogSumExpGradOpx
.. autoclass:: popart.popx.ReduceLogSumExpOpx
.. autoclass:: popart.popx.ReduceLogSumGradOpx
.. autoclass:: popart.popx.ReduceLogSumOpx
.. autoclass:: popart.popx.ReduceMaxGradOpx
.. autoclass:: popart.popx.ReduceMaxOpx
.. autoclass:: popart.popx.ReduceMeanGradOpx
.. autoclass:: popart.popx.ReduceMeanOpx
.. autoclass:: popart.popx.ReduceMedianGradOpx
.. autoclass:: popart.popx.ReduceMedianOpx
.. autoclass:: popart.popx.ReduceMinGradOpx
.. autoclass:: popart.popx.ReduceMinOpx
.. autoclass:: popart.popx.ReduceProdGradOpx
.. autoclass:: popart.popx.ReduceProdOpx
.. autoclass:: popart.popx.ReduceSumGradOpx
.. autoclass:: popart.popx.ReduceSumOpx
.. autoclass:: popart.popx.ReduceSumSquareGradOpx
.. autoclass:: popart.popx.ReduceSumSquareOpx
.. autoclass:: popart.popx.ReluGradOpx
.. autoclass:: popart.popx.ReluInplaceOpx
.. autoclass:: popart.popx.ReluOpx
.. autoclass:: popart.popx.RemoteBaseOpx
.. autoclass:: popart.popx.RemoteLoadInplaceOpx
.. autoclass:: popart.popx.RemoteLoadOpx
.. autoclass:: popart.popx.RemoteStoreOpx
.. autoclass:: popart.popx.ReplicatedAllGatherOpx
.. autoclass:: popart.popx.ReplicatedAllReduceInplaceOpx
.. autoclass:: popart.popx.ReplicatedAllReduceOpx
.. autoclass:: popart.popx.ReplicatedReduceScatterOpx
.. autoclass:: popart.popx.RescaleAccumulateOpx
.. autoclass:: popart.popx.ReshapeBaseOpx
.. autoclass:: popart.popx.ReshapeGradOpx
.. autoclass:: popart.popx.ReshapeInplaceOpx
.. autoclass:: popart.popx.ReshapeOpx
.. autoclass:: popart.popx.ResizeGradOpx
.. autoclass:: popart.popx.ResizeOpx
.. autoclass:: popart.popx.RestoreBaseOpx
.. autoclass:: popart.popx.ReverseBaseOpx
.. autoclass:: popart.popx.ReverseGradOpx
.. autoclass:: popart.popx.ReverseInplaceOpx
.. autoclass:: popart.popx.ReverseOpx
.. autoclass:: popart.popx.RoundInplaceOpx
.. autoclass:: popart.popx.RoundOpx
.. autoclass:: popart.popx.SGD0VarUpdateOpx
.. autoclass:: popart.popx.SGD1AcclUpdateOpx
.. autoclass:: popart.popx.SGD1VarUpdateOpx
.. autoclass:: popart.popx.ScaleInplaceOpx
.. autoclass:: popart.popx.ScaleGradOpx
.. autoclass:: popart.popx.ScaleOpx
.. autoclass:: popart.popx.ScaledAddLhsInplaceOpx
.. autoclass:: popart.popx.ScaledAddOpx
.. autoclass:: popart.popx.ScaledAddRhsInplaceOpx
.. autoclass:: popart.popx.ScaledVarUpdateOpx
.. autoclass:: popart.popx.ScatterDataGradOpx
.. autoclass:: popart.popx.ScatterOpx
.. autoclass:: popart.popx.ScatterReduceGradOpx
.. autoclass:: popart.popx.ScatterReduceOpx
.. autoclass:: popart.popx.ScatterUpdateGradOpx
.. autoclass:: popart.popx.SeluGradOpx
.. autoclass:: popart.popx.SeluInplaceOpx
.. autoclass:: popart.popx.SeluOpx
.. autoclass:: popart.popx.SequenceSliceInplaceOpx
.. autoclass:: popart.popx.SequenceSliceOpx
.. autoclass:: popart.popx.ShapedDropoutOpx
.. autoclass:: popart.popx.ShrinkGradOpx
.. autoclass:: popart.popx.ShrinkInplaceOpx
.. autoclass:: popart.popx.ShrinkOpx
.. autoclass:: popart.popx.SigmoidGradOpx
.. autoclass:: popart.popx.SigmoidInplaceOpx
.. autoclass:: popart.popx.SigmoidOpx
.. autoclass:: popart.popx.SignInplaceOpx
.. autoclass:: popart.popx.SignOpx
.. autoclass:: popart.popx.SinOpx
.. autoclass:: popart.popx.SinhGradOpx
.. autoclass:: popart.popx.SinhInplaceOpx
.. autoclass:: popart.popx.SinhOpx
.. autoclass:: popart.popx.SliceInplaceOpx
.. autoclass:: popart.popx.SliceOpx
.. autoclass:: popart.popx.SoftPlusGradOpx
.. autoclass:: popart.popx.SoftPlusInplaceOpx
.. autoclass:: popart.popx.SoftPlusOpx
.. autoclass:: popart.popx.SoftSignGradOpx
.. autoclass:: popart.popx.SoftSignInplaceOpx
.. autoclass:: popart.popx.SoftSignOpx
.. autoclass:: popart.popx.SoftmaxGradDirectOpx
.. autoclass:: popart.popx.SoftmaxGradOpx
.. autoclass:: popart.popx.SoftmaxInplaceOpx
.. autoclass:: popart.popx.SoftmaxOpx
.. autoclass:: popart.popx.SparseAccumulateOpx
.. autoclass:: popart.popx.SplitOpx
.. autoclass:: popart.popx.SqrtOpx
.. autoclass:: popart.popx.SquareOpx
.. autoclass:: popart.popx.StashOpx
.. autoclass:: popart.popx.SubgraphOpx
.. autoclass:: popart.popx.SubsampleGradOpx
.. autoclass:: popart.popx.SubsampleInplaceOpx
.. autoclass:: popart.popx.SubsampleOpx
.. autoclass:: popart.popx.SubtractArg0GradOpx
.. autoclass:: popart.popx.SubtractOpx
.. autoclass:: popart.popx.SumArgGradOpx
.. autoclass:: popart.popx.SumOpx
.. autoclass:: popart.popx.SwishGradOpx
.. autoclass:: popart.popx.SwishInplaceOpx
.. autoclass:: popart.popx.SwishOpx
.. autoclass:: popart.popx.SyncOpx
.. autoclass:: popart.popx.TanhGradOpx
.. autoclass:: popart.popx.TanhOpx
.. autoclass:: popart.popx.TensorRemapOpx
.. autoclass:: popart.popx.ThresholdedReluGradOpx
.. autoclass:: popart.popx.ThresholdedReluInplaceOpx
.. autoclass:: popart.popx.ThresholdedReluOpx
.. autoclass:: popart.popx.TiedGatherOpx
.. autoclass:: popart.popx.TileGradOpx
.. autoclass:: popart.popx.TileOpx
.. autoclass:: popart.popx.TopKGradOpx
.. autoclass:: popart.popx.TopKOpx
.. autoclass:: popart.popx.TransposeGradOpx
.. autoclass:: popart.popx.TransposeInplaceOpx
.. autoclass:: popart.popx.TransposeOpx
.. autoclass:: popart.popx.VarUpdateOpx
.. autoclass:: popart.popx.WhereLhsInplaceOpx
.. autoclass:: popart.popx.WhereOpx
.. autoclass:: popart.popx.WhereRhsInplaceOpx
.. autoclass:: popart.popx.WhereXGradOpx
.. autoclass:: popart.popx.WhereYGradOpx
.. autoclass:: popart.popx.ZerosOpx

Patterns
--------

.. autoclass:: popart.Patterns
    :show-inheritance:

.. autoclass:: popart.PreAliasPattern
    :show-inheritance:

Available patterns
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: popart.AllReduceToIdentityPattern
.. autoclass:: popart.BinaryGradOpPattern
.. autoclass:: popart.ContiguateIpuCopyIndicesPattern
.. autoclass:: popart.ConvDataGradPattern
.. autoclass:: popart.ConvFlipWeightsDoubleFlipPattern
.. autoclass:: popart.ConvFlipWeightsGradOpPattern
.. autoclass:: popart.ConvTransposePattern
.. autoclass:: popart.CosGradOpPattern
.. autoclass:: popart.CoshOpPattern
.. autoclass:: popart.DecomposeBinaryConstScalar
.. autoclass:: popart.ElementWiseGradOpPattern
.. autoclass:: popart.ExpGradOpPattern
.. autoclass:: popart.ExpandCastPattern
.. autoclass:: popart.Expm1GradOpPattern
.. autoclass:: popart.Fuser
.. autoclass:: popart.InitAccumulatePattern
.. autoclass:: popart.LSTMPattern
.. autoclass:: popart.LambSerialisedWeightPattern
.. autoclass:: popart.LikeOpsPattern
.. autoclass:: popart.Log1pGradOpPattern
.. autoclass:: popart.LogGradOpPattern
.. autoclass:: popart.LoopScanOutPattern
.. autoclass:: popart.MatMulGradPattern
.. autoclass:: popart.MatMulPattern
.. autoclass:: popart.MulArgGradOpPattern
.. autoclass:: popart.NlllWithSoftmaxGradDirect
.. autoclass:: popart.OptimizerDecompose
.. autoclass:: popart.PackedDataBlockPattern
.. autoclass:: popart.PadSumPattern
.. autoclass:: popart.PostNRepl
.. autoclass:: popart.PreUniRepl
.. autoclass:: popart.ReciprocalGradOpPattern
.. autoclass:: popart.RemoveUnnecessaryLossGradCast
.. autoclass:: popart.ScanToLoopPattern
.. autoclass:: popart.SequenceExpander
.. autoclass:: popart.SplitGatherPattern
.. autoclass:: popart.SplitOpPattern
.. autoclass:: popart.SqrtGradOpPattern
.. autoclass:: popart.SumToAddPattern
.. autoclass:: popart.TiedGatherAccumulatePattern
.. autoclass:: popart.TiedGatherPattern
.. autoclass:: popart.TransposeToIdentityOrReshapePattern
.. autoclass:: popart.UpsampleToResizePattern
.. autoclass:: popart.ViewSimplifyPattern
.. autoclass:: popart.AdamDecompose
.. autoclass:: popart.AdaptiveDecompose
.. autoclass:: popart.Atan2Arg0GradOpPattern
.. autoclass:: popart.Atan2Arg1GradOpPattern
.. autoclass:: popart.DivArg0GradOpPattern
.. autoclass:: popart.DivArg1GradOpPattern
.. autoclass:: popart.FmodArg0GradOpPattern
.. autoclass:: popart.MatMulLhsGradPattern
.. autoclass:: popart.MatMulRhsGradPattern
.. autoclass:: popart.NegativeOneScalePattern
.. autoclass:: popart.OpToIdentityPattern
.. autoclass:: popart.PowArg0GradOpPattern
.. autoclass:: popart.PowArg1GradOpPattern
.. autoclass:: popart.SGD0Decompose
.. autoclass:: popart.SGD1Decompose
.. autoclass:: popart.SGD2Decompose
.. autoclass:: popart.SoftmaxGradDirect
.. autoclass:: popart.SplitGradOpToConcatPattern
.. autoclass:: popart.SubtractArg1GradOpPattern

Transforms
----------

.. autoclass:: popart.Transform
    :show-inheritance:

Available transforms
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: popart.AccumulateOuterFragmentParallelizer
.. autoclass:: popart.AutoVirtualGraph
.. autoclass:: popart.Autodiff
.. autoclass:: popart.AutomaticLossScale
.. autoclass:: popart.BatchSerialize
.. autoclass:: popart.ClipWeightGradientsByNorm
.. autoclass:: popart.ContiguateCollectivesTransform
.. autoclass:: popart.DecomposeGradSum
.. autoclass:: popart.DecomposeLoops
.. autoclass:: popart.DynamicOpTransform
.. autoclass:: popart.EnsureFp32LossScale
.. autoclass:: popart.ExplicitRecompute
.. autoclass:: popart.HostIOSetup
.. autoclass:: popart.InferPipelineStages
.. autoclass:: popart.InplaceAccumulateGradPartialsIntoOptimizerAccumTensor
.. autoclass:: popart.InterIpuCopy
.. autoclass:: popart.IoComputeTileCopy
.. autoclass:: popart.MainLoops
.. autoclass:: popart.MergeAllVarUpdates
.. autoclass:: popart.MergeAuto
.. autoclass:: popart.MergeLooseThreshold
.. autoclass:: popart.MergeTightThreshold
.. autoclass:: popart.MergeCollectivesTransform
.. autoclass:: popart.MergeCopies
.. autoclass:: popart.MergeDuplicateOps
.. autoclass:: popart.MergeExchange
.. autoclass:: popart.MergeLoops
.. autoclass:: popart.MergeVarUpdates
.. autoclass:: popart.OverlapIO
.. autoclass:: popart.Pipeline
.. autoclass:: popart.PreAutomaticLossScale
.. autoclass:: popart.Prune
.. autoclass:: popart.RandomSetup
.. autoclass:: popart.RemoteSetup
.. autoclass:: popart.SerializeMatMuls
.. autoclass:: popart.StochasticRounding
.. autoclass:: popart.StreamingMemory
.. autoclass:: popart.SubgraphOutline
.. autoclass:: popart.BwdGraphInfo
.. autoclass:: popart.ExpectedConnectionType
.. autoclass:: popart.ExpectedConnection


Utility classes
---------------

Writer
^^^^^^

.. automodule:: popart.writer

Graph
^^^^^

.. autoclass:: popart.graphutils::CallStack
.. autoclass:: popart.graphutils::TensorAndCallStack

Region
^^^^^^

.. autoclass:: popart.view.Region

Error handling
^^^^^^^^^^^^^^

.. autoclass:: popart.ErrorSource
.. autoclass:: popart.OutOfMemoryException
.. autoclass:: popart.error
.. autoclass:: popart.internal_error
.. autoclass:: popart.memory_allocation_err
.. autoclass:: popart.popx.devicex_memory_allocation_err
.. autoclass:: popart.runtime_error

Debug context
^^^^^^^^^^^^^

.. autoclass:: popart.DebugContext
.. autoclass:: popart.DebugInfo
.. autoclass:: popart.OnnxOpDebugInfo
.. autoclass:: popart.OnnxVariableDebugInfo
.. autoclass:: popart.OpDebugInfo
.. autoclass:: popart.TensorDebugInfo

Attributes
^^^^^^^^^^

.. autoclass:: popart.Attributes

Void data
^^^^^^^^^

.. autoclass:: popart.ConstVoidData
.. autoclass:: popart.MutableVoidData

Input shape information
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: popart.InputShapeInfo

Profiling
^^^^^^^^^

.. autoclass:: popart.liveness::LivenessAnalyzer
.. autoclass:: popart.liveness::SubgraphPartitioner
.. autoclass:: popart.liveness::AliasZeroCopy
.. autoclass:: popart.liveness::Intervals
.. autoclass:: popart.liveness::ProducerInterval

Task information
^^^^^^^^^^^^^^^^

.. autoclass:: popart.TaskId

Type definitions
^^^^^^^^^^^^^^^^

.. doxygenfile:: names.hpp
  :sections: innernamespace typedef enum
.. autoclass:: popart.FwdGraphToBwdGraphInfo
.. autoclass:: popart.popx.PreparedCopyTensors
.. autoclass:: popart.popx.PreparedTensorInfos

Enums
^^^^^

.. autoclass:: popart.AccumulationType
.. autoclass:: popart.ActivationFunction
.. autoclass:: popart.AutoPad
.. autoclass:: popart.CollectiveOperator
.. autoclass:: popart.CommGroupType
.. autoclass:: popart.DeviceSelectionCriterion
.. autoclass:: popart.InitType
.. autoclass:: popart.MatMulPartialsType
.. autoclass:: popart.ResizeCoordinateTransformationMode
.. autoclass:: popart.ResizeMode
.. autoclass:: popart.ResizeNearestMode
.. autoclass:: popart.ScatterReduction
.. autoclass:: popart.TensorRemapType

Structs
^^^^^^^

.. autoclass:: popart.BranchInfo
.. autoclass:: popart.ClonedGraphMaps
.. autoclass:: popart.ConvParameters
.. autoclass:: popart.popx.OpxInAndOutIndex
.. autoclass:: popart.PTensorCmp
.. autoclass:: popart.ReplicatedTensorShardingOpInfo

Other classes
^^^^^^^^^^^^^
.. autoclass:: popart.BasicOptional
.. autoclass:: popart.ExchangeDescriptor
.. autoclass:: popart.GraphId
.. autoclass:: popart.LeakyReluOpBaseAttributes
.. autoclass:: popart.MultiConvOptions
.. autoclass:: popart.OpEquivIdCreator
.. autoclass:: popart.OpJsonSerialiser
.. autoclass:: popart.OpSerialiser
.. autoclass:: popart.OpSerialiserBase
.. autoclass:: popart.PriTaskDependency
.. autoclass:: popart.ReplicaEqualAnalysisProxy
.. autoclass:: popart.ReplicatedTensorShardingTracer
.. autoclass:: popart.TensorLocationInfo
.. autoclass:: popart.PyWeightsIO
.. autoclass:: popart.popx.InputCreatorCandidate
.. autoclass:: popart.popx.InputMultiCreatorCandidate
.. autoclass:: popart.popx.IsInfx
.. autoclass:: popart.popx.IsNaNx
.. autoclass:: popart.popx.ViewChanger
.. autoclass:: popart.popx.ViewChangers
.. autoclass:: popart.popx.ReplicatedGatherInScatterOutViewChanger
.. autoclass:: popart.popx.ReplicatedGatherOutScatterInViewChanger
.. autoclass:: popart.popx.serialization::Reader
