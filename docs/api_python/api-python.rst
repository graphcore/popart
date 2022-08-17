PopART Python API
=================

Sessions
--------

.. autoclass:: popart.Session

Training session
^^^^^^^^^^^^^^^^

.. autoclass:: popart.TrainingSession
.. autoclass:: popart_core._TrainingSessionCore

Inference session
^^^^^^^^^^^^^^^^^

.. autoclass:: popart.InferenceSession
.. autoclass:: popart_core._InferenceSessionCore

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
.. autoclass:: popart.ExecutionPhaseIOSchedule
.. autoclass:: popart.ExecutionPhaseSchedule
.. autoclass:: popart.ExecutionPhaseSettings
.. autoclass:: popart.GradientTensorTrackingMethod
.. autoclass:: popart.Instrumentation
.. autoclass:: popart.IrSerializationFormat
.. autoclass:: popart.MeanReductionStrategy
.. autoclass:: popart.MergeVarUpdateType
.. autoclass:: popart.PyWeightsIO
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
.. autoclass:: popart.ReplicatedTensorSharding
.. autoclass:: popart.TensorInfo
.. autoclass:: popart.TensorLocation
.. autoclass:: popart.TensorStorage
.. autoclass:: popart.TileSet
.. automodule:: popart.tensorinfo

Optimizers
----------

.. autoclass:: popart.Optimizer
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
.. autoclass:: popart.builder._BuilderCore


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


Patterns
--------

.. autoclass:: popart.Patterns
    :show-inheritance:

.. autoclass:: popart.PreAliasPattern
    :show-inheritance:



Utility classes
---------------

Writer
^^^^^^

.. automodule:: popart.writer


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

Input shape information
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: popart.InputShapeInfo


Type definitions
^^^^^^^^^^^^^^^^

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

