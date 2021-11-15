// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_NAMES_HPP
#define GUARD_NEURALNET_NAMES_HPP

// TODO T7106 : determine what the cost of including these
// in every compilation unit is, consider moving to another header
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

// forward declaring onnx classes, done to
// help reduce the number of compilation units
// including the onnx headers.
namespace onnx {
class NodeProto;
class TensorProto;
class AttributeProto;
class GraphProto;
class ValueInfoProto;
class ModelProto;
class TypeProto;
} // namespace onnx

namespace google {
namespace protobuf {
template <class T> class RepeatedPtrField;
}
} // namespace google

namespace popart {

// Some type aliases, which hopefully
// make popart code clearer
using Shape        = std::vector<int64_t>;
using Rank         = int;
using TensorId     = std::string;
using DnfTensorIds = std::vector<std::set<TensorId>>;
using OpName       = std::string;
using OpDomain     = std::string;
using OpType       = std::string;
using OpVersion    = unsigned;
using OpId         = int;
using ReturnPeriod = int;
// The index of a subgraph for an Op.
using SubgraphIndex = int;
// The index of the subgraph part.
using SubgraphPartIndex = int;
// Identifies a part of an Opx grow function
using OpxGrowPartId = int;
// The position at which a Tensor is consumed by an Op
using InIndex = int;
// The position at which a Tensor is output by an Op
using OutIndex = int;
// The set of indices that have to be replica sharded together, and the outputs
// that will be replica sharded as a result
using ReplicatedTensorShardingIndices =
    std::set<std::pair<std::set<InIndex>, std::set<OutIndex>>>;
using PipelineCycle = int64_t;
using VGraphId      = int64_t;
// Virtual graphs (IPUs) are counted from 0
static constexpr const VGraphId unusedVGraphId = -1;
using PipelineStage                            = int64_t;
// Pipeline stages are counted from 0
static constexpr const PipelineStage unusedPipelineStage = -1;
using ExecutionPhase                                     = int64_t;
// Phase -1 is used for loading weights to phase 0, phase -2 is unused
static constexpr const ExecutionPhase unusedExecutionPhase = -2;
using BatchSerializedPhase                                 = int64_t;
// Phase -1 is used to initialize accumulators, phase -2 is unused
static constexpr const BatchSerializedPhase unusedBatchSerializedPhase = -2;

static constexpr const OpxGrowPartId unusedGrowPartId = -1;

/**
 * Used to describe the stochastic rounding which is applied to the output(s) of
 * an Op. See also `docs/notes/ir/attributes/stochasticroundingmethod.md`
 **/
enum StochasticRoundingMethod {
  /// Apply stochastic rounding with a replica-local seed. That is, stochastic
  /// rounding performed by an Op on one replica is nominally different to
  /// stochastic rounding performed by the same Op on another replica. Use this
  /// setting for Ops where you want to apply stochastic rounding but you cannot
  /// meet the condition of StochasticRoundingMethod::IdenticalSeeds. For
  /// example, this setting can be useful for gradient accumulation steps.
  DifferingSeeds = 1,
  /// Apply stochastic rounding with a RNG state (the value of
  /// poplar::getHwSeeds) that is identical across replicas. Use this option on,
  /// e.g., the weight update step to ensure that the weight tensor on each
  /// replica has stochastic rounding applied to it in the same way and there is
  /// no weight drift.
  ///
  /// REQUIREMENT: The ability to provide an RNG state (the value of
  /// poplar::getHwSeeds) that is identical on each replica relies on all Ops
  /// that use this setting to behave in a way that does not violate this
  /// property for Ops that follow it. More formally, you must only apply this
  /// setting to Ops for which you can guarantee that if the RNG state is the
  /// same across replicas before the Op is executed then the RNG state is still
  /// the same on all replicas after the Op is done executing. A typically
  /// sufficient (but not necessary) condition is that all input tensors of
  /// the Op have the same value across replicas.
  IdenticalSeeds = 2
};

using StashIndex = int64_t;

// The identifier for a remote buffer
using RemoteBufferId = int64_t;
// The index within a remote buffer
using RemoteBufferIndex = int64_t;

// Identifier for Random Reference Tensors
using RandomReferenceId = int64_t;

// For decreasing the verbosity of MultiConv Op parameter names
using ConvInputs    = std::vector<TensorId>;
using ConvDilations = std::vector<int64_t>;
using ConvGroup     = int64_t;
using ConvPads      = std::vector<int64_t>;
using ConvStrides   = std::vector<int64_t>;
using ConvTruncs    = std::vector<int64_t>;

using MultiConvInputs    = std::vector<ConvInputs>;
using MultiConvDilations = std::vector<ConvDilations>;
using MultiConvGroups    = std::vector<ConvGroup>;
using MultiConvPads      = std::vector<ConvPads>;
using MultiConvStrides   = std::vector<ConvStrides>;

// forward declaring several popart classes
class DataFlow;
class InputShapeInfo;
class Ir;
class Loss;
class Op;
class Optimizer;
class Pattern;
class Tensor;
class Tensors;
class Scheduler;
class TensorIndexMap;
class TensorInfo;
class ConstVoidData;
class MutableOutData;
class IStepIO;
class IWeightsIO;
class GradInOutMapper;
class InputMapWrapper;
class OutputMapWrapper;
class Vertex;
class TopoCons;
class Scope;
class Graph;
class GraphId;
enum class TensorType;
struct POpCmp;

namespace view {
class Region;
using Regions = std::vector<Region>;
using RegMap  = std::function<Regions(const Region &)>;
class Link;
class Chains;
using LowBounds = std::vector<int64_t>;
using UppBounds = std::vector<int64_t>;
} // namespace view

// A mapping from tensor name to the layout on each tile
using TensorInterval     = std::pair<size_t, size_t>;
using TensorIntervalList = std::vector<TensorInterval>;

// equivalent to decltype(&ONNX_NAMESPACE::NodeProto().attribute(0))
using onnxAttPtr = const ONNX_NAMESPACE::AttributeProto *;
using NodeAttributes =
    google::protobuf::RepeatedPtrField<ONNX_NAMESPACE::AttributeProto>;
using OnnxTensors    = std::map<TensorId, ONNX_NAMESPACE::TensorProto>;
using Node           = ONNX_NAMESPACE::NodeProto;
using OnnxTensorPtrs = std::map<TensorId, const ONNX_NAMESPACE::TensorProto *>;
using OpsBeforeKey   = std::map<Op *, std::vector<Op *>, POpCmp>;

// Custom node attribute names
static constexpr const char *sVirtualGraphAttribute     = "__ipu_number";
static constexpr const char *sExecutionPhaseAttribute   = "__execution_phase";
static constexpr const char *sExecutionContextAttribute = "__execution_context";
static constexpr const char *sInplaceOpNames            = "__inplace_op_names";
static constexpr const char *sInplaceOpPriorities = "__inplace_op_priorities";
static constexpr const char *sRecomputeOutputAttribute =
    "__recompute_output_in_backward_pass";
static constexpr const char *sPartialsTypeAttribute  = "__partials_type";
static constexpr const char *sAvailMemAttribute      = "__available_memory";
static constexpr const char *sPipelineStageAttribute = "__pipeline_stage";
static constexpr const char *sEnableConvDitheringAttribute =
    "__enable_conv_dithering";
static constexpr const char *sOutputTensorLocationAttribute =
    "__output_tensor_location";
static constexpr const char *sOutputTypeAttribute       = "__output_type";
static constexpr const char *sExcludePatternsAttribute  = "__exclude_patterns";
static constexpr const char *sSchedulePriority          = "__schedule_priority";
static constexpr const char *sTileSetAttribute          = "__tile_set";
static constexpr const char *sOutlineAttribute          = "__outline";
static constexpr const char *sDebugInfoId               = "__debug_info_id";
static constexpr const char *sExchangeStrategyAttribute = "__exchange_strategy";
static constexpr const char *sReplicatedStreamMode = "__replicated_stream_mode";
static constexpr const char *sCommGroupType        = "__comm_group_type";
static constexpr const char *sCommGroupSize        = "__comm_group_size";
static constexpr const char *sVariableSettings     = "__variable_settings";
static constexpr const char *sVariableRetrievalMode =
    "__variable_retrieval_mode";

static constexpr const char *sSerializeMatMulModeAttribute =
    "__serialize_matmul_mode";
static constexpr const char *sSerializeMatMulFactorAttribute =
    "__serialize_matmul_factor";
static constexpr const char *sSerializeMatMulPrecisionAttribute =
    "__serialize_matmul_precision";

// The deliminator used in popart tensor names
static constexpr const char *sNameDelimiter = "/";

static constexpr const char *sSerializeMatMulMode_None = "none";
static constexpr const char *sSerializeMatMulMode_InputChannels =
    "input_channels";
static constexpr const char *sSerializeMatMulMode_ReducingDim = "reducing_dim";
static constexpr const char *sSerializeMatMulMode_OutputChannels =
    "output_channels";

// Poplar stream name prefixes for streams used in host side reductions
static constexpr const char *gradientStoreStreamPrefix = "gradientStore__";
static constexpr const char *gradientLoadStreamPrefix  = "gradientLoad__";
static constexpr const char *weightLoadStreamPrefix    = "weightLoad__";

static constexpr const char *onnxDebugIdInputMetaDataKey = "__debug_id/input/";
static constexpr const char *sCollectiveOperator  = "__collectiveOperator";
static constexpr const char *sCollectiveCommGroup = "__collectiveCommGroup";
} // namespace popart

#endif
