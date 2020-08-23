// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_NAMES_HPP
#define GUARD_NEURALNET_NAMES_HPP

// TODO T7106 : determine what the cost of including these
// in every compilation unit is, consider moving to another header
#include <functional>
#include <map>
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
using TaskId       = std::string;
using OpName       = std::string;
using OpDomain     = std::string;
using OpType       = std::string;
using OpVersion    = unsigned;
using OpId         = int;
using ReturnPeriod = int;
// The position at which a Tensor is consumed by an Op
using InIndex = int;
// The position at which a Tensor is output by an Op
using OutIndex      = int;
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
using StashIndex        = int64_t;

// The identifier for a remote buffer
using RemoteBufferId = int64_t;
// The index within a remote buffer
using RemoteBufferIndex = int64_t;

// For decreasing the verbosity of MultiConv Op parameter names
using ConvInputs    = std::vector<TensorId>;
using ConvDilations = std::vector<int64_t>;
using ConvGroup     = int64_t;
using ConvPads      = std::vector<int64_t>;
using ConvStrides   = std::vector<int64_t>;

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
using TensorTileMap = std::map<std::string, std::vector<TensorIntervalList>>;

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
static constexpr const char *sOutputTensorLocationAttribute =
    "__output_tensor_location";
static constexpr const char *sOutputTypeAttribute      = "__output_type";
static constexpr const char *sExcludePatternsAttribute = "__exclude_patterns";
static constexpr const char *sSchedulePriority         = "__schedule_priority";
static constexpr const char *sIOTilesAttribute         = "__io_tiles";
static constexpr const char *sOutlineAttribute         = "__outline";

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
} // namespace popart

#endif
