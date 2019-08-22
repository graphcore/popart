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
using PipelineStage = int64_t;
using StashIndex    = int64_t;

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

namespace view {
class Region;
using RegMap  = std::function<Region(const Region &)>;
using Regions = std::vector<Region>;
class Link;
class Chains;
using LowBounds = std::vector<int64_t>;
using UppBounds = std::vector<int64_t>;
} // namespace view

// A mapping from tensor name to the layout on each tile
using TensorInterval     = std::pair<size_t, size_t>;
using TensorIntervalList = std::vector<TensorInterval>;
using TensorTileMap = std::map<std::string, std::vector<TensorIntervalList>>;

// equivalent to decltype(&onnx::NodeProto().attribute(0))
using onnxAttPtr     = const onnx::AttributeProto *;
using NodeAttributes = google::protobuf::RepeatedPtrField<onnx::AttributeProto>;
using OnnxTensors    = std::map<TensorId, onnx::TensorProto>;
using Node           = onnx::NodeProto;
using OnnxTensorPtrs = std::map<TensorId, const onnx::TensorProto *>;
using OpsBeforeKey   = std::map<Op *, std::vector<Op *>>;

// Custom node attribute names
static constexpr const char *sVirtualGraphAttribute = "__ipu_number";
static constexpr const char *sInplaceOpNames        = "__inplace_op_names";
static constexpr const char *sInplaceOpPriorities   = "__inplace_op_priorities";
static constexpr const char *sRecomputeOutputAttribute =
    "__recompute_output_in_backward_pass";
static constexpr const char *sPartialsTypeAttribute  = "__partials_type";
static constexpr const char *sPipelineStageAttribute = "__pipeline_stage";

// The deliminator used in popart tensor names
static constexpr const char *sNameDelimiter = "/";

} // namespace popart

#endif
